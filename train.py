"""
Train model from scratch or continue to train pretrain model

"""

# ------------------------ Import Modules / Classes -------------------------

import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from model import GPTConfig, GPT

# ----------------------------- Hyperparameters -----------------------------

# Model Configuration
batch_size = 64  # Batch size for training and evaluation
block_size = 256  # Context of up to 256 previous characters
n_layer = 6  # Number of transformer layers in the model
n_head = 6  # Number of attention heads in each transformer layer
n_embd = 384  # Dimension of the embedding layer (size of the token embeddings)
dropout = 0.2  # Dropout probability to prevent overfitting
bias = False  # Whether to include bias in linear layers (e.g., attention)

# Optimizer Configuration
weight_decay = 1e-1  # Weight decay for regularization in the optimizer
learning_rate = 1e-3  # Learning rate for the optimizer
decay_lr = True  # Whether to decay the learning rate during training
max_iters = 5000  # Maximum number of training iterations
lr_decay_iters = 5000  # Number of iterations after which to decay the learning rate
min_lr = 1e-4  # Minimum learning rate (used in learning rate decay)
beta1 = 0.9  # Beta1 parameter for the AdamW optimizer
beta2 = 0.99  # Beta2 parameter for the AdamW optimizer
warmup_iters = 100  # Number of warm-up iterations for learning rate scheduling
gradient_accumulation_steps = 1  # Number of steps to accumulate gradients before updating the model
grad_clip = 1.0  # Clip gradients at this value to prevent exploding gradients

# Evaluation and Logging
eval_interval = 10  # Interval (in iterations) for evaluating the model during training
eval_iters = 3  # Number of iterations for each evaluation
log_interval = 5  # Interval (in iterations) for logging training progress
always_save_checkpoint = True  # Whether to always save model checkpoints

# Device and Data Type
device = 'cpu'  # Device to run the model on (e.g., 'cpu', 'cuda')
device_type = 'cpu'  # Type of device (e.g., 'cpu', 'cuda')
dtype = 'float16'  # Data type for model parameters (e.g., 'float32', 'float16')

# Training State and Seed
iter_num = 0  # Current iteration number (used to resume training)
best_val_loss = 1e9  # Best validation loss during training (used to track model performance)
eval_only = False  # If True, the script will exit after the first evaluation
seed_offset = 0  # Offset added to the random seed for reproducibility
torch.manual_seed(1337 + seed_offset)  # Setting the random seed for PyTorch

# Context Manager and Data Directory
ctx = nullcontext()  # Context manager for handling training state (nullcontext does nothing)
data_dir = os.getcwd()  # Current working directory, used as the data directory

# Initialization from Checkpoint
init_from = 'resume'  
# scratch - Initialize the model from scratch 
# resume - Option to initialize the model from a saved checkpoint

# ------------------------------- Load Data -----------------------------------

# Calculate the number of tokens processed in each iteration (used for learning rate scheduling)
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"Tokens per iteration will be: {tokens_per_iter:,}")

# Load training and validation data using memory-mapped arrays
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

# Combine training and validation data to determine the vocabulary size
combined_data = np.concatenate((train_data, val_data), axis=0)
vocab_size = len(np.unique(combined_data))

# Data loader function to get a batch of data for training or validation
def get_batch(split):
    # Choose the appropriate data (train or val) based on the split argument
    data = train_data if split == 'train' else val_data

    # Randomly select batch_size indices from the data array with a context of block_size
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Create input (x) and target (y) tensors for the selected indices
    # Each input tensor has block_size elements, and the target tensor has the following block_size elements.
    # The targets are shifted one position to the right compared to the inputs.
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    # Move the tensors to the specified device (e.g., 'cpu' or 'cuda')
    x, y = x.to(device), y.to(device)
    return x, y

# --------------------------- Initialising Model ------------------------------

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout
)  # start with model_args from command line

if init_from == 'scratch':
    # Initialize a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from checkpoint")
    # Load Pytorch model (.pt) from checkpoint
    ckpt_path = os.path.join(data_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Force these config attributes to be equal; otherwise, we can't even resume training.
    # The rest of the attributes (e.g., dropout) can stay as desired from the command line.
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    # Create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # Bug fix to remove unwanted prefix from checkpoints 
    unwanted_prefix = '_orig_mod.'
    state_dict = checkpoint['model']
    # Removes the prefix from keys of the state_dict in place
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    # Load training information 
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# Crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size 

model.to(device)  # Move the model to the specified device (e.g., 'cpu' or 'cuda')

# Initializes a GradScaler, which is used for mixed-precision training.
# Mixed-precision training combines both single-precision (float32) and half-precision (float16) arithmetic to speed up training and reduce memory usage while maintaining numerical stability.
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# AdamW Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# Load previous optimizer if resuming training
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None  # Free up memory


@torch.no_grad() # Decorator is used to temporarily disable gradient computation in PyTorch for this function, used to reduce computation time
def estimate_loss():

    """ Estimate loss over either 'train' or 'val' split using 'eval_iters' # batches """

    out = {}  # Dictionary to store the mean loss for 'train' and 'val' splits
    model.eval()  # Set the model to evaluation mode (no gradient updates)
    
    # Iterate through the 'train' and 'val' splits
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)  # Tensor to store losses for multiple batches
        
        # For each split, compute the loss for 'eval_iters' number of batches
        for k in range(eval_iters):
            X, Y = get_batch(split)  # Get a batch of input and target data
            with ctx: # ctx is the context manager used to allocate compute resources effectively
                logits, loss = model(X, Y)  # Forward pass to get logits and loss
            losses[k] = loss.item()  # Store the loss for this batch
        
        # Compute the mean loss for the split and store it in the 'out' dictionary
        out[split] = losses.mean()
    
    model.train()  # Set the model back to training mode (enable gradient updates)
    return out  # Return the dictionary containing mean losses for 'train' and 'val'


def get_lr(it):

    """ Learning rate decay scheduler (cosine with warmup) """

    # 1) Linear warmup for 'warmup_iters' steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    
    # 2) If 'it' > 'lr_decay_iters', return the minimum learning rate
    if it > lr_decay_iters:
        return min_lr
    
    # 3) In between, use cosine decay down to the minimum learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 'coeff' ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# ------------------------------- Training -----------------------------------

# Initialize the first batch for training
X, Y = get_batch('train')

t0 = time.time()
local_iter_num = 0  # Number of iterations in the lifetime of this process
running_mfu = -1.0  # Running memory footprint utilization

while True:
    # Determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # If loss all time best or always save checkpoint, save model to ckpt.pt
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {data_dir}")
                torch.save(checkpoint, os.path.join(data_dir, 'ckpt.pt')) 

    # If set eval_only = True, evaluates the loss on the first iteration and ends training
    if iter_num == 0 and eval_only:
        break

    # Forward, backward, and update with optional gradient accumulation to simulate larger batch size
    # Using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # Scale the loss to account for gradient accumulation

        # Immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')

        # Backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    # Clip the gradient to improve model stability 
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # Flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # Timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0: 
        # Get loss as a float. Note: this is a CPU-GPU sync point.
        # Scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps

        if local_iter_num >= 5:  # Let the training loop settle a bit
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    local_iter_num += 1

    # Termination conditions
    if iter_num > max_iters:
        break