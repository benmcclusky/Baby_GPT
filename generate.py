"""
Generate sample data from a pretrained model

"""

# ------------------------ Import Modules / Classes -------------------------

import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# ----------------------------- Hyperparameters -----------------------------

# Optional Prompt 
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"

# Generation hyperparameters
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability

# Device and Data Type
device = 'cpu'  # Device to run the model on (e.g., 'cpu', 'cuda')
device_type = 'cpu'  # Type of device (e.g., 'cpu', 'cuda')
dtype = 'float16'  # Data type for model parameters (e.g., 'float32', 'float16')

# Seed
seed_offset = 0  # Offset added to the random seed for reproducibility
torch.manual_seed(1337 + seed_offset)  # Setting the random seed for PyTorch

# Context Manager and Data Directory
ctx = nullcontext()  # Context manager for handling training state (nullcontext does nothing)
data_dir = os.getcwd()  # Current working directory, used as the data directory

# Initialization from Checkpoint
init_from = 'resume' 

# --------------------------- Initialising Model ------------------------------

# Load pretrained model 
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(data_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model.eval()

# Meta pickle is custom encoding specified in prepare.py. 
load_meta = False
if init_from == 'resume': # Check if already pretrained model
    meta_path = os.path.join(data_dir, 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # Specify custom encoding / decoding functions
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s] 
    decode = lambda l: ''.join([itos[i] for i in l])

model.to(device)

# Encode the beginning of the optional prompt
if start.startswith('FILE:'):
    # If the 'start' string starts with 'FILE:', it is treated as a file path to read the prompt from
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

# Encode the 'start' string into token IDs using the 'encode' function
start_ids = encode(start)

# Create a tensor 'x' containing the token IDs of the 'start' sequence
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# Run text generation for 'num_samples' samples
with torch.no_grad(): # Specify to pytorch it is not gradient calcuation
    with ctx: # Speicfy to use context manager
        for k in range(num_samples):
            # Generate text using the model starting from the 'x' sequence
            # Generate 'max_new_tokens' number of new tokens, controlled by 'temperature' and 'top_k'
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            
            # Decode the generated token IDs back to text and print the output
            print(decode(y[0].tolist()))
            print('---------------')


