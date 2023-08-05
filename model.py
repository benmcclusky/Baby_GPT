""" 
Classic transformer architechture built using Pytorch 

"""

# ------------------------ Import Modules / Classes -------------------------

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# ------------------------------ Transformer  -------------------------------

class LayerNorm(nn.Module): # Defines a new class which inherits the nn.Module from PyTorch. 

    """ Custom LayerNorm module but with an optional bias"""

    def __init__(self, ndim, bias): 
        # The constructor of the LayerNorm class, initialising the object. Takes in ndim (number of dimensions) and bias (boolean Yes/No)

        super().__init__() # The super() calls the __init__ of the parent class nn.Module 

        # Creates two trainable parameters: weight and bias. 
        self.weight = nn.Parameter(torch.ones(ndim))
        # Initialises weights as a 1 dimensional tensor of shape (ndim)
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        # Initialises bias as 1 dimensional tensor of shape (ndim). However if bias is set to false, self.bias is set to None not creating a new parameter 


    def forward(self, input):

        """ Defines the forward pass of the LayerNorm module, takes in an input tensor and applies the F.layer_norm function from PyTorch. """

        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    
        # weight.shape is used as a scaling factor for normalisation. Normalisation divides the input values by the square root of the varience. 
        # However when the variance is too small, dividing by it can lead to instability. Therefore 1e-5 is added to the denominator to avoid this. 



class CasualSelfAttention(nn.Module):

    def __init__(self, config):

        """ Constructor of the self-attention class, importing model
        hyper parameters from a config file"""

        super().__init__() 
    
        # assert statement checks that the embeddding dimensions is divisible 
        # by the number of attention heads before proceeding 
        assert config.n_embd % config.n_head == 0 
        
        # Performs the input tensor projection into key, query and value for all heads but in a batch
        # Projects tensor of shape n_embd with learnable weight matrix of shape (3 * n_embd, n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Output linear projection after the self attention mechanism
        # Captures relevant infomation from the self-attention mechanism
        # Projects tensor of shape n_embd with learnable weight matrix of shape (n_embd, n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        # nn.Dropout is used to randomly zero out elements of the input tensor during training which helps prevent overfitting
        # config.dropout is the probablity of an element being zeroed out 
        self.attn_dropout = nn.Dropout(config.dropout) # dropout applied during attn
        self.resid_dropout = nn.Dropout(config.dropout) # dropout applied after linear transformation after self-attention

        # Initialising hyperparameters from config file 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # checks if PyTorch module has 'scaled_dot_product_attention'
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.tril creates tensor with top right trangle of 1 values 
            # mask is stored as a buffer named 'bias' with a shape (1,1, block_size, block size)
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
            

    def forward(self, x):

        """ Forward pass of the self-attention mechanism, takes an input tensor (x) """

        # B - batch size (number of samples processed at a time during training)
        # T - sequence length or time dimension
        # C - embedding dimensionality (n_embd) or channels (C)
        # labeling 3D matrices (row, columns, pages)
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # uses c_attn function to produce output tensor of shape (B, T, 3 * C)
        # splits by the 3rd dimension (dim = 2) every n_embd to create q,k,v tensors of dimensions (B, T, C)

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)

        # Splits C into respective attention heads 
        # Reshapes tensor from (B, T, C) to (B, n_heads, T, C // n_heads) refered to as (B, nh, T, hs)
        # C // n_heads (hs) is the chosen embedding dimension

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            # creates scaled dot product
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # if bias is not none, creates top right triangle for last 2 dimensions of 0's, set to -inf
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            # applies softmax from torch.nn.functional
            att = F.softmax(att, dim=-1)
            # applies dropout 
            att = self.attn_dropout(att)
            # uses att as weights and multiplies by values
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # transpose (1,2), swaps the 1 & 2 dimension (nh,T)
        # contiguous() returns an identical tensor but with a contiguous memory layour, making it more efficient
        # view(B, T, C) concatenates all the (B, nh, T, hs) tensors (nh x hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection to capture relevant infomation from attention heads
        # applies dropout the second time (resid_dropout)
        y = self.resid_dropout(self.c_proj(y))

        return y
    

class MLP(nn.Module):

    """ Multi-Layer Perceptron (Feed Forward Neural Network)"""

    def __init__(self, config):
        
        "Initiliser importing the config file"

        super().__init__()
        
        # defines attribute which represents a linear transformation layer to a hidden layer of size 4 x n_embd
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # apply guassian error linear unit (GELU) element wise
        self.gelu    = nn.GELU()
        # linear projection back to n_embd size
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        # applies dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):

        """ Pass an input tensor (x) through the forward pass of the MLP"""

        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
    


class Block(nn.Module):

    """ The full forward pass for the stackable layers (blocks) of the transformer encoding stage"""

    def __init__(self, config):

        """Initialises the stages required as attributes"""

        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):

        """Applies the stages in the transformer including the residual connection"""

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    



# The @dataclass decorater automatically generates common methods such as  __init__ & __repr__
# By specifying this you do not need to write these methods explicitly

@dataclass 
class GPTConfig:
    # Default values for all these hyperparameters, replaced by config file do I really need this?
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    """ Generative Pre-trained Transformer object"""

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Creates transformer dictionary object, imports the hyperparameters from the config file.
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # embedding layer for word (token) embeddings of size (vocab_size, n_embd)
            wpe = nn.Embedding(config.block_size, config.n_embd), # embedding layer for positional encodings of size (block_size, n_embd). Note - sin/cosine formulas not used
            drop = nn.Dropout(config.dropout), # dropout layer with specified dropout rate
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # h is a list of block modules (transformer blocks), n_layer sets the number of transformer blocks
            ln_f = LayerNorm(config.n_embd, bias=config.bias), # layer normalisation module with n_embd as input size
        ))

        # linear tranformation layer that maps output of transformer from n_embd to vocab size
        # This is used to map it back to the probability distribution over the vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # This ties the weights of the wte layer and the lm_head layer so thats the weights are reused
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights according to per GPT-2 paper recomendations
        # uses a scaled normal distribution for normal weights (mean = 0.0, std = 0.02)
        self.apply(self._init_weights)
        # apply special scaled init to the residual projection parameters, per GPT-2 paper (mean=0.0, std=0.02/math.sqrt(2 * config.n_layer)
        # this is done to ensure proper gradient propergation
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("Number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters()) # numel() counts number of elements in each parameter tensor
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel() # subtracting position embeddings
        return n_params
    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear): # checks if module is a fully connected layer (nn.Linear)
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # Initialises weights
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # if model has a bias term, initialises them as zero
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # if module is an instance of embedding, initialises the weights accordingly
                                  
    
    def forward(self, idx, targets=None): 
        # Forward pass of the GPT object, idx = input tensor of token indices, targets is an optional tensor for the correct answers for training 

        device = idx.device # Get the device of the input tensor (CPU)
        b, t = idx.size() # Get the batch size and sequence length of the input tensor 

        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # Create pos tensor of shape (t), represents positions in the input sequence, values from 0 to t-1
        assert torch.max(idx) < self.transformer.wte.weight.shape[0], "Invalid token index detected!"
       
        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd) using input tensor
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd) using pos tensor
        x = self.transformer.drop(tok_emb + pos_emb) # add position embedding to token embeddings, then applying dropout function
        for block in self.transformer.h: # Loop through each block in list of block modules h (transformer blocks)
            x = block(x)
        x = self.transformer.ln_f(x) # Apply layer normalisation 

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x) # apply linear tranformation to get the logits (tensor of vocab size (probability distribution))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) # Compute cross-entropy loss (ignoring padding tojens (index - 1))
            # padding tokens (e.g. <PAD>) are added to sequences shorter than the block size, important to remove these
            # computes loss for all logits in the sequence as target is available
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
            
        return logits, loss
    
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters, named_parameters is a module function
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad therefore not trainable 
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2] # Parameters with greater than 2 dimensions subject to weight decay
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay}, # setting weight decay rates
            {'params': nodecay_params, 'weight_decay': 0.0} 
        ]
        # Weight decay is a regularisation technique to prevent individual weights getting too large therefore becoming too sensitive to input data
        # Adding a weight decay term to the loss function optimises for small weights as well as minimising loss
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda' # only apply fused if cuda / GPU available 
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args) # AdamW optimiser, feed optim_groups which is dict giving decay rates 
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


    @torch.no_grad() # Specifies to torch that this doesnt require optimising therefore increasing efficiency
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens): # Loop through to generate max_new_tokens number of tokens
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # self(idx_cond) invokes the __call__ method of the GPT class, returns logits and loss. 
            # By using logits, _. This creates the logits variable and discards the loss by assigning it to _
            logits, _ = self(idx_cond)
            # pluck the logits at the final positon and scale by desired temperature
            # temperature controls the randomness of the generated tokens, higer makes the distribution more uniform. Lower makes it more peaky and deterministic 
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options. For example below the 'k'th largest value, set to -inf. So only ever consider the top 3 values etc 
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx # returns the running sequence followed by the newly generated tokens
    