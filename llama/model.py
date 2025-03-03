import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclasses
from typing import Optional

class ModelArgs:
    dim = 4096
    n_layers = 32
    n_heads = 32 # Number of heads for the queries
    n_kv_heads = None # Number of heads for K and V
    vocab_size = -1 # This will be set when we load the tokenizer
    multiple_of = 256
    ffn_dim_multiplier = None
    norm_eps = 1e-3

    # Needed for KV cache
    max_batch_size = 32
    max_seq_len = 2048

    device = None

def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta):
    # As written in the paper, the dimension of the encoding must be even.
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameters
    # According to the formula theta_i = 10000 ^ (-2(i-1)/dim) for i = [1, 2,... dim/2 ]
    # Shape (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # construct the position (the 'm' parameter)
    # shape (Seq_len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each postion using the product
    # Shape (Seq_Len) outer_product * (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute the complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follows
    freq_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freq_complex

def apply_rotary_embedding(x, freq_complex, device):
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (Seq_Len, Head_Dim / 2) -> (1, Seq_Len, 1, Head_Dim / 2)
    freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)
    # (B, Seq_Len, H, Head_Dim / 2) * (1, Seq_Len, 1, Head_Dim / 2) = (B, Seq_Len, H, Head_Dim / 2)
    x_rotated = x_complex * freq_complex
    # (Seq_Len, Head_Dim / 2) -> (B, Seq_Len, H, Head_Dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim / 2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device=device)

class Transformer(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)
    
    def forward(self, tokens, start_pos):
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        # (B, Seq_len) -> (B, Seq_Len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the position [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply all the encoder layers 
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
