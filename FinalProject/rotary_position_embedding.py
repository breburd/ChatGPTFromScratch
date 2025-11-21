import torch

"""
This class defines a custom rotary position embedding that defines an 'angle' between words rather than
the absolute position embeddings of the word within the sentence.
"""

import torch
import torch.nn as nn
import numpy as np


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        N = 10000
        inv_freq = 1. / (N ** (torch.arange(0, d_model, 2).float() / d_model))
        inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        position = torch.arange(max_seq_len).float()
        sinusoid_inp = torch.outer(position, inv_freq)
        self.register_buffer("cos", sinusoid_inp.cos())
        self.register_buffer("sin", sinusoid_inp.sin())

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin[:seq_len].view(1, seq_len, 1, -1)
        return apply_rotary_pos_emb(x, cos, sin)


# class RotaryPositionalEmbedding(torch.nn.Module):
#
#     def __init__(self, d_model, max_seq_len):
#         super().__init__()
#         # Create a rotation matrix.
#         self.rotation_matrix = torch.zeros(d_model, d_model)
#         for i in range(d_model):
#             for j in range(d_model):
#                 self.rotation_matrix[i, j] = torch.cos(i * j * 0.01)
#
#         # Create a positional embedding matrix.
#         self.positional_embedding = torch.zeros(max_seq_len, d_model)
#         for i in range(max_seq_len):
#             for j in range(d_model):
#                 self.positional_embedding[i, j] = torch.cos(i * j * 0.01)
#
#     def forward(self, x):
#         # Add the positional embedding to the input tensor.
#         x += self.positional_embedding
#
#         # Apply the rotation matrix to the input tensor.
#         x = x @ self.rotation_matrix


"""
References

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y. (2023). RoFormer: Enhanced transformer with Rotary Position 
    Embedding. Neurocomputing, 568, 127063. https://doi.org/10.1016/j.neucom.2023.127063
    
    https://medium.com/ai-insights-cobet/rotary-positional-embeddings-a-detailed-look-and-comprehensive-understanding-4ff66a874d83
    
    https://huggingface.co/blog/designing-positional-encoding
    
"""
