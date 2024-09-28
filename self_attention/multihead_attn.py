import torch
import torch.nn as nn
from embedding import embedded_sentence
from tokenizer import BaseTokenizer

from self_attention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out_qk: int, d_out_v: int, num_heads: int):
        super().__init__()

        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_qk, d_out_v) for _ in range(num_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


if __name__ == "__main__":

    d_in, d_out_kq, d_out_v, num_heads = 3, 2, 4, 2

    sa = MultiHeadAttention(d_in, d_out_kq, d_out_v, num_heads)
    context_matrix = sa(embedded_sentence)
    print("shape of context_matrix: ", context_matrix.size())
