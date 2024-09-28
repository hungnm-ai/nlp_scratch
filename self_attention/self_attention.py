import torch
import torch.nn as nn
from embedding import embedded_sentence


class SelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out_qk: int, d_out_v: int):
        super().__init__()
        self.d_in = d_in
        self.d_out_qk = d_out_qk
        self.d_out_v = d_out_v

        self.Wq = nn.Parameter(torch.rand(d_in, d_out_qk))
        self.Wk = nn.Parameter(torch.rand(d_in, d_out_qk))
        self.Wv = nn.Parameter(torch.rand(d_in, d_out_v))

    def forward(self, x):
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv

        attn_scores = torch.softmax(q @ k.T / self.d_out_qk**0.5, dim=-1)

        contex_vector = attn_scores @ v
        return contex_vector


if __name__ == "__main__":
    from embedding import embedded_sentence

    d_in, d_out_kq, d_out_v = 3, 2, 4

    sa = SelfAttention(d_in, d_out_kq, d_out_v)
    print(sa(embedded_sentence))
