"""
Note: Embedding layer same as linear layer that perform matrix multiply,
but embedding is efficency.
Reference: https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/03_bonus_embedding-vs-matmul/embeddings-and-linear-layers.ipynb
 

"""

import torch
import torch.nn as nn

from self_attention.tokenizer import BaseTokenizer


class InputEmbedding(nn.Module):
    """
    Args:
            vocab_size (int): Vocab size is the number of tokens
            d_model (int): Embedding dimensiton
    """

    def __init__(self, vocab_size: int, d_model):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        embedded = self.embedding(x) * (self.d_model**0.5)

        return embedded


if __name__ == "__main__":
    torch.manual_seed(123)
    token_ids = torch.tensor([0, 2, 1, 4, 1])
    input_embedding = InputEmbedding(12, 4)
    input_embedded = input_embedding(token_ids)
    print("input_embedded: ", input_embedded)
