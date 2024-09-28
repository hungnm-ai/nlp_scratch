from typing import Dict, List

import torch
import torch.nn.functional as F
from tokenizer import BaseTokenizer

torch.manual_seed(123)


sentence = "Life is short, eat dessert first"

dc = {s: i for i, s in enumerate(sorted(sentence.replace(",", "").split()))}

print("dc: ", dc)

tokenizer = BaseTokenizer(dc)

token_ids = tokenizer.encode(sentence)
token_ids = torch.tensor(token_ids)

print("token_ids: ", token_ids, "shape: ", token_ids.size())


### Embedding
d = 3
vocab_size = 50_000
embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)

print("embedding layer: ", embed)

embedded_sentence = embed(token_ids).detach()
if __name__ == "__main__":
    print("shape of embedded_sentence: ", embedded_sentence.size())
    print("embedded_sentence: ", embedded_sentence)

    # Wq (d, dk), Wk = (d, dk), Wv = (d, dv) is the parameters that adjusted while training
    # q = x*Wq
    # k = x*Wk
    # v = x*Wv

    # init
    dq, dk, dv = 2, 2, 4
    Wq = torch.nn.Parameter(torch.rand(d, dq))
    Wk = torch.nn.Parameter(torch.rand(d, dk))
    Wv = torch.nn.Parameter(torch.rand(d, dv))

    x1 = embedded_sentence[1]
    q1 = x1 @ Wq  # @ is torch.matmul = matrix multiply
    k1 = x1 @ Wk
    v1 = x1 @ Wv

    # shape of q1 = (1, 3)*(3,2) = (1, 2)
    # shape of k1 = (1, 3)*(3,2) = (1, 2)
    # shape of v1 = (1, 3)*(3,4) = (1, 4)
    print("shape of x1: ", x1.size())
    print("shape of q1: ", q1.size())
    print("shape of k1: ", k1.size())
    print("shape of v1: ", v1.size())

    keys = torch.matmul(embedded_sentence, Wk)
    values = embedded_sentence @ Wv
    # keys = (num_tokens, 2)
    # values = (num_tokens, 4)
    print("shape of keyes: ", keys.size())
    print("shape of values: ", values.size())

    # next step calculates unnormalized attention weight between q1 and keys
    omega1 = q1 @ keys.T
    # omega = 1, 6
    print("shape of omega1: ", omega1.size())
    print("omega1: ", omega1)

    # mormalize attention weight
    attention_weights_1 = F.softmax(omega1 / dk**0.5, dim=-1)
    print("attention_weights_1: ", attention_weights_1)

    # compute the context vector z1
    z1 = attention_weights_1 @ values
    # z1 = (1, 6) x (6, 4) = (1, 4)
    print("shape of z1: ", z1.size())
