import torch
import math

torch.manual_seed(42)


d_k=6
seq_length = 3
batch_size = 1

q = torch.rand(batch_size, seq_length, d_k)
k = torch.rand(batch_size, seq_length, d_k)
v = torch.rand(batch_size, seq_length, d_k)
# print(q)
# print(10*"=")
# print(q.transpose(1, 2))
# print(10*"-")
# print(q.transpose(-2, -1))
# q.transpose(-2, -1) same as q.transpose(1, 2)


score = torch.softmax(q@k.transpose(1, 2)/ math.sqrt(d_k), dim=-1)
# shape of score (batch_size, seq_length, seq_length)
print("shape of score: ", score.size())
print("score: ", score)

z = score@v
print("shape of z: ", z.size())
print(z)




