import torch
import math

torch.manual_seed(42)






def attention(q, k, v, d_k):

	score = torch.softmax(q@k.transpose(-2, -1)/ math.sqrt(d_k), dim=-1)
	# shape of score (batch_size, seq_length, seq_length)
	z = score@v
	return z


### Single self-attention

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

z = attention(q, k, v, d_k)
print("Shape of Z (single attention): ", z.size())
print("Single attention: ")
print(z)

################


### Multi-head attention
d_model = 6
head = 2
d_k = d_model // head

q = torch.rand(batch_size, seq_length, d_model)
k = torch.rand(batch_size, seq_length, d_model)
v = torch.rand(batch_size, seq_length, d_model)

q = q.view(batch_size, seq_length, head, d_k)
k = k.view(batch_size, seq_length, head, d_k)
v = v.view(batch_size, seq_length, head, d_k)

q = q.transpose(1, 2) # size = (batch_size, head, seq_length, d_k)
k = k.transpose(1, 2) # size = (batch_size, head, seq_length, d_k)
v = v.transpose(1, 2) # size = (batch_size, head, seq_length, d_k)

# calucate self-attention for multi-head
z_heads = attention(q, k, v, d_k) # size = (batch_size, head, seq_length, seq_length)

print("Shape of Z (Multi-head attention): ", z_heads.size())
print("Multi-head attention: ")
print(z_heads)


# Masked Multi-head Self-Attention

def attention_masked(q, k, v, d_k):
	
	attention_scores = q@k.transpose(-2, -1)/ math.sqrt(d_k)

	mask = torch.triu(torch.ones(attention_scores.size()), diagonal=1)

	attention_scores.masked_fill_(mask.bool(), -torch.inf)

	score = torch.softmax(attention_scores/d_k**0.5, dim=-1)
	print("attn_scores: ", score)


	# shape of score (batch_size, seq_length, seq_length)
	z = score@v
	return z



# calucate self-attention for multi-head
z_heads = attention_masked(q, k, v, d_k) # size = (batch_size, head, seq_length, seq_length)

print("Shape of Z (Masked Multi-head attention): ", z_heads.size())
print("Multi-head attention: ")
print(z_heads)





