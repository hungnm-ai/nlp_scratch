import torch
import torch.nn as nn




class CausalAttention(nn.Module):
	def __init__(self, d_in, d_qk, d_v):
		super().__init__()
		self.d_qk = d_qk
		self.Wq = nn.Parameter(torch.rand(d_in, d_qk))
		self.Wk = nn.Parameter(torch.rand(d_in, d_qk))
		self.Wv = nn.Parameter(torch.rand(d_in, d_v))

	def forward(self, x):
		queries = x @ self.Wq
		keys = x @ self.Wk
		values = x @ self.Wv

		attn_scores = queries @ keys.T

		

		### avoid attent the tokens in feature

		# Approach 1
		# attn_weights = torch.softmax(attn_scores / self.d_qk ** 0.5, dim=-1)
		# triagle_matrix = torch.tril(torch.ones(attn_weights.size()))

		# attn_weights = attn_weights*triagle_matrix

		# attn_weights /= torch.sum(attn_weights, dim=-1,keepdim=True)

		# Approach 2
		mask = torch.triu(torch.ones(attn_scores.size()), diagonal=1) # triu stands for triagle upper
		masked = attn_scores.masked_fill(mask.bool(), -torch.inf)

		attn_weights = torch.softmax(masked/self.d_qk**0.5, dim=-1)

		print("attn_weights: ", attn_weights)

		context_vectors = attn_weights @ values

		return context_vectors

if __name__ == '__main__':
	from embedding import embedded_sentence

	d_in, d_out_kq, d_out_v = 3, 2, 4

	attention = CausalAttention(d_in, d_out_kq, d_out_v)
	print(attention(embedded_sentence))