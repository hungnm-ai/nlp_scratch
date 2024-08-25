import torch
import torch.nn as nn
from embedding import embedded_sentence



class CrossAttention(nn.Module):
	def __init__(self, d_in, d_out_kq, d_out_v):
		super().__init__()
		self.d_out_kq = d_out_kq
		self.Wq = nn.Parameter(torch.rand(d_in, d_out_kq))
		self.Wk = nn.Parameter(torch.rand(d_in, d_out_kq))
		self.Wv = nn.Parameter(torch.rand(d_in, d_out_v))

	def forward(self, x1, x2):
		queries_1 = x1 @ self.Wq
		keys_2 = x2 @ self.Wk
		values_2 = x2 @ self.Wv

		# attention scores between queries in decoder and keys in encoder
		attn_scores = torch.softmax(queries_1 @ keys_2.T/self.d_out_kq**0.5, dim=-1)
		context = attn_scores @ values_2
		return context

if __name__ == "__main__":
	d_in, d_out_kq, d_out_v = 3, 2, 4

	crossattn = CrossAttention(d_in, d_out_kq, d_out_v)

	first_input = embedded_sentence # (6, 3)
	second_input = torch.rand(8, d_in) # (8, 3)

	print("First input shape:", first_input.shape)
	print("Second input shape:", second_input.shape)

	context_vectors = crossattn(first_input, second_input)

	print(context_vectors)
	print("Output shape:", context_vectors.shape)

