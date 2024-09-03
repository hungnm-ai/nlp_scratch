"""
Reference:
https://adeveloperdiary.com/data-science/deep-learning/nlp/coding-transformer-model-from-scratch-using-pytorch-part-1/#positional-encoding
https://nlp.seas.harvard.edu/2018/04/03/attention.html
https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
"""

# https://nlp.seas.harvard.edu/annotated-transformer/#background

import torch
import math
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout: float = 0.1, max_len:int =5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)

		div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10_000) / d_model))


		# outer product between 2 vector will creat a matrix
		# position = (max_len, 1) , div_term = (d/2, )
		# position * div_term = (max_len, d/2)
		pe[:, 0::2] = torch.sin(position * div_term) 
		pe[:, 1::2] = torch.cos(position * div_term)

		# create batch size -> (1, max_len, d_model)
		pe = pe.unsqueeze(0)

		# sử dụng để lưu trữ tensor trong module mà không cần là 1 tham số (không cần learn)
		# nó tự động chuyển đổi giữa các device khi model chuyển đổi,
		# hoặc khi model được lưu xuống disk thì nó cũng được lưu xuống disk
		# khi load model nó cũng được load
		self.register_buffer("pe", pe)

	def forward(self, x):
		x += Variable(self.pe[:, :x.size(1)], requires_grad=False)
		
		return self.dropout(x)



if __name__ == "__main__":
	position_encoding = PositionalEncoding(d_model=16, max_len=32)
