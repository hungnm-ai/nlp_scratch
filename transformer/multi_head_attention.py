import math

import torch
import torch.nn as nn
from torch.nn import Linear


class MultiHeadAttention(nn.Module):
	def __init__(self, head, d_model, dropout=0.1):
		super(MultiHeadAttention, self).__init__()

		assert(d_model % head)

		self.d_k = d_model // head
		self.head = head
		self.Wq = Linear(in_features=d_model, out_features=d_model, bias=False)
		self.Wk = Linear(in_features=d_model, out_features=d_model, bias=False)
		self.Wv = Linear(in_features=d_model, out_features=d_model, bias=False)

		# feed forward
		self.Wo = Linear(in_features=d_model, out_features=d_model, bias=False)

		self.dropout = nn.Dropout(p=dropout)

		self.attn = None

	def attention(q, k, v, mask=None, dropout=None):

		d_k = q.size(-1)

		scores = q@k.transpose(-2, -1)/math.sqrt(d_k)
		if mask is not None:
			scores.masked_fill_(mask.bool(), -torch.inf)

		attn_weighteds = torch.softmax(scores, dim=-1)

		if dropout is not None:
			attn_weighteds = dropout(attn_weighteds)

		return attn_weighteds@v, attn_weighteds


	def forward(self, q, k, v, mask=None):
		"""shape of q, k and v is (batch, seq_length, d_model)
		"""
		bs = q.size(0)

		if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)


		

        # linear projections
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        # convert shape of q, k, v to (bs, seq_length, head, d_k)
        q = q.view(bs, -1, self.head, self.d_k)
        k = k.view(bs, -1, self.head, self.d_k)
        v = v.view(bs, -1, self.head, self.d_k)

        # convert to shape (bs, head, seq_length, d_k)
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        z, self.attn = self.attention(q, k, v, mask, dropout)

        # shape of z = (bs, head, seq_length, d_k)

        # convert shape z to (bs, seq_length, head*d_k) = (bs, seq_length, d_model)
        z = z.transpose(1, 2).contiguous().view(bs, -1, self.head*d_k)

        return self.Wo(z)





