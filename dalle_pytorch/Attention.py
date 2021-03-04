import torch
from einops import rearrange
from torch import nn

from .helperFunctions import *


class Attention(nn.Module) :
	"""
	enhances the important parts of the input data and fades out the rest
	"""

	def __init__(self, dim, seq_len, causal = True, heads = 8, dim_head = 64, dropout = 0., noncausal_attn_len = 0) :
		super().__init__()
		inner_dim = dim_head * heads
		self.heads = heads
		self.seq_len = seq_len
		self.scale = dim_head ** -0.5

		self.causal = causal
		self.noncausal_attn_len = noncausal_attn_len

		self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
		self.to_out = nn.Sequential(
			nn.Linear(inner_dim, dim),
			nn.Dropout(dropout)
		)

	def forward(self, x, mask = None) :
		b, n, _, h, device = *x.shape, self.heads, x.device
		qkv = self.to_qkv(x).chunk(3, dim=-1)
		q, k, v = map(lambda t : rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

		dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
		mask_value = -torch.finfo(dots.dtype).max

		if exists(mask) :
			mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
			dots.masked_fill_(~mask, mask_value)
			del mask

		if self.causal :
			i, j = dots.shape[-2 :]
			mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()

			if self.noncausal_attn_len > 1 :
				ind = slice(0, self.noncausal_attn_len)
				mask[ind, ind] = False

			dots.masked_fill_(mask, mask_value)

		attn = dots.softmax(dim=-1)

		out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
		out = rearrange(out, 'b h n d -> b n (h d)')
		out = self.to_out(out)
		return out
