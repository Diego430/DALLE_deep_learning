from functools import partial

from torch import nn

from .Attention import Attention
from .FeedForward import FeedForward
from .PreNorm import PreNorm
from .SequentialSequence import SequentialSequence
from .helperFunctions import *


class Transformer(nn.Module) :
	def __init__(
			self,
			*,
			dim,
			depth,
			seq_len,
			causal = True,
			heads = 8,
			dim_head = 64,
			ff_mult = 4,
			attn_dropout = 0.,
			ff_dropout = 0.,
			noncausal_attn_len = 0,
			sparse_attn = False,
	) :
		super().__init__()
		layers = nn.ModuleList([])
		sparse_layer = cast_tuple(sparse_attn, depth)

		for _, sparse_attn in zip(range(depth), sparse_layer) :
			attn_class = Attention

			layers.append(nn.ModuleList([
				PreNorm(dim, attn_class(dim, causal=causal, seq_len=seq_len, heads=heads, dim_head=dim_head,
										dropout=attn_dropout, noncausal_attn_len=noncausal_attn_len)),
				PreNorm(dim, FeedForward(dim, mult=ff_mult, dropout=ff_dropout))
			]))

		execute_type = SequentialSequence
		route_attn = ((True, False),) * depth
		attn_route_map = {'mask' : route_attn}

		self.layers = execute_type(layers, args_route=attn_route_map)

	def forward(self, x, **kwargs) :
		return self.layers(x, **kwargs)
