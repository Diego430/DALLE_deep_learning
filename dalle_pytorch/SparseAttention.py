import torch
import torch.nn.functional as F
from einops import rearrange

from .Attention import Attention
from .helperFunctions import *


class SparseAttention(Attention):
    def __init__(
        self,
        *args,
        block_size = 16,
        num_random_blocks = None,
        sparse_attn_global_indices = [],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        from deepspeed.ops.sparse_attention import SparseSelfAttention, VariableSparsityConfig
        self.block_size = block_size

        num_random_blocks = default(num_random_blocks, self.seq_len // block_size // 4)
        global_blocks = uniq(map(lambda t: t // block_size, sparse_attn_global_indices))

        self.attn_fn = SparseSelfAttention(
            sparsity_config = VariableSparsityConfig(
                num_heads = self.heads,
                block = self.block_size,
                num_random_blocks = num_random_blocks,
                global_block_indices = global_blocks,
                attention = 'unidirectional' if self.causal else 'bidirectional'
            ),
            max_seq_length = self.seq_len,
            attn_mask_mode = 'add'
        )

    def forward(self, x, mask = None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        remainder = n % self.block_size
        mask = default(mask, lambda: torch.ones(b, n, device = device).bool())

        if remainder > 0:
            padding = self.block_size - remainder
            x = F.pad(x, (0, 0, 0, padding), value = 0)
            mask = F.pad(mask, (0, padding), value = False)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        key_pad_mask = None
        if exists(mask):
            key_pad_mask = ~mask

        attn_mask = None
        if self.causal:
            i, j = q.shape[-2], k.shape[-2]
            mask = torch.ones(i, j, device = device).triu_(j - i + 1).bool()
            attn_mask = torch.zeros(i, j, device = device).to(q)
            mask_value = -(torch.finfo(q.dtype).max / 2)
            attn_mask.masked_fill_(mask, mask_value)

            if self.noncausal_attn_len:
                ind = slice(0, self.noncausal_attn_len)
                attn_mask[ind, ind] = 0.

        out = self.attn_fn(q, k, v, attn_mask = attn_mask, key_padding_mask = key_pad_mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out[:, :n]