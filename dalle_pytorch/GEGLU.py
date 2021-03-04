import torch.nn.functional as F
from torch import nn


class GEGLU(nn.Module):
    """
    Gaussian Error Linear Units (GELUs)
    a high-performing neural network activation function.
    The GELU activation function is xΦ(x), where Φ(x) the standard Gaussian cumulative distribution function.
    """
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)