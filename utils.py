from typing import (
    Tuple
)
import torch as T
from torch import nn

class Multiply(nn.Module):
    def __init__(self, *shape: int, bias: bool=False):
        super().__init__()
        self.shape = shape
        self.weight = nn.Parameter(T.empty(self.shape))
        if bias:
            self.bias = nn.Parameter(T.empty(self.shape))
        else:
            self.bias = None

        self.init_weights_()

    def init_weights_(self):
        nn.init.uniform_(self.weight, -1.0, 1.0)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1.0, 1.0)

    def forward(self, x: T.Tensor) -> T.Tensor:
        if self.bias is not None:
            return T.addcmul(self.bias, x, self.weight)
        else:
            return T.mul(x, self.weight)
