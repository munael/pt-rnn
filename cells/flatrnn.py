# from __future__ import __annotations__, annotations
from typing import (
    Tuple,
    List,
    Optional,
    Dict,
    Callable,
    Union,
    cast,
    NamedTuple,
)
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import torch as T
from torch import nn, jit
from torch.nn import functional as F

from torch import Tensor

from ..rnn_base import (
    IRecurrentCell,
    IRecurrentCellBuilder,
    RecurrentLayer,
    BasicRecurrentLayerStack,
)
from .. import utils as uu

__all__ = [
    'FlatRNN',
    'FlatRNN_Cell'
]

ACTIVATIONS: Dict[str, nn.Module] = {
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'hard_tanh': nn.Hardtanh(),
    'relu': nn.ReLU(),
}

@dataclass
class FlatRNN_Cell_Builder(IRecurrentCellBuilder):
    hidden_size : int
    activation  : Union[str, nn.Module] = 'relu'

    vertical_dropout  : float = 0.0
    recurrent_dropout : float = 0.0

    # enable_second_memory : bool = False
    linear_recurrence : bool = False

    input_kernel_initialization : str = 'xavier_uniform'

    def make(self, input_size: int):
        return FlatRNN_Cell(input_size, self)

class FlatRNN_Cell(IRecurrentCell):
    def __repr__(self):
        args = ', '.join([
            f'in: {self.Dx}',
            f'hid: {self.Dh}',
            f'rdo: {self.recurrent_dropout_p}',
            f'vdo: {self.vertical_dropout_p}'
        ])
        return f'{self.__class__.__name__}({args})'

    def __init__(
            self,
            input_size: int,
            args: FlatRNN_Cell_Builder,
    ):
        super().__init__()
        self._args = args
        self.Dx = input_size
        self.Dh = args.hidden_size

        ## Parameters
        self.recurrent_kernel = uu.Multiply(self.Dh, bias=False)
        self.input_kernel     = nn.Linear(self.Dx, self.Dh, bias=True)
        ## END Parameters

        self.recurrent_dropout_p = args.recurrent_dropout or 0.0
        self.vertical_dropout_p  = args.vertical_dropout or 0.0
        
        self.recurrent_dropout = (
            nn.Dropout(self.recurrent_dropout_p) if self.recurrent_dropout_p > 0.0
            else nn.Identity()
        )
        self.vertical_dropout  = (
            nn.Dropout(self.vertical_dropout_p)  if self.vertical_dropout_p  > 0.0
            else nn.Identity()
        )

        if isinstance(args.activation, str):
            self.activation = ACTIVATIONS[args.activation]
        else:
            self.activation = args.activation

        self.reset_parameters_()

        self._dummy = nn.Parameter(T.ones(1))

    def device(self):
        return self._dummy.device

    # @jit.ignore
    def get_recurrent_weights(self) -> T.Tensor:
        ww = self.recurrent_kernel.weight
        return ww

    # @jit.ignore
    def get_input_weights(self) -> Tuple[T.Tensor, T.Tensor]:
        W = self.input_kernel.weight
        b = self.input_kernel.bias
        return W, b

    @jit.ignore
    def reset_parameters_(self):
        rec_w = self.get_recurrent_weights()
        # nn.init.uniform_(rec_w)

        in_W, in_b = self.get_input_weights()
        nn.init.zeros_(in_b)
        nn.init.xavier_uniform_(in_W)

    @jit.export
    def get_init_state(self, input: Tensor=None) -> Tuple[Tensor, Tensor]:
        h0 = T.zeros(1, self.Dh, device=self.device())
        c0 = T.zeros(1, self.Dh, device=self.device())
        return (h0, c0)

    def apply_input_kernel(self, xt: Tensor) -> T.Tensor:
        xto = self.vertical_dropout(xt)
        out = self.input_kernel(xto)
        return out

    def apply_recurrent_kernel(self, h_tm1: Tensor):
        #^ h_tm1 : [b h]
        if self.recurrent_dropout_p > 0.0:
            hto = self.recurrent_dropout(h_tm1)
            out = self.recurrent_kernel(hto)
        else:
            out = self.recurrent_kernel(h_tm1)
        return out

    def forward(
            self, input: T.Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        #^ input : [b i]
        #^ state.h : [b h]

        (h_tm1, z_tm1) = state

        Xh = self.apply_input_kernel(input)
        if self._args.linear_recurrence:
            Hh = self.apply_recurrent_kernel(z_tm1)
        else:
            Hh = self.apply_recurrent_kernel(h_tm1)

        zt = Xh + Hh
        ht = self.activation(zt)

        return ht, (ht, zt)

    @jit.export
    def loop(
            self,
            inputs: List[T.Tensor],
            state_t0: Tuple[T.Tensor],
            mask: Optional[List[T.Tensor]]=None
    ) -> Tuple[List[T.Tensor], Tuple[T.Tensor]]:
        '''
        This loops over t (time) steps
        '''
        #^ inputs      : t * [b i]
        #^ state_t0[0] : [b s]
        #^ out         : [t b h]
        state = state_t0
        outs = []
        for xt in inputs:
            ht, state = self(xt, state)
            outs.append(ht)

        return outs, state

class FlatRNN(BasicRecurrentLayerStack):
    def __init__(
            self,
            input_size: int,
            num_layers: int,
            **kargs,
    ):
        '''
        From:
            - https://arxiv.org/abs/1602.02218 "Strongly-Typed Recurrent Neural Networks" (Typed-Minimal RNN: T-MR)
            - https://arxiv.org/abs/1910.06251 "Deep Independently Recurrent Neural Network (FlatRNN)"
        Example:
            self.rnn = RecurrentLayerStack(
                FlatRNN_Cell_Builder(
                    hidden_size=256,
                    activation='relu',
                    vertical_dropout=0.0,
                    recurrent_dropout=0.0,
                ),
                in_size,
                num_layers,
                batch_first=True,
                return_states=False,
            )
        '''
        builder = FlatRNN_Cell_Builder
        super().__init__(
            builder(**kargs),
            input_size,
            num_layers,
            return_states=True,
            scripted=False,
        )
