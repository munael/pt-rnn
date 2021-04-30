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

from .rnn_base import (
    IRecurrentCell,
    IRecurrentCellBuilder,
    RecurrentLayer,
    # RecurrentLayerStack,
)
import utils as uu

__all__ = [
    'IndRNN',
    'IndRNN_Cell'
]

ACTIVATIONS: Dict[str, nn.Module] = {
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'hard_tanh': nn.Hardtanh(),
    'relu': nn.ReLU(),
}

class GateSpans(NamedTuple):
    I: T.Tensor
    F: T.Tensor
    G: T.Tensor
    O: T.Tensor

@dataclass
class IndRNN_Cell_Builder(IRecurrentCellBuilder):
    hidden_size : int
    activation  : Union[str, nn.Module] = 'relu'

    vertical_dropout  : float = 0.0
    recurrent_dropout : float = 0.0

    input_kernel_initialization : str = 'xavier_uniform'

    def make(self, input_size: int):
        return IndRNN_Cell(input_size, self)

class IndRNN_Cell(IRecurrentCell):
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
            args: IndRNN_Cell_Builder,
    ):
        super().__init__()
        self._args = args
        self.Dx = input_size
        self.Dh = args.hidden_size

        ## Parameters
        self.recurrent_kernel = uu.Multiply(self.Dh, self.Dh, bias=False)
        self.input_kernel     = nn.Linear(self.Dx, self.Dh)
        ## END Parameters

        self.recurrent_dropout_p    = args.recurrent_dropout or 0.0
        self.vertical_dropout_p     = args.vertical_dropout or 0.0
        
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

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor]) -> Tuple[Tensor, Tuple[Tensor]]
        #^ input : [b i]
        #^ state.h : [b h]

        (h_tm1,) = state

        Xh = self.apply_input_kernel(input)
        Hh = self.apply_recurrent_kernel(h_tm1)

        ht = self.activation(Xh + Hh)

        return ht, (ht,)

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

class RecurrentLayerStack(nn.Module):
    def __init__(
            self,
            cell_builder  : IRecurrentCellBuilder,
            input_size    : int,
            num_layers    : int,
            *,
            batch_first   : bool = False,
            scripted      : bool = True,
            return_states : bool = False,
    ):
        '''
        '''
        super().__init__()
        self._cell_builder = cell_builder

        Dh = cell_builder.hidden_size
        def make(in_size: int):
            cell = cell_builder.make_scripted(in_size)
            return RecurrentLayer(cell, 'forward', batch_first=batch_first)

        rnns = [make(input_size)]
        if num_layers > 1:
            rnns += [make(Dh) for _ in range(num_layers - 1)]

        self.rnn = nn.Sequential(*rnns)

        self.input_size = input_size
        self.hidden_size = self._cell_builder.hidden_size
        self.num_layers = num_layers
        self.return_states = return_states

    def __repr__(self):
        args = ', '.join([
            f'in={self.input_size}',
            f'hid={self.hidden_size}',
            f'layers={self.num_layers}',
        ])
        return f'${self.__class__.__name__}({args}; {self._cell_builder})'

    def forward(self, input, state_t0=None):
        X = input
        for layer_idx, rnn in enumerate(self.rnn):
            is_last = (layer_idx == (len(self.rnn) - 1))
            X, state = rnn(X, state_t0, is_last) 
        if self.return_states:
            return X, state
        return X

class IndRNN(RecurrentLayerStack):
    def __init__(
            self,
            input_size: int,
            num_layers: int,
            *args,
            **kargs,
    ):
        '''
        self.rnn = RecurrentLayerStack(
            IndRNN_Cell_Builder(
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
        builder = IndRNN_Cell_Builder
        super().__init__(
            builder(*args, **kargs),
            input_size,
            num_layers,
        )
