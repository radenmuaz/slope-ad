import mygrad
import numpy as np
from mygrad.arrays import ShapedArray
from typing import List, Tuple
class LLOp:
    @staticmethod
    def forward(*args):
        raise NotImplementedError

    @staticmethod
    def vmap(*args):
        raise NotImplementedError

    @staticmethod
    def jvp(*args):
        raise NotImplementedError

    @staticmethod
    def forward_shape(*args):
        raise NotImplementedError

    @classmethod
    def bind1(cls, *args, **params):
        (out,) = mygrad.RT.bind(cls, *args, **params)
        return out


