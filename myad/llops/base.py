import myad
import numpy as np
from myad.arrays import ShapedArray
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
    def shape_forward(*args):
        raise NotImplementedError

    @classmethod
    def bind1(cls, *args, **params):
        (out,) = myad.RT.bind(cls, *args, **params)
        return out
