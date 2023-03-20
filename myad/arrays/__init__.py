from typing import Tuple
import numpy as np

from myad import llops
from myad import utils


class ShapedArray:
    array_abstraction_level = 1
    shape: Tuple[int, ...]
    dtype: np.dtype

    @classmethod
    def raise_to_shaped(cls, aval):
        return cls(aval.shape, aval.dtype)

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    @staticmethod
    def _bool(tracer):
        raise Exception("ShapedArray can't be unambiguously convemygrad.RTed to bool")

    @staticmethod
    def _nonzero(tracer):
        raise Exception("ShapedArray can't be unambiguously convemygrad.RTed to bool")

    def str_short(self):
        return f'{self.dtype.name}[{",".join(str(d) for d in self.shape)}]'

    def __hash__(self):
        return hash((self.shape, self.dtype))

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.shape == other.shape
            and self.dtype == other.dtype
        )

    def __repr__(self):
        return f"ShapedArray(shape={self.shape}, dtype={self.dtype})"


class ConcreteArray(ShapedArray):
    array_abstraction_level = 2
    val: np.ndarray

    def __init__(self, val):
        self.val = val
        self.shape = val.shape
        self.dtype = val.dtype

    @staticmethod
    def _bool(tracer):
        return bool(tracer.aval.val)

    @staticmethod
    def _nonzero(tracer):
        return bool(tracer.aval.val)


def get_aval(cls, x):
    if isinstance(x, cls):
        return x.aval
    # print(f"warn: {x} ({type(x)}) is not Tracer")
    elif type(x) in cls.TYPES:
        return ConcreteArray(np.asarray(x))
    else:
        raise TypeError(x)
