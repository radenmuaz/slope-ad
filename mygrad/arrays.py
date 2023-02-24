from typing import Tuple
import numpy as np

from mygrad import primitives as pm
from mygrad import utils


class ShapedArray:
    array_abstraction_level = 1
    shape: Tuple[int, ...]
    dtype: np.dtype

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    _neg = staticmethod(pm.neg)
    _add = staticmethod(pm.add)
    _radd = staticmethod(utils.swap(pm.add))
    _mul = staticmethod(pm.mul)
    _rmul = staticmethod(utils.swap(pm.mul))
    _gt = staticmethod(pm.greater)
    _lt = staticmethod(pm.less)

    @staticmethod
    def _bool(tracer):
        raise Exception("ShapedArray can't be unambiguously converted to bool")

    @staticmethod
    def _nonzero(tracer):
        raise Exception("ShapedArray can't be unambiguously converted to bool")

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


def raise_to_shaped(aval):
    return ShapedArray(aval.shape, aval.dtype)
