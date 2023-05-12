import numpy as np
from typing import Tuple, Any


class ArrayShape:
    array_abstraction_level = 1
    shape: Tuple[int, ...]
    dtype: np.dtype

    @classmethod
    def like(cls, aval):
        return cls(aval.shape, aval.dtype)

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    @staticmethod
    def _bool(tracer):
        raise Exception("ArrayShape can't be unambiguously converted to bool")

    @staticmethod
    def _nonzero(tracer):
        raise Exception("ArrayShape can't be unambiguously converted to bool")

    def str_short(self):
        return f'{str(self.dtype)}[{",".join(str(d) for d in self.shape)}]'

    def __hash__(self):
        return hash((self.shape, self.dtype))

    def __eq__(self, other):
        return tuple(self.shape) == tuple(other.shape) and self.dtype == other.dtype

    def __repr__(self):
        return f"ArrayShape(shape={self.shape}, dtype={self.dtype})"