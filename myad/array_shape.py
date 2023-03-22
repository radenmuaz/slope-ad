import numpy as np
from typing import Tuple
class ArrayShape:
    array_abstraction_level = 1
    shape: Tuple[int, ...]
    dtype: np.dtype

    @classmethod
    def from_numpy(cls, aval):
        return cls(aval.shape, aval.dtype)

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    @staticmethod
    def _bool(tracer):
        raise Exception("ArrayShape can't be unambiguously convemygrad.RTed to bool")

    @staticmethod
    def _nonzero(tracer):
        raise Exception("ArrayShape can't be unambiguously convemygrad.RTed to bool")

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
        return f"ArrayShape(shape={self.shape}, dtype={self.dtype})"
