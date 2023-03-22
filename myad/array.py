import numpy as np

class Array:
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

    @property
    def ndim(self):
        return len(self.shape)

    def str_short(self):
        return repr(self.val)
        # return f'{self.vadtype.name}[{",".join(str(d) for d in self.shape)}]'

    def __hash__(self):
        return hash((self.val.shape, self.val.dtype))

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.shape == other.shape
            and self.dtype == other.dtype
        )

    def __repr__(self):
        return repr(self.val)
