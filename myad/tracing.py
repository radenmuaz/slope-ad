from typing import NamedTuple
from contextlib import contextmanager
from typing import Type, Optional, Any, List, Tuple
import operator as op

from myad import llops
import numpy as np

from myad.array import Array

from typing import Tuple
import numpy as np

class MainTrace(NamedTuple):
    level: int
    trace_type: Type["Trace"]
    global_data: Optional[Any]


class Trace:
    main: MainTrace

    def __init__(self, main: MainTrace) -> None:
        self.main = main

    def pure(self, val):
        raise NotImplementedError

    def lift(self, val):
        raise NotImplementedError

    def run_llop(self, LLOp, tracers, params):
        raise NotImplementedError


class Tracer:
    TYPES = {
        bool,
        int,
        float,
        np.bool_,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        np.ndarray,
    }
    _trace: Trace

    __array_priority__ = 1000

    @property
    def aval(self):
        raise NotImplementedError

    @property
    def ndim(self):
        return len(self.shape)

    def __neg__(self):
        return llops.Neg.bind1(self)

    def __add__(self, other):
        return llops.Add.bind1(self, other)

    def __radd__(self, other):
        return llops.Add.bind1(other, self)

    def __mul__(self, other):
        return llops.Mul.bind1(self, other)

    def __rmul__(self, other):
        return llops.Mul.bind1(other, self)

    def __bool__(self):
        return self.aval._bool(self)

    def __nonzero__(self):
        return self.aval._nonzero(self)

    @classmethod
    def get_aval(cls, x):
        if isinstance(x, cls):
            return x.aval
        # print(f"warn: {x} ({type(x)}) is not Tracer")
        elif type(x) in cls.TYPES:
            return Array(np.asarray(x))
        else:
            raise TypeError(x)

    def full_lower(self):
        return self  # default implementation

    def __getattr__(self, name):
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def zeros_like(self, val):
        aval = self.get_aval(val)
        return np.zeros(aval.shape, aval.dtype)

