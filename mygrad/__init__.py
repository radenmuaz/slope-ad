import numpy as np

def f(x):
    y = sin(x) * 2
    z = -y + x
    return z

from typing import NamedTuple

class Primitive (NamedTuple):
    name: str

add_p = Primitive("add")
mul_p = Primitive("mul")
neg_p = Primitive("neg")
sin_p = Primitive("sin")
cos_p = Primitive("cos")
reduce_sum_p = Primitive("reduce_sum")
greater_p = Primitive("greater")
less_p = Primitive("less")
transpose_p = Primitive("transpose")
broadcast_p = Primitive("broadcast")

def add(x,y): return bind1(add_p, x, y)
def mul(x): return bind1(mul_p, x, y)
def neg(x): return bind1(neg_p, x, y)
def sin(x): return bind1(sin_p, x)
def cos(x): return bind1(cos_p, x)
def greater(x, y): return bind1(greater_p, x, y)
def less(x, y): return bind1(gless_p, x, y)
def transpose(x, perm): return bind1(transpose_p, x, perm=perm)
def broadcast(x, shape, axes): return bind1(broadcast_p, x, shape=shape, axes=axes)
def reduce_sum(a, axis=None):
    if axis is None:
        axis = tuple(range(np.ndim(x)))
    if type(axis) is int:
        axis = (axis, )
    return bind1(reduce_sum_p, x, axis=axis)

def bind1(prim, *args, **params):
    out, = bind(prim, *args, **params)
    return out

from contextlib import contextmanager
from typing import Type, List, Tuple, Sequence, Optional, Any

class MainTrace(NamedTuple):
    level: int
    trace_type: Type["Trace"]
    global_data: Optional[Any]

trace_stack: List[MainTrace] = []
dynamic_trace: Optional[MainTrace] = None

@contextmanager
def new_main(trace_type: Type["Trace"], global_data=None):
    level = len(trace_stack)
    main = MainTrace(level, trace_type, global_data)
    trace_stack += [main]

    try:
        yield main
    finally:
        trace_stack.pop()

class Trace:
    main: MainTrace

    def __init__(self, main: MainTrace) -> None:
        self.main = main
    
    def pure(self, val): 
        raise NotImplementedError
    
    def lift(self, val):
        raise NotImplementedError
    
    def process_primitive(self, primitive, tracers, params):
        raise NotImplementedError

class Tracer
