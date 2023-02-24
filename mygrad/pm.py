from typing import NamedTuple
import numpy as np
from mygrad import trc

class Primitive(NamedTuple):
    name: str

add_p = Primitive('add')
mul_p = Primitive('mul')
neg_p = Primitive("neg")
sin_p = Primitive("sin")
cos_p = Primitive("cos")
reduce_sum_p = Primitive("reduce_sum")
greater_p = Primitive("greater")
less_p = Primitive("less")
transpose_p = Primitive("transpose")
broadcast_p = Primitive("broadcast")

def add(x, y): return bind1(add_p, x, y)
def mul(x, y): return bind1(mul_p, x, y)
def neg(x): return bind1(neg_p, x)
def sin(x): return bind1(sin_p, x)
def cos(x): return bind1(cos_p, x)
def greater(x, y): return bind1(greater_p, x, y)
def less(x, y): return bind1(less_p, x, y)
def transpose(x, perm): return bind1(transpose_p, x, perm=perm)
def broadcast(x, shape, axes): return bind1(broadcast_p, x, shape=shape, axes=axes)
def reduce_sum(x, axis=None):
    if axis is None:
        axis = tuple(range(np.ndim(x)))
    if type(axis) is int:
        axis = (axis,)
    return bind1(reduce_sum_p, x, axis=axis)

def bind1(prim, *args, **params):
    out, = bind(prim, *args, **params)
    return out

def bind(prim, *args, **params):
  top_trace = trc.find_top_trace(args)
  tracers = [trc.full_raise(top_trace, arg) for arg in args]
  outs = top_trace.process_primitive(prim, tracers, params)
  return [trc.full_lower(out) for out in outs]
