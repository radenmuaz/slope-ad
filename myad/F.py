import myad
from myad import ops
import numpy as np

def expand_dims(x, axis):
    reshape_shape = list(x.shape)
    if type(axis) is int:
        axis = (axis,)
    for a in axis:
        reshape_shape.insert(a, 1)
    x = ops.Reshape.do(x, reshape_shape)
    return x

def dot(x, y):
    # a, b = x.shape[-2], x.shape[-1]
    # c, d = y.shape[-2], y.shape[-1]
    # assert b == c
    y = ops.Transpose.do(y, perm=(-1, -2))
    x = expand_dims(x, -3)
    y = expand_dims(y, -2)
    br_shape = (d, a, b)
    x = ops.Broadcast.do(x, perm=(*x.shape[:-3], *br_shape))
    y = ops.Broadcast.do(x, perm=(*y.shape[:-3], *br_shape))
    x = x * y
    x = ops.ReduceSum.do(x, -1)
    return x

def relu(x):
    return max(x, np.zeros(x.shape, x.dtype))

def softmax(x, axis):
    m = x - ops.ReduceMax.do(x, axis)
    e = ops.Exp.do(m)
    s_e = ops.ReduceSum.do(e, axis)
    
def cross_entropy(x, y):
    return x * ops.Log.do(y)

def mse(x, y):
    return (x - y)**2