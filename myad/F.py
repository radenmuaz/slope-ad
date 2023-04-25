import myad
from myad import ops

# UnaryOps
def identity(x):
    return myad.RT.bind1(ops.Identity, x)
def exp(x):
    return myad.RT.bind1(ops.Exp, x)
def log(x):
    return myad.RT.bind1(ops.Log, x)
def neg(x):
    return myad.RT.bind1(ops.Neg, x)

# BinaryOps
def add(x, y):
    return myad.RT.bind1(ops.Add, x, y)
def sub(x, y):
    return myad.RT.bind1(ops.Sub, x, y)
def mul(x, y):
    return myad.RT.bind1(ops.Mul, x, y)
def div(x, y):
    return myad.RT.bind1(ops.Div, x, y)
def pow(x, y):
    return myad.RT.bind1(ops.Pow, x, y)

# ReduceOps
def reduce_sum(x, axis):
    return myad.RT.bind1(ops.ReduceSum, x, axis)
def reduce_max(x, axis):
    return myad.RT.bind1(ops.ReduceMax, x, axis)

# ShapeOps

def broadcast(x, shape):
    return myad.RT.bind1(ops.Broadcast, x, shape)

def reshape(x, shape):
    return myad.RT.bind1(ops.Reshape, x, shape)

def transpose(x, perm):
    return myad.RT.bind1(ops.Transpose, x, perm)

# ML
def expand_dims(x, axis):
    reshape_shape = list(x.shape)
    if type(axis) is int:
        axis = (axis,)
    for a in axis:
        reshape_shape.insert(a, 1)
    x = reshape(x, reshape_shape)
    return x

def dot(x, y):
    a, b = x.shape[-2], x.shape[-1]
    c, d = y.shape[-2], y.shape[-1]
    assert b == c
    y = transpose(y, (-1, -2))
    x = expand_dims(x, -3)
    y = expand_dims(y, -2)
    br_shape = (d, a, b)
    x = broadcast(x, (*x.shape[:-3], *br_shape))
    y = broadcast(x, (*y.shape[:-3], *br_shape))
    x = x * y
    x = reduce_sum(x, -1)
    return x