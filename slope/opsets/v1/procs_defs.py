import slope as sp
from slope.opsets.v1.ops_defs import ops
import math

procs = sp.ProcsDir()

# Functions


@procs.register
def zeros(shape, dtype=sp.Array.default_dtype, **kwargs):
    return ops.full(shape, 0.0, dtype, **kwargs)


@procs.register
def ones(shape, dtype=sp.Array.default_dtype, **kwargs):
    return ops.full(shape, 1.0, dtype, **kwargs)


@procs.register
def full_like(other, fill_value, **kwargs):
    return ops.full(other.shape, fill_value, dtype=other.dtype, **kwargs)


@procs.register
def zeros_like(other, **kwargs):
    return zeros(other.shape, dtype=other.dtype, **kwargs)


@procs.register
def ones_like(other, **kwargs):
    return ops.full(other.shape, fill_value=1.0, dtype=other.dtype, **kwargs)


@procs.register
def where(self, trueval, falseval):
    cond = self != 0.0
    cond = cond.convert(trueval.dtype)  # TODO: type promotion logic
    return cond * trueval + (1.0 - cond) * falseval


@procs.register
def pow(self, y):
    assert type(y) is int
    if y == 0:
        return self.ones_like(x)
    is_reciprocal = y < 0
    if is_reciprocal:
        y = -y
    acc = None
    while y > 0:
        if y & 1:
            acc = x if acc is None else acc * x
        y >>= 1
        if y > 0:
            x = x * x
    ret = acc
    if is_reciprocal:
        ret = self.ones_like(acc) / acc
    return ret


@procs.register
def cross_entropy(x, y):
    return x * y.log()


@procs.register
def mse(x, y):
    return pow((x - y), 2)


@procs.register
def mean(self, axes=None, keepdims=False):
    out = self.sum(axes=axes, keepdim=keepdims)
    return out * (math.prod(out.shape) / math.prod(self.shape))


@procs.register
def minimum(self, other):
    return -self.maximum(-self, -other)


@procs.register
def min(self, axes=None, keepdims=False):
    return -((-self).max(self, axes, keepdims))


@procs.register
def flatten(self, start_dim=0):
    return self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))


@procs.register
@classmethod
def glorot_uniform(cls, *shape, **kwargs):
    return cls.rand(*shape, **kwargs).mul(
        (6 / (shape[0] + math.prod(shape[1:]))) ** 0.5
    )


@procs.register
def T(self):
    perm = list(range(self.ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return self.transpose(perm)


@procs.register
def _softmax(self, axes):
    m = self - self.max(axes, keepdims=True)
    e = m.exp()
    return m, e, e.sum(axes, keepdims=True)


@procs.register
def softmax(self, axes=-1):
    _, e, ss = self._softmax(axes)
    return e.div(ss)


@procs.register
def log_softmax(self, axes=-1):
    m, _, ss = self._softmax(axes)
    return m - ss.log()


@procs.register
def dot(self, w):
    x = self.reshape((*self.shape[0:-1], 1, self.shape[-1]))
    w = w.reshape((*w.shape[0:-2], 1, w.shape[-2], w.shape[-1])).T()
    return (x * w).sum(-1).reshape((*x.shape[0:-2], -1))


@procs.register
def square(self):
    return self * self


@procs.register
def clip(self, min_, max_):
    return ((self - min_).relu() + min_) - (self - max_).relu()


@procs.register
def abs(self):
    return self.relu() + (-self).relu()


@procs.register
def sign(self):
    return self / (self.abs() + 1e-10)


@procs.register
def reciprocal(self):
    return 1.0 / self
