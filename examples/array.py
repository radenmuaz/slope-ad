import slope
import numpy as np
from slope import ad, ops

# x = ad.Array(np.ones((1,3)))
# y = ad.Array(np.full(shape=(1,3), fill_value=2.))
# z = x + y
# print(z)


# dot product
x = ad.Array(np.random.randn(1, 3))
y = ad.Array(np.random.randn(2, 3))


def f(x, y):
    out = x
    out = out + ad.Array(np.random.randn(1, 3))
    out = ops.dot(out, ops.T(y))
    out = ops.log_softmax(out, axis=(1,))
    out = ops.reduce_sum(out, axis=(0, 1))
    return out


out, grad_out = ad.grad(f)(x, y)
print(x)
print(y)
print(out)
print(grad_out)
