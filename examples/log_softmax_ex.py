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
    # out = out + ad.Array(np.random.randn(1, 3))
    # out = out.T
    out = out.dot(y.T)
    out = out.log_softmax(1)
    # out = out.sum(axes=(0, 1))
    out = out.sum()
    return out

# print(f(x,y))
out, grad_out = ad.grad(f)(x, y)
print(x)
print(y)
print(out)
print(grad_out)
