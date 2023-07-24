import slope
import numpy as np
from slope import ad, base_ops


# dot product
x = ad.Array(np.random.randn(4, 2, 3, 3))


def f(x):
    out = x
    out = base_ops.reshape(out, (out.shape[0], -1))
    return out


out = f(x)
print(out.shape)
out = ad.vmap(f, (0,))(x)
print(out.shape)
out = ad.vmap(f, (1,))(x)
print(out.shape)


# out, grad_out = ad.grad(f)(x, y)
# print(x)
# print(y)
# print(out)
# print(grad_out)
