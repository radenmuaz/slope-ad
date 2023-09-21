import slope
from slope import numpy as snp


def f(x):
    y = x * 3.0
    # z = g(y)
    return y


@slope.jit
def g(x):
    return x + 1, x


x = snp.ones((3,))
# print(f(x)
out = f(x)
print(out)
# print(slope.machine.backend.callable.cache_info())
# print(f(x))
# print(slope.machine.backend.callable.cache_info())
