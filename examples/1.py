import slope
from slope import numpy as snp


def f(x):
    y = x * 3.0
    z = g(y)
    return z


@slope.jit
def g(x):
    return x


x = snp.ones(3)
print(f(x))
print(slope.machine.backend.callable.cache_info())
print(f(x))
print(slope.machine.backend.callable.cache_info())
