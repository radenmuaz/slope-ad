import slope
from slope import numpy as snp


@slope.jit
def f(x):
    y = x * 3.0
    z = g(y)
    return z


@slope.jit
def g(x):
    return x


x = snp.ones(3)
out = f(x)
print(out)
