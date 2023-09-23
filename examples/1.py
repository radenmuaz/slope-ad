import slope
from slope import numpy as snp


def f(x):
    # y = x * 3.0
    y = x.cos()
    # z = g(y)
    return y


@slope.jit
def g(x):
    return x + 1, x


x = snp.ones(())
x_dot = snp.ones(())
out = f(x)
print(out)


out, jvp_out = slope.jvp(f, (x,), (x_dot,))
print(out, jvp_out)
out, f_lin = slope.linearize(f, x)
print(out)
# print(jvp_out)
# out = f(x)
g_out = slope.grad(f)(x)
print(g_out)
# print(slope.machine.backend.callable.cache_info())
# print(f(x))
# print(slope.machine.backend.callable.cache_info())
