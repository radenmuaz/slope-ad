import slope
import numpy as np
# x = slope.tensor(np.array([1,2,3]))
x = slope.ones((3,))
# print(x)

@slope.jit
def f(x):
    y = x.sum()
    return y



# x = slope.ones(())
# print(x)
print(f(x))
print(f(x))
print(f(x))



# @slope.jit
# def f(x):
#     y = x * 3.0
#     # y = x.cos()
#     # z = g(y)
#     return y


# @slope.jit
# def g(x):
#     return x + 1, x

# print(out)
# out = slope.jit(f)(x)
# print(out)

# out, jvp_out = slope.jvp(f, (x,), (x_dot,)); print(out, jvp_out)
# out, f_lin = slope.linearize(f, x); print(out)
# out_jvp = f_lin(x_dot); print(out_jvp)
# print(jvp_out)
# out = f(x); print(out)

g_out = slope.grad(f)(x); print(g_out)
# g_out = slope.grad(f)(x); print(g_out)

# print(f(x))
# print(slope.machine.backend.callable.cache_info())
