import slope
from slope import environment as sev


@slope.jit
def f(x):
    y = x * 3.0
    # y = x.cos()
    # z = g(y)
    return y


@slope.jit
def g(x):
    return x + 1, x


x = sev.ones(())
x_dot = sev.ones(())
out = f(x)
print(out)
out = slope.jit(f)(x)
# print(out)

# out, jvp_out = slope.jvp(f, (x,), (x_dot,)); print(out, jvp_out)
# out, f_lin = slope.linearize(f, x); print(out)
# out_jvp = f_lin(x_dot); print(out_jvp)
# print(jvp_out)
# out = f(x)

# g_out = slope.grad(f)(x); print(g_out)
# print(slope.machine.backend.gen_jit_fn.cache_info())
# g_out = slope.grad(f)(x); print(g_out)
# print(slope.machine.backend.gen_jit_fn.cache_info())

# print(f(x))
# print(slope.machine.backend.callable.cache_info())
