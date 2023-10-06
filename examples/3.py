import slope
from slope import environment as sev

x = slope.ones((3,))
x_dot = slope.ones((3,))

# Concat


@slope.jit
def f(x):
    y = x
    y = slope.concatenate([x, x])
    y = y.sum()
    return y


out = f(x)
out, out_deriv = slope.jvp(f, (x,), (x_dot,))
g_out = slope.grad(f)(x)
print(g_out)
