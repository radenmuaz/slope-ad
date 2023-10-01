import slope
from slope import environment as sev

x = sev.ones((3,))
x_dot = sev.ones((3,))

@slope.jit
def f(x):
    y = x
    y = sev.concatenate([x,x])
    y = y.sum()
    return y

# out = f(x)
# out, out_deriv = slope.jvp(f, (x,), (x_dot,))
g_out = slope.grad(f)(x)
print(g_out)