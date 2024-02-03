import slope


# Pad
def f(x):
    out = x
    out = out.pad(1, 0)
    out = out.sum()
    return out


x = slope.ones(3)
x_dot = slope.ones(3)
# print(slope.grad(f)(x))

# Concat


@slope.jit
def f(x):
    y = x
    y = slope.cat([x, x])
    y = y.sum()
    return y


out = f(x)
out, out_deriv = slope.jvp(f, (x,), (x_dot,))
g_out = slope.grad(f)(x)
print(g_out)

# Sum


@slope.jit
def f(x):
    y = x
    y = y[0:1]
    y = y.sum()
    return y


@slope.jit
def f(x):
    y = x
    y = y.flip()
    return y
