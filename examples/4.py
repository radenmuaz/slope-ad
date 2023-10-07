import slope


x = slope.ones((3,))
x_dot = slope.ones((3,))

# Concat


@slope.jit
def f(x):
    y = x
    y = y[0:1]
    y = y.sum()
    return y


# @slope.jit
# def f(x):
#     y = x
#     y = y.flip()
#     return y


# out = f(x)
# out, out_deriv = slope.jvp(f, (x,), (x_dot,))
# print(out)
g_out = slope.grad(f)(x)
print(g_out)

# g_out = slope.grad(f)(x)
# print(g_out)
