import slope


x = slope.ones((1, 3, 4, 4))
x_dot = slope.ones((1, 3, 4, 4))
w = slope.full((3, 8, 2, 2), 2.)
w_dot = slope.full((3, 8, 2, 2), 2.)
print(x.shape, w.shape)

def f(x, w):
    y = x.conv_transpose(w,stride=2)
    return y

y = f(x, w)
y, w_dot = slope.jvp(f, (x, w), (x_dot, w_dot))
print(f"{y.shape=}, {w.shape=}")
y, (gL_x, gL_w) = slope.value_and_grad(lambda *args: f(*args).sum())(x, w)
print(f"{y.shape=},  {gL_x.shape=}, {gL_w.shape=}")
