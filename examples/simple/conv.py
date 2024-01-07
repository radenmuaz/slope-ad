import slope
import slope.nn as nn

stride = 2
padding = 2
dilation = 1
groups=1

w_size = (64, 32, 3, 3)
x_size = (2, 32, 16, 16)

x = slope.ones(x_size)
w = slope.ones(w_size)

@slope.jit
def f(x, w):
    y = x.conv(w, groups, stride, dilation, padding)
    return y

y = f(x,w)
y, y_dot = slope.jvp(f, (x, w), (x.ones_like(), w.ones_like()))
print(f"{x.shape=}, {w.shape=}")
print(f"{y.shape=}, {y_dot.shape=}")

y, (gL_x, gL_w) = slope.value_and_grad(lambda *args: f(*args).sum(), argnums=(0,1))(x, w)
print(f"{y.shape=},  {gL_x.shape=}, {gL_w.shape=}")

x = slope.ones(10, *x_size)
y_vmap = slope.vmap(f, in_dim=(0, None))(x, w)
print(f"{x.shape=}, {w.shape=}")
print(f"{y_vmap.shape=}")
