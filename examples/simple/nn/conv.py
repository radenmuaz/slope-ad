import slope
import slope.nn as nn
x = slope.ones((2, 3, 16, 16))
w = slope.ones((8, 3, 3, 3))

@slope.jit
def f(x, w):
    y = x.conv(w)
    return y
y = f(x,w)
y, w_dot = slope.jvp(f, (x, w), (x.ones_like(), w.ones_like()))
y, (gL_x, gL_w) = slope.value_and_grad(lambda *args: f(*args).sum(), argnums=(0,1))(x, w)
print(f"{x.shape=}, {w.shape=}")
print(f"{y.shape=},  {gL_x.shape=}, {gL_w.shape=}")

x = slope.ones((10, 2, 3, 16, 16))
y_vmap = slope.vmap(f, in_dim=(0, None))(x, w)
print(f"{x.shape=}, {w.shape=}")
print(f"{y_vmap.shape=}")
