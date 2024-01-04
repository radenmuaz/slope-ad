import slope
import slope.nn as nn
# x = slope.ones((1, 3, 16, 16))
# x_dot = slope.ones((1, 3, 16, 16))
# w = slope.ones((8, 3, 3, 3)) * 2
# w_dot = slope.ones((8, 3, 3, 3)) * 2

# def f(x, w):
#     y = x.conv(w)
#     return y
# y = f(x, w)

# y, w_dot = slope.jvp(f, (x, w), (x_dot, w_dot))
# y, (grad_L_x, grad_L_w) = slope.value_and_grad(lambda *args: f(*args).sum(), argnums=(0,1))(x, w)
# print(f"{x.shape=}, {w.shape=}")
# print(f"{y.shape=},  {grad_L_x.shape=}, {grad_L_w.shape=}")

x = slope.ones((10, 1, 3, 16, 16))
x_dot = slope.ones((1, 3, 16, 16))
w = slope.ones((8, 3, 3, 3)) * 2
w_dot = slope.ones((8, 3, 3, 3)) * 2

def f(x, w):
    y = x.conv(w)
    return y
y = slope.vmap(f)(x, w)
print(f"{x.shape=}, {w.shape=}")
print(f"{y.shape=}")

# y, w_dot = slope.jvp(f, (x, w), (x_dot, w_dot))
# y, (grad_L_x, grad_L_w) = slope.value_and_grad(lambda *args: f(*args).sum(), argnums=(0,1))(x, w)
# print(f"{x.shape=}, {w.shape=}")
# print(f"{y.shape=},  {grad_L_x.shape=}, {grad_L_w.shape=}")

