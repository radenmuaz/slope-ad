import slope



# x = slope.ones((1, 784))
# x_dot = slope.ones((1, 784))
# w = slope.ones((784, 100))
# w_dot = slope.ones((784, 100))
# print(x.shape, w.shape)

# def f(x, w):
#     y = x @ w
#     return y

# y = f(x, w)
# y, w_dot = slope.jvp(f, (x, w), (x_dot, w_dot))
# y, grad_L_y = slope.grad(lambda *args: f(*args).sum())(x, w)


# x = slope.ones((1, 3, 16, 16))
# x_dot = slope.ones((1, 3, 16, 16))
# w = slope.ones((8, 3, 3, 3)) * 2
# w_dot = slope.ones((8, 3, 3, 3)) * 2
# print(x.shape, w.shape)


# def f(x, w):
#     y = x.conv(w)
#     return y

# y = f(x, w)
# y, w_dot = slope.jvp(f, (x, w), (x_dot, w_dot))
# y, grad_L_y = slope.grad(lambda *args: f(*args).sum())(x, w)


# x = slope.ones((1, 3, 4, 4))
# x_dot = slope.ones((1, 3, 4, 4))
# w = slope.full((3, 8, 2, 2), 2.)
# w_dot = slope.full((3, 8, 2, 2), 2.)
# print(x.shape, w.shape)


# def f(x, w):
#     y = x.conv_transpose(w)
#     return y

# y = f(x, w)


x = slope.ones((1, 3, 16, 16))
x_dot = slope.ones((1, 3, 16, 16))
w = slope.ones((8, 3, 3, 3)) * 2
w_dot = slope.ones((8, 3, 3, 3)) * 2
print(x.shape, w.shape)

# @slope.jit
def f(x, w):
    y = x.conv(w,padding=1)
    return y

y = f(x, w)
y, w_dot = slope.jvp(f, (x, w), (x_dot, w_dot))
y, grad_L_y = slope.grad(lambda *args: f(*args).sum())(x, w)

