import slope

# matmul

# x = slope.ones((1, 784))
# x_dot = slope.ones((1, 784))
# w = slope.ones((784, 100))
# w_dot = slope.ones((784, 100))

# def f(x, w):
#     y = x @ w
#     return y

# y = f(x, w)
# print(y)
# y, w_dot = slope.jvp(f, (x, w), (x_dot, w_dot))
# y, (grad_L_x, grad_L_w) = slope.value_and_grad(lambda *args: f(*args).sum())(x, w)
# print(f"{x.shape=}, {w.shape=}")
# print(f"{y.shape=},  {grad_L_x.shape=}, {grad_L_w.shape=}")

# x = slope.ones((2, 1, 4, 784))
# w = slope.ones((1, 1, 784, 100))

# x = slope.ones((5, 6, 2, 4, 784))
# w = slope.ones((5, 6, 2, 784, 100))

# x = slope.ones(784,)
# w = slope.ones((2, 784, 100))

x = slope.ones(2, 4, 784)
w = slope.ones(784)

def f(x, w):
    y = x @ w
    return y

# y = slope.vmap(f)(x, w)
y = f(x, w)
print(f"{x.shape=}, {w.shape=}")
print(f"{y.shape=}")