import slope

# matmul

# XDIMS = (1, 784)
# WDIMS = (784, 100)
XDIMS = (3, 4, 5)
WDIMS = (3, 5, 6)
# XDIMS = (5, 6, 2, 4, 784)
# WDIMS = (5, 6, 2, 784, 100)

# TODO: broadcast support
# XDIMS = (1, 4, 784)
# WDIMS = (2, 1, 784, 100)

x = slope.ones(XDIMS)
x_dot = slope.ones(XDIMS)
w = slope.ones(WDIMS)
w_dot = slope.ones(WDIMS)

@slope.jit
def f(x, w):
    y = x @ w
    return y

# y = f(x, w)
y = slope.vmap(f)(x, w)
y, w_dot = slope.jvp(f, (x, w), (x_dot, w_dot))
y, (grad_L_x, grad_L_w) = slope.value_and_grad(lambda *args: f(*args).sum(), argnums=(0, 1))(x, w)
print(f"{x.shape=}, {w.shape=}")
print(f"{y.shape=},  {grad_L_x.shape=}, {grad_L_w.shape=}")

# x = slope.ones((2, 1, 4, 784))
# w = slope.ones((1, 1, 784, 100))

# x = slope.ones((5, 6, 2, 4, 784))
# w = slope.ones((5, 6, 2, 784, 100))

# x = slope.ones(784,)
# w = slope.ones((2, 784, 100))

# x = slope.ones(2, 4, 784)
# w = slope.ones(784)

# def f(x, w):
#     y = x @ w
#     return y

# # y = slope.vmap(f)(x, w)
# y = f(x, w)
# print(f"{x.shape=}, {w.shape=}")
# print(f"{y.shape=}")