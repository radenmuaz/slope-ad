import slope

# matmul

x = slope.ones((1, 784))
x_dot = slope.ones((1, 784))
w = slope.ones((784, 100))
w_dot = slope.ones((784, 100))

def f(x, w):
    y = x @ w
    return y

y = f(x, w)
y, w_dot = slope.jvp(f, (x, w), (x_dot, w_dot))
y, (grad_L_x, grad_L_w) = slope.value_and_grad(lambda *args: f(*args).sum())(x, w)
print(f"{x.shape=}, {w.shape=}")
print(f"{y.shape=},  {grad_L_x.shape=}, {grad_L_w.shape=}")
