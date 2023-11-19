import slope



x1 = slope.ones((1, 784))
x1_dot = slope.ones((1, 784))
x2 = slope.ones((784, 100))
x2_dot = slope.ones((784, 100))
print(x1.shape, x2.shape)

def f(x1, x2):
    y = x1 @ x2
    return y

y = f(x1, x2)
y, y_dot = slope.jvp(f, (x1, x2), (x1_dot, x2_dot))
y, grad_y = slope.grad(lambda *args: f(*args).sum())(x1, x2)


# x = slope.ones((1, 3, 16, 16))
# y = slope.ones((8, 3, 3, 3)) * 2
# print(x.shape, y.shape)
# print(slope.conv(x, y).shape)

# def test_with_input(f, input_tensors, out_tensors):

