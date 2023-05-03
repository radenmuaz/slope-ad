import slope
import numpy as np
from slope import ad, ops

# x = np.ones((1, 3))
# y = np.ones((3, 1))

# x = np.array([[1, 2, 3]])
# y = np.array([[1, 2, 3]]).T
# x = np.array([[1, 2, 3], [1, 2, 3]]).T
# y = np.array([[1, 2, 3], [1, 2, 3]])

# out = ops.dot(x, y)

# print('in')
# print(x.shape)
# print(x)
# print()
# print(y.shape)
# print(y)
# print()
# print('out')
# print(out)
# print(out.shape)

# x = np.array([[1, 2, 3], [1, 2, 3]]).T


## # dot product
# x = np.random.randn(1, 3)
# y = np.random.randn(2, 3)


# def f(x, y):
#     out = x
#     out = ops.dot(out, ops.T(y))
#     out = ops.softmax(out, axis=(1,))
#     out = ops.reduce_sum(out, axis=(0, 1))
#     return out


# out, grad_out = ad.grad(f)(x, y)
# print(x)
# print(y)
# print(out)
# print(grad_out)



# dot product
x = np.random.randn(1, 3)
y = np.random.randn(2, 3)


def f(x, y):
    out = x
    out = ops.dot(out, ops.T(y))
    out = ops.log_softmax(out, axis=(1,))
    out = ops.reduce_sum(out, axis=(0, 1))
    return out


out, grad_out = ad.grad(f)(x, y)
print(x)
print(y)
print(out)
print(grad_out)


# def f(x,y):
#     out = ops.dot(x, ops.T(y))
#     # out = ops.mul(x,y)
#     return out

# x_dot=y_dot=np.array([[1,1,1],[1,1,1]])
# p, t= slope.jvp(f, (x,y), (x_dot,y_dot))
#
# print(p)
# print(t)
