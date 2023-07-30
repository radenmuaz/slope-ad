import slope as sp
import numpy as np
from slope import ops

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

x = sp.ops.randn((1, 3))
y = sp.ops.randn((1, 3))

breakpoint()


def f(x, y):
    out = x
    out = out + y
    out = out.sum()
    # out = ops.dot(out, ops.T(y))
    # out = ops.softmax(out, axis=(1,))
    return out


out, grad_out = sp.grad(f)(x, y)
breakpoint()
print(x)
print(y)
print(out)
print(grad_out)


# dot product


# def f(x, y):
#     out = x * x
#     # out = ops.mm(out, ops.T(y))
#     # out = ops.log_softmax(out, axes=(1,))
#     out = base_ops.sum(out, axes=(0, 1))
#     return out


# out, grad_out = ad.grad(f)(x, y)
# print(x)
# print(y)
# print(out)
# print(grad_out)


# def f(x,y):
#     out = ops.dot(x, ops.T(y))
#     # out = ops.mul(x,y)
#     return out

# x_dot=y_dot=np.array([[1,1,1],[1,1,1]])
# p, t= slope.jvp(f, (x,y), (x_dot,y_dot))
#
# print(p)
# print(t)
