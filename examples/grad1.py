from autodidax import linearize_flat
import slope
import numpy as np
from slope import ops

# def f(x):
#     y = x * x
#     return y

# p, t = slope.jvp(f, (np.random.randn(1),), (np.ones(1),))
# print(p)
# print(t)
# l = slope.grad(f)(2)
# print(l)

# x, x_dot = np.array([3.0]), np.array([1.0])
# y = f(x)
# print('eval', y)

# def add_one_to_a_scalar(scalar):
#     assert np.ndim(scalar) == 0
#     return np.array(1) + scalar

# vector_in = np.arange(3.0)
# vector_out = slope.vmap(add_one_to_a_scalar, (0,))(vector_in)
# print(vector_in)
# print(vector_out)

# x, x_dot = np.array([3.0]), np.array([1.0])
# y, y_dot = slope.jvp(f, (x,), (x_dot,))
# print('jvp', y, y_dot)


# x, x_dot = np.array([3.0]), np.array([1.0])
# y, f_lin = slope.linearize(f, x)
# print(y)
# y_dot = f_lin(x_dot)
# print(y_dot)


def f(x):
    y = x
    y = ops.broadcast(y, (3, 3), (0,))
    y = ops.reduce_sum(y, (0, 1))
    return y


x = np.ones([3])
x_dot = np.ones([3])
out = f(x)
# l = slope.jvp(f, (x,), (x_dot))
print(out)


# def f(x):
#     y = ops.Broadcast.do(x, shape=(3,))
#     # y = ops.ReduceSum.do(y, axis=(0,))
#     return y

g = slope.ad.grad(f)
l = g(np.ones((1)))
print(l)
# l = slope.jvp(f, (np.ones([1,]),), (np.ones([1,]),))
# print(l)

# print('jvp', y, y_dot)

# jaxpr, consts, _ = slope.make_jaxpr(f, ArrayShape.from_numpy(x))
# print('jaxpr')
# print(jaxpr)
