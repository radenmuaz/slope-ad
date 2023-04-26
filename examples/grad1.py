from autodidax import linearize_flat
import myad
import numpy as np
from myad import ops
def f(x):
    y = x * x
    y = y + x
    return y

    
# x, x_dot = np.array([3.0]), np.array([1.0])
# y = f(x)
# print('eval', y)

# def add_one_to_a_scalar(scalar):
#     assert np.ndim(scalar) == 0
#     return np.array(1) + scalar

# vector_in = np.arange(3.0)
# vector_out = myad.vmap(add_one_to_a_scalar, (0,))(vector_in)
# print(vector_in)
# print(vector_out)

# x, x_dot = np.array([3.0]), np.array([1.0])
# y, y_dot = myad.jvp(f, (x,), (x_dot,))
# print('jvp', y, y_dot)


# x, x_dot = np.array([3.0]), np.array([1.0])
# y, f_lin = myad.linearize(f, x)
# print(y)
# y_dot = f_lin(x_dot)
# print(y_dot)



def f(x):
    y = ops.ReduceSum.do(x, axis=(0,))
    return y

g = myad.grad(f)
l = g(np.ones((3)))
print(l)


# def f(x):
#     y = ops.Broadcast.do(x, shape=(3,))
#     # y = ops.ReduceSum.do(y, axis=(0,))
#     return y

# # g = myad.grad(f)
# # l = g(np.ones((1)))
# l = myad.jvp(f, (np.ones([1,]),), (np.ones([1,]),))
# print(l)

# print('jvp', y, y_dot)

# jaxpr, consts, _ = myad.make_jaxpr(f, ArrayShape.from_numpy(x))
# print('jaxpr')
# print(jaxpr)
