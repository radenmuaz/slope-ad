import myad
import numpy as np

def f(x):
    y = x * x
    y = y + x
    return y
# x, x_dot = np.array([3.0]), np.array([1.0])
# y = f(x)
# print('eval', y)

def add_one_to_a_scalar(scalar):
    assert np.ndim(scalar) == 0
    return np.array(1) + scalar

vector_in = np.arange(3.0)
vector_out = myad.vmap(add_one_to_a_scalar, (0,))(vector_in)

print(vector_in)
print(vector_out)

# y, y_dot = myad.jvp(f, (x,), (x_dot,))
# print('jvp', y, y_dot)

# jaxpr, consts, _ = myad.make_jaxpr(f, ArrayShape.from_numpy(x))
# print('jaxpr')
# print(jaxpr)
