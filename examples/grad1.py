import myad
import numpy as np
from myad.tensor_shape import TensorShape

from myad.tensor import Tensor
# from mygrad import fwd

# from myad.array import Array
# a=Array(np.array([1,]))

def f(x):
    # y = ops.Mul.bind(x, np.array([2.0]))
    # breakpoint()
    y = x*x
    y = y+x
    # y = x*x
    # y = y+x

    return y


# x, x_dot = Tensor.array([3.0]), Tensor.array([1.0])
# y = f(x)
# print('eval', y)

def add_one_to_a_scalar(scalar):
  assert Tensor.ndim(scalar) == 0
  return Tensor.array(1) + scalar

vector_in = np.arange(3.)
vector_out = myad.vmap(add_one_to_a_scalar, (0,))(vector_in)

print(vector_in)
print(vector_out)
# y, y_dot = myad.jvp(f, (x,), (x_dot,))
# print('jvp', y, y_dot)

# jaxpr, consts, _ = myad.make_jaxpr(f, TensorShape.from_numpy(x))
# print('jaxpr')
# print(jaxpr)

