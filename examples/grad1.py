import myad
import numpy as np
from myad.tensor_shape import TensorShape

from myad.tensor import Tensor, new_tensor
# from mygrad import fwd

# from myad.array import Array
# a=Array(np.array([1,]))

def f(x):
    # y = llops.Mul.bind(x, np.array([2.0]))
    # breakpoint()
    y = x*x
    y = y+x
    # y = x*x
    # y = y+x

    return y


x, x_dot = new_tensor([3.0]), new_tensor([1.0])
y = f(x)
print('eval', y)

y, y_dot = myad.jvp(f, (x,), (x_dot,))
print('jvp', y, y_dot)

jaxpr, consts, _ = myad.make_jaxpr(f, TensorShape.from_numpy(x))
print('jaxpr')
print(jaxpr)

