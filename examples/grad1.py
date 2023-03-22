import myad
from myad import llops
from myad.forward_diff import jvp
from myad.ir import make_jaxpr
import numpy as np
from myad.array_shape import ArrayShape
from myad.tracing import Trace, Tracer, MainTrace

# from mygrad import fwd

# from myad.array import Array
# a=Array(np.array([1,]))

def f(x):
    # y = llops.Mul.bind(x, np.array([2.0]))
    y = llops.Mul.bind1(x, x)
    # y = x*x
    # y = y+x

    return y


x, x_dot = np.array([3.0]), np.array([1.0])
y = f(x)
print(y)
y, y_dot = jvp(f, (x,), (x_dot,))
print(y, y_dot)
jaxpr, consts, _ = make_jaxpr(f, ArrayShape.from_numpy(x))
print(jaxpr)

