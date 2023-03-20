import myad
from myad import llops
from myad.tracing.forward_diff import jvp
from myad.tracing.ir import make_jaxpr
import numpy as np
from myad.arrays import ShapedArray
from myad.tracing.base import Trace, Tracer, MainTrace

# from mygrad import fwd


def f(x):
    y = llops.Mul.bind1(x, np.array([2.0]))
    return y


x, x_dot = np.array([3.0]), np.array([1.0])
print(f(x))
jaxpr, consts, _ = make_jaxpr(f, ShapedArray.raise_to_shaped(Tracer.get_aval(x)))
print(jaxpr)

# y, y_dot = jvp(f, (x,), (x_dot,))
# print(y, y_dot)
