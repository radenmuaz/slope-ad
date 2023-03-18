import mygrad
from mygrad import llops
from mygrad.tracing.forward_diff import jvp
import numpy as np
# from mygrad import fwd

def f(x):
    # y = llops.Mul.bind1(x, np.array([2.0]))
    y = llops.Mul.bind1(x, 2.0)
    return y


x, x_dot = np.array([3.0]), np.array([1.0])
# print(y)
y, y_dot = jvp(f, (x,), (x_dot,))
print(y, y_dot)
# print(ydot)
