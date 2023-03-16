import mygrad
from mygrad import llops
# from mygrad import fwd


def f(x):
    y = llops.Mul.bind1(x, 2.0)
    return y


x, xdot = 3.0, 1.0
print(f(x))
# print(y)
# y, ydot = fwd.jvp(f, (x,), (xdot,))
# print(ydot)
