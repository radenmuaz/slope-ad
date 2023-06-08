import unittest

import slope
from slope import ad
from slope.array import Array
class TestGrad(unittest.TestCase):
    def test_maximum(self):
        def f(x):
            z = Array.zeros_like(x)
            out = x.maximum(z)
            return out
        x = Array([1,0.5,-0.4])
        x_dot = Array([1,1,1])
        y, y_dot = ad.jvp(f, (x,), (x_dot,))
        print(y)
        print(y_dot)
        # y, f_lin = ad.linearize(f, x)
        # y_dot = f_lin(x_dot)
    # def test_grad1(self):
    #     def f(x):
    #         # y = x*x
    #         y = x+1
    #         y = y.broadcast(shape=(3,))
    #         y = y.sum(axes=(0,))
    #         return y
    #     y, f_lin = ad.linearize(f, x)
    #     x, x_dot = Array(3), Array(1.)
    #     print(y)
    #     y_dot = f_lin(x_dot)
    #     print(y_dot)
        


if __name__ == "__main__":
    unittest.main()
