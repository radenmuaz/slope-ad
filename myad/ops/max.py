from myad.ops.base import ReduceOp
from myad.tensor_shape import TensorShape
from typing import Tuple, List


class Sum(ReduceOp):
    @staticmethod
    def eval(x, *, axis):
        return [x.sum(axis)]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]
