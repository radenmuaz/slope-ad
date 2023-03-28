from myad.ops.base import UnaryOp
from myad.tensor import Tensor

class Exp(UnaryOp):
    @staticmethod
    def eval(x):
        return [Tensor.exp(x)]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [Tensor.exp(x)], [x_dot * Tensor.exp(x)]
