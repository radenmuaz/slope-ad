from myad.ops.base import UnaryOp
from myad.tensor import Tensor

class Log(UnaryOp):
    @staticmethod
    def eval(x):
        return [Tensor.log(x)]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [Tensor.log(x)], [x_dot / x]
