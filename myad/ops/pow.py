from myad.ops.base import BinaryOp

class Pow(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x ** y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]
