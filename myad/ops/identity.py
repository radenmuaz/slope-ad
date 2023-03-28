from myad.ops.base import UnaryOp


class Identity(UnaryOp):
    @staticmethod
    def eval(x):
        return [x]
