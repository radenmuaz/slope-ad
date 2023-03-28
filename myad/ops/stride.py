from myad.ops.base import ShapeOp


class Stride(ShapeOp):
    @staticmethod
    def eval(x, *, perm):
        return [x.transpose(perm)]
