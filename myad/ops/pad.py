from myad.ops.base import ShapeOp


class Pad(ShapeOp):
    @staticmethod
    def eval(x, *, perm):
        return [x.transpose(perm)]
