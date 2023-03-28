from myad.ops.base import ShapeOp


class Permute(ShapeOp):
    @staticmethod
    def eval(x, *, perm):
        return [x.transpose(perm)]
