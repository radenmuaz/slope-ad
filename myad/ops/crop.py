from myad.ops.base import ShapeOp


class Crop(ShapeOp):
    @staticmethod
    def eval(x, *, perm):
        return [x.transpose(perm)]
