from myad.ops.base import ShapeOp


class Transpose(ShapeOp):
    @staticmethod
    def forward(x, *, perm):
        return [x.transpose(perm)]
