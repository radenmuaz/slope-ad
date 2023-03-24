from myad.ops.base import UnaryOp


class Identity(UnaryOp):
    @staticmethod
    def forward(x):
        return [x]
