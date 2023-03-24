from myad.ops.base import UnaryOp
from myad.tensor_shape import TensorShape
from typing import List


class Neg(UnaryOp):
    @staticmethod
    def forward(x):
        return [-x]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [-x], [-x_dot]

    @staticmethod
    def shape_forward(x: TensorShape) -> List[TensorShape]:
        return [TensorShape(x.shape, x.dtype)]
