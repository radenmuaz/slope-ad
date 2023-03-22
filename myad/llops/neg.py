from myad.llops.base import LLOp
from myad.array_shape import ArrayShape
from typing import List


class Neg(LLOp):
    @staticmethod
    def forward(x):
        return [-x]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [-x], [-x_dot]

    @staticmethod
    def shape_forward(x: ArrayShape) -> List[ArrayShape]:
        return [ArrayShape(x.shape, x.dtype)]
