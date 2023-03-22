from myad.llops.base import LLOp
from myad.array_shape import ArrayShape
from typing import List


class Add(LLOp):
    @staticmethod
    def forward(x, y):
        return [x + y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        breakpoint()
        return [x + y], [x_dot + y_dot]

    @staticmethod
    def shape_forward(x: ArrayShape, y: ArrayShape) -> List[ArrayShape]:
        breakpoint()
        if not isinstance(x, ArrayShape) or not isinstance(y, ArrayShape):
            raise TypeError
        if  ArrayShape.from_numpy(x) !=  ArrayShape.from_numpy(y):
            raise TypeError
        return [ArrayShape(x.shape, x.dtype)]

