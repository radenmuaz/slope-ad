from mygrad.llops.base import LLOp
from mygrad.arrays import ShapedArray
from typing import List

class Mul(LLOp):
    @staticmethod
    def forward(x, y):
        return [x * y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]

    @staticmethod
    def forward_shape(x: ShapedArray, y: ShapedArray) -> List[ShapedArray]:
        if not isinstance(x, ShapedArray) or not isinstance(y, ShapedArray):
            raise TypeError
        if ShapedArray.raise_to_shaped(x) != ShapedArray.raise_to_shaped(y):
            raise TypeError
        return [ShapedArray(x.shape, x.dtype)]