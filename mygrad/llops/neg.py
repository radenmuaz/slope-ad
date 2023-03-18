from mygrad.llops.base import LLOp
from mygrad.arrays import ShapedArray
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
    def forward_shape(x: ShapedArray) -> List[ShapedArray]:
        return [ShapedArray(x.shape, x.dtype)]