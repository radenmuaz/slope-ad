from mygrad.llops.base import LLOp
from mygrad.arrays import ShapedArray
from typing import Tuple, List, Sequence

import numpy as np
class Expand(LLOp):
    @staticmethod
    def forward(x, *, axis):
        return [np.broadcast_to(x, axis)]

    # @staticmethod
    # def jvp(primals, tangents):
    #     (x, y), (x_dot, y_dot) = primals, tangents
    #     return [x * y], [x_dot * y + x * y_dot]


    @staticmethod
    def forward_shape(
        x: ShapedArray, *, shape: Sequence[int], axes: Sequence[int]
    ) -> List[ShapedArray]:
        return [ShapedArray(tuple(shape), x.dtype)]
