from myad.ops.base import ShapeOp
from myad.tensor_shape import TensorShape
from myad.tensor import Tensor
from typing import List, Sequence

import numpy as np
class Expand(ShapeOp):
    @staticmethod
    def eval(x, *, shape, axes):
        for axis in sorted(axes):
            # out_ndim = len(axis) + x.ndim
            # shape_it = iter(x.shape)
            # shape = [1 if ax in axis else next(shape_it)
            #          for ax in range(out_ndim)]
            # x = x.reshape(shape)
            x = Tensor.expand_dims(x, axis)
        return [x.broadcast(shape)]

    # @staticmethod
    # def jvp(primals, tangents):
    #     (x, y), (x_dot, y_dot) = primals, tangents
    #     return [x * y], [x_dot * y + x * y_dot]

    @staticmethod
    def shape_eval(
        x: TensorShape, shape: Sequence[int], axes: Sequence[int]
    ) -> List[TensorShape]:
        return [TensorShape(tuple(shape), x.dtype)]
