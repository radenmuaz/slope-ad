from myad.llops.base import LLOp
from myad.tensor_shape import TensorShape
from typing import List
from myad import tracing
class Add(LLOp):
    @staticmethod
    def forward(x, y):
        return [x + y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x + y], [x_dot + y_dot]

    @staticmethod
    def vmap(op, axis_size, vals_in, dims_in):
        (x, y), (x_bdim, y_bdim) = vals_in, dims_in
        if x_bdim != y_bdim:
            if x_bdim is tracing.not_mapped:
                x = tracing.move_batch_axis(axis_size, x_bdim, y_bdim, x)
                x_bdim = y_bdim
            else:
                y = tracing.move_batch_axis(axis_size, y_bdim, x_bdim, y)
        return [op(x, y)], [x_bdim]

    @staticmethod
    def shape_forward(x: TensorShape, y: TensorShape) -> List[TensorShape]:
        if not isinstance(x, TensorShape) or not isinstance(y, TensorShape):
            raise TypeError
        if  TensorShape.from_numpy(x) !=  TensorShape.from_numpy(y):
            raise TypeError
        return [TensorShape(x.shape, x.dtype)]

