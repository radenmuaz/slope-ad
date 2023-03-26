import myad
from myad import tracing
from myad.tensor import Tensor

class Op:
    @staticmethod
    def forward(*args):
        raise NotImplementedError

    @staticmethod
    def vmap(*args):
        raise NotImplementedError

    @staticmethod
    def jvp(*args):
        raise NotImplementedError

    @staticmethod
    def shape_forward(*args):
        raise NotImplementedError

    @staticmethod
    def pprint():
        return None

class UnaryOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in):
        (x,), (x_bdim,) = vals_in, dims_in
        return [cls.forward(x)], [x_bdim]

class BinaryOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in):
        def move_batch_axis(axis_size, src, dst, x):
            if src is None:
                target_shape = list(x.shape)
                target_shape.insert(dst, axis_size)
                return Tensor.broadcast_to(x, target_shape, [dst])
            elif src == dst:
                return x
            else:
                perm = [i for i in range(x.ndim) if i != src]
                perm.insert(dst, src)
                return x.transpose(perm)
        (x, y), (x_bdim, y_bdim) = vals_in, dims_in
        if x_bdim != y_bdim:
            if x_bdim is None:
                x = move_batch_axis(axis_size, x_bdim, y_bdim, x)
                x_bdim = y_bdim
            else:
                y = move_batch_axis(axis_size, y_bdim, x_bdim, y)
        return [myad.RT.bind1(cls, x, y)], [x_bdim]

class ReduceOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in, *, axis):
        (x,), (x_bdim,) = vals_in, dims_in
        new_axis = tuple(ax + (x_bdim <= ax) for ax in axis)
        out_bdim = x_bdim - sum(ax < x_bdim for ax in axis)
        return [myad.RT.bind1(cls, x, new_axis)], [out_bdim]

class ShapeOp(Op):
    pass
