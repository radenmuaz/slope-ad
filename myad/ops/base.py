import myad
from myad.tensor import Tensor
from myad.tensor_shape  import TensorShape
from typing import List, Tuple, Sequence, Any
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod


class Op(ABC):
    @staticmethod
    @abstractmethod
    def eval(*args):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def vmap(*args):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def jvp(*args):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def shape_eval(*args: Any, **kwargs: Any) -> Any :
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pprint():
        return None

class UnaryOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in):
        (x,), (x_bdim,) = vals_in, dims_in
        return [cls.eval(x)], [x_bdim]

    @staticmethod
    def shape_eval(x: TensorShape) -> List[TensorShape]:
        return [TensorShape(x.shape, x.dtype)]

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

    @staticmethod
    def shape_eval(x: TensorShape, y: TensorShape) -> List[TensorShape]:
        if not isinstance(x, TensorShape) or not isinstance(y, TensorShape):
            raise TypeError
        if  TensorShape.from_numpy(x) !=  TensorShape.from_numpy(y):
            raise TypeError
        return [TensorShape(x.shape, x.dtype)]

class ReduceOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in, *, axis):
        (x,), (x_bdim,) = vals_in, dims_in
        new_axis = tuple(ax + (x_bdim <= ax) for ax in axis)
        out_bdim = x_bdim - sum(ax < x_bdim for ax in axis)
        return [myad.RT.bind1(cls, x, new_axis)], [out_bdim]


    @staticmethod
    def shape_eval(x: TensorShape, *, axis: Tuple[int, ...]) -> List[TensorShape]:
        axis_ = set(axis)
        new_shape = [d for i, d in enumerate(x.shape) if i not in axis_]
        return [TensorShape(tuple(new_shape), x.dtype)]


class ShapeOp(Op):
    pass


#-----------------------
# UnaryOps
#-----------------------


class Identity(UnaryOp):
    @staticmethod
    def eval(x):
        return [x]


class Exp(UnaryOp):
    @staticmethod
    def eval(x):
        return [Tensor.exp(x)]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [Tensor.exp(x)], [x_dot * Tensor.exp(x)]

class Log(UnaryOp):
    @staticmethod
    def eval(x):
        return [Tensor.log(x)]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [Tensor.log(x)], [x_dot / x]


class Neg(UnaryOp):
    @staticmethod
    def eval(x):
        return [-x]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [-x], [-x_dot]



#-----------------------
# BinaryOps
#-----------------------

class Add(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x + y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x + y], [x_dot + y_dot]


class Sub(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x - y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x + y], [x_dot + y_dot]

class Mul(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x * y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]

    @staticmethod
    def T(ct, x, y):
        z_bar, = ct
        assert (x is None) ^ (y is None)
        return [(z_bar * y), None] if x is None else [None, (x * z_bar)]


class Div(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x / y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x / y], [(x_dot / y) + (-y_dot * x * (y**-2))]


class Pow(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x ** y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]


#-----------------------
# ReduceOps
#-----------------------

class Max(ReduceOp):
    @staticmethod
    def eval(x, *, axis):
        return [x.sum(axis)]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]



class Sum(ReduceOp):
    @staticmethod
    def eval(x, *, axis):
        return [x.sum(axis)]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]


#-----------------------
# ShapeOps
#-----------------------

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



class Crop(ShapeOp):
    @staticmethod
    def eval(x, *, perm):
        return [x.transpose(perm)]




class Reshape(ShapeOp):
    @staticmethod
    def eval(x, *, perm):
        return [np.reshape(x, perm)]




class Permute(ShapeOp):
    @staticmethod
    def eval(x, *, perm):
        return [x.transpose(perm)]



class Pad(ShapeOp):
    @staticmethod
    def eval(x, *, perm):
        return [x.transpose(perm)]
