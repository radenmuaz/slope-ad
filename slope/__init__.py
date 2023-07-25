from typing import (
    Callable,
    NamedTuple,
    Dict,
    Type,
    Hashable,
    Tuple,
    List,
    Any,
    Iterable,
    Iterator,
    Type,
    List,
    Tuple,
    Optional,
    Any,
    Union,
    NamedTuple,
    Dict,
    Set,
    DefaultDict,
    Callable,
    Hashable,
    Final,
    Sequence,
)
from contextlib import contextmanager
import itertools
import weakref
from functools import lru_cache, reduce, partial
from collections import defaultdict
from enum import Enum, auto
import operator as op
import string
import numpy as np
import math
import pickle


class IDHashable:
    val: Any

    def __init__(self, val):
        self.val = val

    def __hash__(self) -> int:
        return id(self.val)

    def __eq__(self, other):
        return type(other) is IDHashable and id(self.val) == id(other.val)


def swap(f):
    return lambda x, y: f(y, x)


def unzip2(pairs):
    lst1, lst2 = [], []
    for x1, x2 in pairs:
        lst1.append(x1)
        lst2.append(x2)
    return lst1, lst2


def list_map(f: Any, *xs: Any) -> Any:
    return list(map(f, *xs))


def list_zip(*args: Any) -> Any:
    fst, *rest = args = list_map(list, args)
    n = len(fst)
    for arg in rest:
        assert len(arg) == n
    return list(zip(*args))


def split_half(lst: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert not len(lst) % 2
    return split_list(lst, len(lst) // 2)


def merge_lists(which: List[bool], l1: List[Any], l2: List[Any]) -> List[Any]:
    l1, l2 = iter(l1), iter(l2)
    out = [next(l2) if b else next(l1) for b in which]
    assert next(l1, None) is next(l2, None) is None
    return out


def split_list(lst: List[Any], n: int) -> Tuple[List[Any], List[Any]]:
    assert 0 <= n <= len(lst)
    return lst[:n], lst[n:]


def partition_list(bs: List[bool], l: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert len(bs) == len(l)
    lst1: List[Any] = []
    lst2: List[Any] = []
    lists = lst1, lst2
    # lists = lst1: List[Any], lst2: List[Any] = list(), list()
    for b, x in list_zip(bs, l):
        lists[b].append(x)
    return lst1, lst2


class MainTrace(NamedTuple):
    level: int
    trace_type: Type["Trace"]
    global_data: Optional[Any]


class Trace:
    main: MainTrace

    def __init__(self, main: MainTrace) -> None:
        self.main = main

    def pure(self, val):
        raise NotImplementedError

    def lift(self, val):
        raise NotImplementedError

    def run_op(self, op, tracers, params):
        raise NotImplementedError


class EvalTrace(Trace):
    pure = lift = lambda self, x: x

    def run_op(self, op, tracers, params):
        return op.eval(*tracers, **params)


class DType(NamedTuple):
    priority: int
    itemsize: int
    name: str
    np: type

    def __repr__(self):
        return f"dtypes.{self.name}"


class BaseArray:
    bool: Final[DType] = DType(0, 1, "bool", bool)
    float16: Final[DType] = DType(0, 2, "half", np.float16)
    float32: Final[DType] = DType(4, 4, "float", np.float32)
    int8: Final[DType] = DType(0, 1, "char", np.int8)
    int32: Final[DType] = DType(1, 4, "int", np.int32)
    int64: Final[DType] = DType(2, 8, "int64", np.int64)
    uint8: Final[DType] = DType(0, 1, "uchar", np.uint8)
    default_dtype = float32

    def is_int(self) -> bool:
        return self.dtype in (self.int8, self.uint8, self.int32, self.int64)

    def is_float(self) -> bool:
        return self.dtype in (self.float16, self.float32)

    def is_unsigned(self) -> bool:
        return self.dtype is self.uint8

    def __getattr__(self, attr):
        raise NotImplementedError

    __neg__ = lambda self: self.neg()
    __add__ = lambda self, other: self.add(other)
    __radd__ = lambda self, other: self.__class__.add(other, self)
    __sub__ = lambda self, other: self.sub(other)
    __rsub__ = lambda self, other: self.__class__.sub(other, self)
    __mul__  = lambda self, other: self.mul(other)
    __rmul__  = lambda self, other: self.__class__.mul(other, self)
    __div__  = lambda self, other: self.div(other)
    __rdiv__  = lambda self, other: self.__class__.div(other, self)
    __truediv__ = __div__
    __truerdiv__ = __rdiv__
    __eq__ = lambda self, other: self.equal(other)
    __ne__ = lambda self, other: self.not_equal(other)
    __ge__ = lambda self, other: self.maximum(other).equal(self)
    __le__ = lambda self, other: self.minimum(other).equal(self)
    __gt__ = lambda self, other: 1.0 - (self <= other)
    __lt__ = lambda self, other: 1.0 - (self >= other)


class ArrayBuffer:
    def __init__(self, val):
        self.val = val


class Array(BaseArray):
    __array_priority__ = 2000
    default_dtype = BaseArray.float32

    def __init__(
        self,
        val: Union[list, tuple, np.ndarray, ArrayBuffer] = None,
        dtype: Optional[Any] = None,
    ):
        self.buf = (
            val
            if isinstance(val, ArrayBuffer)
            else backend.run_impl(ops.constant, val=val, dtype=dtype).buf
        )

    val = property(lambda self: self.buf.val)
    dtype = property(lambda self: self.buf.val.dtype)
    shape = property(lambda self: self.buf.val.shape)
    ndim = property(lambda self: self.buf.val.ndim)

    def __getattr__(self, attr):
        if attr in self.__dict__.keys():
            return self.__dict__[attr]
        if attr in vars(ops).keys():
            op = getattr(ops, attr)
            if op.op_type is not OpType.Other:
                return partial(op.impl, self)
            else:
                return op
        raise AttributeError(f"{self.__class__.__name__} has no attribute {attr}")

    def __repr__(self):
        return f"{self.__class__.__name__}: {repr(self.val)[6:-1]}"

    __str__ = __repr__

    def __getitem__(self, idx):
        if type(idx) in (tuple, list):
            return self.slice(slice(idx))
        raise NotImplementedError

    def __setitem__(self, idx, val):
        raise NotImplementedError


class TracerArray(BaseArray):
    TYPES = {
        bool,
        int,
        float,
        # Array,
    }
    __array_priority__ = 1000

    default_dtype = np.float32
    _trace: "Trace"

    def __init__(self):
        raise NotImplementedError

    # @property
    # def aval(self):
    #     return self.get_aval(self.val)

    aval = property(lambda self: self.get_aval(self.val))
    dtype = property(lambda self: self.val.dtype)
    shape = property(lambda self: self.val.shape)
    ndim = property(lambda self: self.val.ndim)

    def full_lower(self):
        return self.val

    def __repr__(self):
        return f"{self.__class__.__name__}: {repr(self.val)}"

    def __str__(self):
        return repr(self)

    @staticmethod
    def get_aval(x):
        if isinstance(x, TracerArray):
            return x.aval
        elif type(x) in TracerArray.TYPES:
            return Array(np.asarray(x))
        elif isinstance(x, Array):
            return x
        else:
            raise TypeError(x)

    def full_lower(self):
        return self

    @staticmethod
    def reduceop_patch(op_fn):
        def wrapped_fn(x, axes=None, keepdims=False):
            ret = op_fn(x, axes, keepdims)
            if keepdims:
                if len(ret.shape) == 0:
                    shape = (1,)
                else:
                    shape = tuple(1 if i in axes else d for i, d in enumerate(x.shape))
                ret = ret.reshape(shape)
            return ret

        return wrapped_fn

    def __getattr__(self, attr):
        if attr in vars(ops).keys():
            op = getattr(ops, attr)
            if op.op_type is OpType.Reduce:
                return self.reduceop_patch(partial(op, self))
            elif op.op_type is not OpType.Other:
                return partial(op, self)
            else:
                return op
        try:
            return getattr(self.aval, attr)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {attr}")

    @property
    def ndim(self):
        return len(self.shape)


class Empty:
    pass


empty = Empty()


class Store:
    val = empty

    def set_value(self, val):
        assert self.val is empty
        self.val = val

    def __call__(self):
        return self.val


class NodeType(NamedTuple):
    name: str
    to_iterable: Callable
    from_iterable: Callable


class PyTreeDef(NamedTuple):
    node_type: NodeType
    node_metadata: Hashable
    child_treedefs: Tuple["PyTreeDef", ...]


class Leaf:
    pass


leaf = Leaf()


def tree_flatten(x: Any) -> Any:
    def _tree_flatten(x: Any) -> Tuple[Iterable, Union[PyTreeDef, Leaf]]:
        node_type = RT.node_types.get(type(x))

        if node_type:
            node_metadata, children = node_type.to_iterable(x)
            children_flat, child_trees = unzip2(list_map(_tree_flatten, children))
            flattened = itertools.chain.from_iterable(children_flat)
            return flattened, PyTreeDef(node_type, node_metadata, tuple(child_trees))
        else:
            return [x], leaf

    children_iter, treedef = _tree_flatten(x)

    return list(children_iter), treedef


def tree_unflatten(treedef: PyTreeDef, xs: List[Any]) -> Any:
    def _tree_unflatten(treedef: PyTreeDef, xs: Iterator) -> Any:
        if treedef is leaf:
            return next(xs)
        else:
            children = (_tree_unflatten(t, xs) for t in treedef.child_treedefs)
            return treedef.node_type.from_iterable(treedef.node_metadata, children)

    return _tree_unflatten(treedef, iter(xs))


def flatten_fun(f, in_tree):
    store = Store()

    def flat_fun(*args_flat):
        pytree_args = tree_unflatten(in_tree, args_flat)
        out = f(*pytree_args)
        out_flat, out_tree = tree_flatten(out)
        store.set_value(out_tree)
        return out_flat

    return flat_fun, store


class Runtime:
    def __init__(self, root_trace=MainTrace(0, EvalTrace, None)):
        self.trace_stack: List[MainTrace] = []
        self.dynamic_trace: Optional[MainTrace] = None
        self.trace_stack += [root_trace]
        self.node_types = dict()
        self.register_pytree_node(tuple, lambda t: (None, t), lambda _, xs: tuple(xs))
        self.register_pytree_node(list, lambda l: (None, l), lambda _, xs: list(xs))
        self.register_pytree_node(
            dict,
            lambda d: map(tuple, unzip2(sorted(d.items()))),
            lambda keys, vals: dict(zip(keys, vals)),
        )

    def register_pytree_node(
        self, ty: Type, to_iter: Callable, from_iter: Callable
    ) -> None:
        self.node_types[ty] = NodeType(str(ty), to_iter, from_iter)

    @contextmanager
    def new_main(self, trace_type: Type["Trace"], global_data=None):
        level = len(self.trace_stack)
        main = MainTrace(level, trace_type, global_data)
        self.trace_stack.append(main)

        try:
            yield main
        finally:
            self.trace_stack.pop()

    @contextmanager
    def new_dynamic(self, main: MainTrace):
        prev_dynamic_trace, self.dynamic_trace = self.dynamic_trace, main
        try:
            yield
        finally:
            self.dynamic_trace = prev_dynamic_trace

    def bind(self, op, *args, **params):
        top_trace = self.find_top_trace(args)
        tracers = [self.full_raise(top_trace, arg) for arg in args]
        outs = top_trace.run_op(op, tracers, params)
        lowered = [self.full_lower(out) for out in outs]
        return lowered

    def bind1(self, *args, **params):
        return self.bind(*args, **params)[0]

    def find_top_trace(self, xs) -> Trace:
        top_main = max(
            (x._trace.main for x in xs if isinstance(x, TracerArray)),
            default=self.trace_stack[0],
            key=op.attrgetter("level"),
        )
        if self.dynamic_trace and self.dynamic_trace.level > top_main.level:
            top_main = self.dynamic_trace
        return top_main.trace_type(top_main)

    def full_raise(self, trace: Trace, val: Any) -> TracerArray:
        if not isinstance(val, TracerArray):
            return trace.pure(val)
        level = trace.main.level
        if val._trace.main is trace.main:
            return val
        elif val._trace.main.level < level:
            return trace.lift(val)
        elif val._trace.main.level > level:
            raise Exception(f"Can't lift level {val._trace.main.level} to {level}.")
        else:  # val._trace.level == level
            raise Exception(f"Different traces at same level: {val._trace}, {trace}.")

    def full_lower(self, val: Any):
        if isinstance(val, TracerArray):
            return val.full_lower()
        else:
            return val


class Backend:
    def __init__(self, name, default_dtype=Array.float32):
        self.name = name
        self.input_handlers = dict(Array=lambda x: x)
        self.default_dtype = default_dtype
        self.impls = dict()
        self.callable = lru_cache(self.callable)

    def callable(
        self,
        hashable_prog: IDHashable,
        hashable_consts: Tuple[IDHashable, ...],
    ):
        prog: Prog = hashable_prog.val
        typecheck_prog(prog)
        consts = [x.val for x in hashable_consts]
        in_avals = [v.aval for v in prog.in_binders[len(consts) :]]
        compiled = self.compile(
            prog, consts, in_avals, name=f"{self.__class__.__name__.lower()}_fn"
        )
        return compiled

    def execute_compiled(self, compiled, out_avals, *args):
        input_bufs = [self.input_handlers[type(x)](x) for x in args]
        out_bufs = compiled.execute(input_bufs)
        return [Array(buf) for aval, buf in zip(out_avals, out_bufs)]

    def compile(self, prog, consts, in_avals, name: str):
        raise NotImplementedError

    def set_compile(self, fn):
        self.compile = partial(fn, self)

    def set_impl(self, op):
        def set_impl_(fn):
            self.impls[op] = fn

        return set_impl_

    # def set_run_impl(self, fn):
    #     self.run_impl = partial(fn, self)

    def run_impl(self, op, *args, **kwargs):
        args = [a.val for a in args]
        return Array(
            ArrayBuffer(self.impls[op](*args, **kwargs))
        )  #     raise NotImplementedError

    #     # self.op_impls[op_name](*args, **kwargs)

    def set_input_handler(self, typ, fn):
        self.input_handlers[typ] = fn

    # def __getattr__(self, attr):
    #     if attr in self.impls.keys():
    #         return partial(self.run_impl, self, attr)
    #     print(f"{attr} not found in {self}.op_impls")
    #     breakpoint()
    #     return getattr(self.attr)


class JitFn:
    def __init__(self, code, fn):
        super().__init__()
        self.code = code
        self.fn = fn

    def __call__(self, *args, **kwargs):
        args = [a.val if isinstance(a, Array) else a for a in args]
        outs = self.fn(*args, **kwargs)
        return [Array(o) for o in outs]
        # return [Array(o) if isinstance(o, np.ndarray) else o for o in outs]


RT = Runtime()


class ArrayShape:
    array_abstraction_level = 1
    shape: Tuple[int, ...]
    dtype: np.dtype

    @classmethod
    def like(cls, aval):
        return cls(aval.shape, aval.dtype)

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def ndim(self):
        return len(self.shape)

    @staticmethod
    def _bool(tracer):
        raise Exception("ArrayShape can't be unambiguously converted to bool")

    @staticmethod
    def _nonzero(tracer):
        raise Exception("ArrayShape can't be unambiguously converted to bool")

    def str_short(self):
        return f'{str(self.dtype)}[{",".join(str(d) for d in self.shape)}]'

    def __hash__(self):
        return hash((self.shape, self.dtype))

    def __eq__(self, other):
        return tuple(self.shape) == tuple(other.shape) and self.dtype == other.dtype

    def __repr__(self):
        return f"ArrayShape(shape={self.shape}, dtype={self.dtype})"


def raise_not_implemented(self, *args, **kwargs):
    raise NotImplementedError


class OpType(Enum):
    Unary = auto()
    Binary = auto()
    Reduce = auto()
    Shape = auto()
    Load = auto()
    Other = auto()


class Op:
    def __init__(self, name, op_type=OpType.Other):
        self.name = name
        self.op_type = op_type
        self.impls = dict()
        self.eval = raise_not_implemented
        self.jvp = raise_not_implemented
        self.vmap = raise_not_implemented
        self.T = raise_not_implemented
        self.shape_eval = raise_not_implemented
        self.args_fixer = None

    def impl(self, *args, **kwargs):
        if self.args_fixer is not None:
            args, kwargs = self.args_fixer(*args, **kwargs)
        return backend.run_impl(self, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.args_fixer is not None:
            args, kwargs = self.args_fixer(*args, **kwargs)
        return RT.bind1(self, *args, **kwargs)[0]

    def set_args_fixer(self, fn):
        self.args_fixer = fn

    def set_eval(self, fn):
        self.eval = fn

    def set_jvp(self, fn):
        self.jvp = fn

    def set_vmap(self, fn):
        self.vmap = fn

    def set_T(self, fn):
        self.T = fn

    def set_shape_eval(self, fn):
        self.shape_eval = fn

    def add_impl(self, backend):
        def add_impl_(impl):
            self[backend] = impl

        return add_impl_

    @classmethod
    def unary(cls, name):
        op = cls(name, OpType.Unary)

        @op.set_vmap
        def fn(self, axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            return [self(x, **params)], [x_bdim]

        @op.set_shape_eval
        def fn(self, axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            return [self(x, **params)], [x_bdim]

        @op.set_jvp
        def fn(self, primals, tangents, **params):
            (x,), (x_dot,) = primals, tangents
            return [self(x, **params)], [self(x_dot, **params)]

        return op

    @classmethod
    def binary(cls, name):
        op = cls(name, OpType.Binary)

        @op.set_args_fixer
        def fn(x, y, **kwargs):
            # if x.shape == y.shape:
            # #     return (x, y), kwargs
            # if type(x) in [bool, int, float, Array]:
            #     x = y._trace.pure(x)
            # elif type(y) in [bool, int, float, Array]:
            #     y = x._trace.pure(y)
            bx = list(range((max(x.ndim, y.ndim) - x.ndim)))
            by = list(range((max(x.ndim, y.ndim) - y.ndim)))
            bx = bx if len(bx) > 0 else None
            by = by if len(by) > 0 else None
            shape_ret = tuple(max(sx, sy) for sx, sy in zip(x.shape, y.shape))
            x = x.broadcast(shape=shape_ret, axes=bx)
            y = y.broadcast(shape=shape_ret, axes=by)
            return (x, y), kwargs

        @op.set_vmap
        def fn(self, axis_size, vals_in, dims_in, **params):
            (x, y), (x_bdim, y_bdim) = vals_in, dims_in
            if x_bdim != y_bdim:
                if x_bdim is None:
                    x = move_batch_axis(axis_size, x_bdim, y_bdim, x)
                    x_bdim = y_bdim
                else:
                    y = move_batch_axis(axis_size, y_bdim, x_bdim, y)
            return [self(x, y, **params)], [x_bdim]

        @op.set_shape_eval
        def fn(x: ArrayShape, y: ArrayShape, **params) -> List[ArrayShape]:
            # if not isinstance(x, ArrayShape) or not isinstance(y, ArrayShape):
            if not type(x) in (Array, ArrayShape) or not type(x) in (Array, ArrayShape):
                # breakpoint()
                raise TypeError
            if ArrayShape.like(x) != ArrayShape.like(y):
                raise TypeError(f"{x} != {y}")
            return [ArrayShape(x.shape, x.dtype)]

        @op.set_jvp
        def fn(self, primals, tangents, **params):
            (x,), (x_dot,) = primals, tangents
            return [self(x, **params)], [self(x_dot, **params)]

        return op

    @classmethod
    def reduce(cls, name):
        op = cls(name, OpType.Reduce)

        @op.set_args_fixer
        def fn(x, axes=None, keepdims=False):
            if axes is None:
                axes = tuple(range(x.ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            axes = tuple(a if a >= 0 else a + len(x.shape) for a in axes)
            return (x,), dict(axes=axes, keepdims=keepdims)

        @op.set_vmap
        def fn(axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            axes = list(params["axes"])
            axes = tuple(a + (x_bdim <= a) for a in axes)
            out_bdim = x_bdim - sum(a < x_bdim for a in axes)
            params["axes"] = tuple(axes)
            return [cls.do(x, **params)], [out_bdim]

        @op.set_shape_eval
        def fn(x: ArrayShape, **params) -> List[ArrayShape]:
            axes = params["axes"]
            axes = [a + len(x.shape) if a < 0 else a for a in axes]
            axes_ = set(axes)
            new_shape = [d for i, d in enumerate(x.shape) if i not in axes_]
            return [ArrayShape(tuple(new_shape), x.dtype)]

        return op

    @classmethod
    def shape(cls, name):
        op = cls(name, OpType.Shape)
        return op

    @classmethod
    def load(cls, name):
        op = cls(name, OpType.Load)

        def set_load_fn(fn_):
            @op.set_eval
            def fn(*args, **kwargs):
                out = fn_(*args, **kwargs)
                return [out]

            @op.set_jvp
            def fn(*args, **kwargs):
                out = fn_(*args, **kwargs)
                out_jvp = Array.ones_like(out)
                return [out], [out_jvp]

        op.set_load_fn = set_load_fn

        @op.set_T
        def fn(self, cts, *args, **kwargs):
            return [cts[0]]

        return op


# ========================
# Ops
# ========================


class OpsDir:
    pass


ops = OpsDir()


class ProcsDir:
    def add(self, fn):
        setattr(self, fn.__name__, fn)
        del fn


procs = ProcsDir()

# -----------------------
# UnaryOps
# -----------------------

ops.stop_gradient = Op.unary("stop_gradient")


@ops.stop_gradient.set_eval
def fn(x):
    return [x]


@ops.stop_gradient.set_jvp
def fn(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x], [Array.zeros_like(x_dot)]


@ops.stop_gradient.set_T
def fn(cts, x):
    (z,) = cts
    assert type(x) is UndefPrimal
    return [Array.zeros_like(z)]


ops.convert = Op.unary("convert")


@ops.convert.set_eval
def fn(x):
    return [x]


@ops.convert.set_jvp
def fn(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x], [Array.zeros_like(x_dot)]


@ops.convert.set_T
def fn(cts, x):
    (z,) = cts
    assert type(x) is UndefPrimal
    return [Array.zeros_like(z)]


ops.sqrt = Op.unary("sqrt")


@ops.sqrt.set_eval
def fn(x):
    return [x.sqrt()]


@ops.sqrt.set_jvp
def fn(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.sqrt()
    return [ans], [x_dot * (0.5 / ans)]


@ops.convert.set_T
def fn(cts, x):
    (z,) = cts
    assert type(x) is UndefPrimal
    return [Array.zeros_like(z)]


ops.sin = Op.unary("sin")


@ops.sin.set_eval
def fn(x):
    return [x.sin()]


@ops.sin.set_jvp
def fn(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.sin()
    return [ans], [x_dot * (1 - ans)]


@ops.sin.set_T
def fn(cts, x):
    (z,) = cts
    return [-z * (1 - x.sin())]


ops.exp = Op.unary("exp")


@ops.exp.set_eval
def fn(x):
    return [x.exp()]


@ops.exp.set_jvp
def fn(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.exp()
    return [ans], [x_dot * ans]


@ops.exp.set_T
def fn(cts, x):
    (z,) = cts
    return [1 / z]


ops.log = Op.unary("log")


@ops.log.set_eval
def fn(x):
    return [x.log()]


@ops.log.set_jvp
def fn(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.log()
    return [ans], [x_dot / x]


@ops.log.set_T
def fn(cts, x):
    (z,) = cts
    return [1 / z]


ops.neg = Op.unary("neg")


@ops.neg.set_eval
def fn(x):
    return [-x]


@ops.neg.set_jvp
def fn(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [-x], [-x_dot]


@ops.neg.set_T
def fn(cts, x):
    (z,) = cts
    return [-z]


ops.relu = Op.unary("relu")


@ops.relu.set_eval
def fn(x):
    return [-x]


@ops.relu.set_jvp
def fn(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [-x], [-x_dot]


@ops.relu.set_T
def fn(cts, x):
    (z,) = cts
    return [-z]


# -----------------------
# BinaryOps
# -----------------------


ops.add = Op.binary("add")


@ops.add.set_eval
def fn(x, y):
    return [x + y]


@ops.add.set_jvp
def fn(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x + y], [x_dot + y_dot]


@ops.add.set_T
def fn(cts, x, y):
    (z_bar,) = cts
    return [z_bar, z_bar]


ops.sub = Op.binary("sub")


@ops.sub.set_eval
def fn(x, y):
    return [x - y]


@ops.sub.set_jvp
def fn(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x - y], [x_dot - y_dot]


@ops.sub.set_T
def fn(cts, x, y):
    (z_bar,) = cts
    return [z_bar, -z_bar]


ops.mul = Op.binary("mul")


@ops.mul.set_eval
def fn(x, y):
    return [x * y]


@ops.mul.set_jvp
def fn(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x * y], [(x_dot * y) + (y_dot * x)]
    # jvp_out = (y * x_dot) + (y_dot * x) # order problem, x*y_dot fails


@ops.mul.set_T
def fn(cts, x, y):
    (z_bar,) = cts
    assert (type(x) is UndefPrimal) ^ (type(y) is UndefPrimal)
    if type(x) is UndefPrimal:
        return [z_bar * y, None]
    elif type(y) is UndefPrimal:
        return [None, x * z_bar]


ops.div = Op.binary("div")


@ops.div.set_eval
def fn(x, y):
    return [x / y]


@ops.div.set_jvp
def fn(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x / y], [
        (x_dot / y) + (-y_dot * x * (y**-2))
    ]  # bug: power returns float64


@ops.div.set_T
def fn(cts, x, y):
    (z_bar,) = cts
    return [z_bar / y, None]


ops.maximum = Op.binary("maximum")


@ops.maximum.set_eval
def fn(x, y):
    return [x.maximum(y)]


@ops.maximum.set_jvp
def fn(primals, tangents):
    def _balanced_eq(x, z, y):
        return ((x == z).where(Array.ones_like(z), Array.zeros_like(z))) / (
            (y == z).where(Array.full_like(z, 2), Array.ones_like(z))
        )

    (x, y), (x_dot, y_dot) = primals, tangents
    eval_out = x.maximum(y)
    jvp_out = x_dot * _balanced_eq(x, eval_out, y) + y_dot * _balanced_eq(
        y, eval_out, x
    )

    return [eval_out], [jvp_out]


@ops.maximum.set_T
def fn(cts, x, y):
    (z_bar,) = cts
    return [z_bar, None]


ops.equal = Op.binary("equal")


@ops.equal.set_eval
def fn(x, y):
    return [x.equal(y)]


@ops.equal.set_jvp
def fn(primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = x.equal(y)
    return [out_primal], [Array.zeros(out_primal.shape, out_primal.dtype)]


@ops.equal.set_T
def fn(cts, x, y):
    (z_bar,) = cts
    return [z_bar, None]


ops.max = Op.reduce("max")


@ops.max.set_eval
def fn(x, axes):
    return [x.max(axes)]


@ops.max.set_jvp
def fn(primals, tangents, axes):
    (x,), (x_dot,) = primals, tangents
    eval_out = x.max(axes)
    locs = x.equal(eval_out.broadcast(x.shape, axes))
    locs = locs.convert(x_dot.dtype)
    counts = locs.sum(axes)
    jvp_out = (x_dot * locs).sum(axes)
    jvp_out = jvp_out / counts.broadcast(jvp_out.shape)

    return [eval_out], [jvp_out]


@ops.max.set_T
def fn(cts, x, *, axes):
    (z,) = cts
    return [z.broadcast(x.aval.shape, ())]


ops.sum = Op.reduce("sum")


@ops.sum.set_eval
def fn(x, *, axes, keepdims):
    return [x.sum(axes, keepdims)]


@ops.sum.set_jvp
def fn(primals, tangents, *, axes, keepdims):
    (x,), (x_dot,) = primals, tangents
    eval_out = x.sum(axes, keepdims)
    jvp_out = x_dot.sum(axes, keepdims)
    return [eval_out], [jvp_out]


@ops.sum.set_T
def fn(cts, x, *, axes, keepdims):
    (z,) = cts
    out = z
    out = z.broadcast(x.aval.shape, axes)
    return [out]


# -----------------------
# ShapeOps
# -----------------------

ops.broadcast = Op.shape("broadcast")


@ops.broadcast.set_args_fixer
def fn(x, *, shape, axes):
    if isinstance(axes, int):
        axes = (axes,)
    if axes is not None:
        axes = tuple(a if a >= 0 else a + len(x.shape) for a in axes)
    return (x,), dict(shape=shape, axes=axes)


@ops.broadcast.set_eval
def fn(x, *, shape, axes):
    if axes is not None:
        for a in sorted(axes):
            x = x.expand_dims(a)
    out = x.broadcast_to(shape)
    return [out]


@ops.broadcast.set_vmap
def fn(axis_size, vals_in, dims_in, *, shape, axes):
    (x,), (x_bdim,) = vals_in, dims_in
    # x1s = [d for i,d in enumerate(x.shape) if i != x_bdim]
    shape_ = list(shape)
    axes_ = list(axes)
    shape = list(shape)
    axes = [a + int(a >= (x_bdim)) for a in axes]
    if all([a < x_bdim for a in axes]):
        x_bdim += 1

    shape = shape[:x_bdim] + [axis_size] + shape[x_bdim:]
    # if sum(int(a<x_bdim) for a in axes) != 0:
    #     breakpoint()

    return [x.broadcast(shape, axes)], [x_bdim]


@ops.broadcast.set_jvp
def fn(primals, tangents, *, shape, axes):
    (x,), (x_dot,) = primals, tangents
    return (
        [x.broadcast(shape=shape, axes=axes)],
        [x_dot.broadcast(shape=shape, axes=axes)],
    )


@ops.broadcast.set_shape_eval
def fn(x: ArrayShape, *, shape: Sequence[int], axes) -> List[ArrayShape]:
    return [ArrayShape(tuple(shape), x.dtype)]


@ops.broadcast.set_T
def fn(cts, x, *, shape, axes):
    (z,) = cts
    out = z
    out = out.sum(axes, keepdims=False)
    more_axes = []
    for i, (dx, dz) in enumerate(zip(x.aval.shape, z.shape)):
        if dz > dx and i not in axes:
            more_axes += [i]
    out = out.sum(axes=tuple(more_axes), keepdims=True)
    # print(axes, z.shape, x.aval.shape, more_axes, out.shape)
    if out.shape != x.aval.shape:
        breakpoint()
    return [out]


ops.reshape = Op.shape("reshape")


@ops.reshape.set_args_fixer
def fn(x, *, shape):
    if -1 in shape:
        others = math.prod([d for d in shape if d != -1])
        numel = math.prod(x.shape)
        shape = tuple(d if d != -1 else (numel // others) for d in shape)
    return (x,), dict(shape=shape)


@ops.reshape.set_eval
def fn(x, *, shape):
    return [x.reshape(shape)]


@ops.reshape.set_jvp
def fn(primals, tangents, *, shape):
    (x,), (x_dot,) = primals, tangents
    return [x.reshape(shape)], [x_dot.reshape(shape)]


@ops.reshape.set_shape_eval
def fn(x: ArrayShape, *, shape: Sequence[int]) -> List[ArrayShape]:
    return [ArrayShape(tuple(shape), x.dtype)]


@ops.reshape.set_T
def fn(cts, x, *, shape):
    (z,) = cts
    return [z.reshape(x.aval.shape)]


ops.transpose = Op.shape("transpose")


@ops.transpose.set_eval
def fn(x, *, perm):
    return [x.transpose(perm)]


@ops.transpose.set_vmap
def fn(axis_size, vals_in, dims_in, *, perm):
    (x,), (x_bdim,) = vals_in, dims_in
    perm_ = list(perm)
    x_bdim_ = int(x_bdim)
    assert x_bdim >= 0
    # perm = [d - int(i >= x_bdim) for i, d in enumerate(perm)]
    perm = perm[:x_bdim] + [x_bdim] + perm[x_bdim:]
    perm = [d + int(d >= x_bdim) if i != x_bdim else d for i, d in enumerate(perm)]
    assert len(set(perm)) == len(perm)
    # perm[:x_bdim] = perm[:x_bdim][::-1]
    # breakpoint()
    return [x.tranpose(perm)], [x_bdim]


@ops.transpose.set_jvp
def fn(primals, tangents, *, perm):
    (x,), (x_dot,) = primals, tangents
    return [x.transpose(perm)], [x_dot.transpose(perm)]


@ops.transpose.set_shape_eval
def fn(x: ArrayShape, *, perm: Sequence[int]) -> List[ArrayShape]:
    shape = [x.shape[i] for i in perm]
    return [ArrayShape(shape, x.dtype)]


@ops.transpose.set_T
def fn(cts, x, *, perm):
    (z,) = cts
    return [z.transpose(perm)]


ops.pad = Op.shape("pad")


@ops.pad.set_eval
def fn(x, *, lo, hi, interior, value):
    return [x.pad(lo, hi, interior, value)]


@ops.pad.set_args_fixer
def fn(x, *, lo, hi, interior, value):
    if interior is None:
        interior = tuple([0] * len(lo))
    return (x,), dict(lo=lo, hi=hi, interior=interior, value=value)


@ops.pad.set_vmap
def fn(axis_size, vals_in, dims_in, *, perm):
    raise NotImplementedError
    operand, padding_value = batched_args
    operand_bdim, padding_value_bdim = batch_dims
    if operand_bdim is None:
        operand_bdim = 0
        operand = broadcast(operand, (padding_value.shape[padding_value_bdim],))

    padding_config = list(padding_config)
    padding_config.insert(operand_bdim, (0, 0, 0))
    if padding_value_bdim is None:
        return pad(operand, padding_value, padding_config), operand_bdim

    assert padding_value_bdim == 0, padding_value_bdim

    x = pad(operand, _zero(operand), padding_config)
    mask = pad(full_like(operand, True, np.bool_), False, padding_config)
    broadcasted_padding = broadcast_in_dim(padding_value, x.shape, (operand_bdim,))
    return select(mask, x, broadcasted_padding), operand_bdim


@ops.pad.set_jvp
def fn(primals, tangents, *, lo, hi, interior, value):
    (x,), (x_dot,) = primals, tangents
    return [x.pad(lo, hi, interior, value)], [x_dot.pad(lo, hi, interior, value)]


@ops.pad.set_shape_eval
def fn(x: ArrayShape, *, lo, hi, interior, value) -> List[ArrayShape]:
    op_shape = np.shape(x)

    def _dilate_dim(d, dilation):
        return 0 if d == 0 else 1 + dilation * (d - 1)

    shape = (
        sum([l, h, _dilate_dim(d, r + 1)])
        for l, h, r, d in zip(lo, hi, interior, op_shape)
    )
    res = ArrayShape(shape, x.dtype)
    if not all(d >= 0 for d in res.shape):
        raise ValueError(
            f"Dimension size after padding is not at least 0, "
            f"got result shape {res}, for {lo=} {hi=} {interior=} {value=}"
            f"{op_shape=}"
        )
    return [res]


@ops.pad.set_T
def fn(cts, x, *, lo, hi, interior, value):
    (z,) = cts

    def t_op():
        unpadded = z.slice(
            lo,
            tuple(s - h for s, h in zip(z.shape, hi)),
            tuple([1] * len(interior)),
        )
        return unpadded.slice(
            tuple([0] * len(lo)), unpadded.shape, tuple(r + 1 for r in interior)
        )

    res = t_op() if isinstance(x, UndefPrimal) else None
    return [res]


ops.slice = Op.shape("shape")


@ops.slice.set_eval
def fn(x, *, starts, limits, strides):
    return [x.slice(starts, limits, strides)]


@ops.slice.set_vmap
def fn(axis_size, vals_in, dims_in, *, starts, limits, strides):
    raise NotImplementedError
    (x,) = vals_in
    (x_bdim,) = dims_in

    new_start_indices = list(starts)
    new_start_indices.insert(x_bdim, 0)

    new_limit_indices = list(limits)
    new_limit_indices.insert(x_bdim, x.shape[x_bdim])

    if strides is None:
        new_strides = None
    else:
        new_strides = list(strides)
        new_strides.insert(x_bdim, 1)

    out = x.slice(new_start_indices, new_limit_indices, new_strides)
    return out, x_bdim


@ops.slice.set_jvp
def fn(primals, tangents, *, starts, limits, strides):
    (x,), (x_dot,) = primals, tangents
    return [x.slice(starts, limits, strides)], [x_dot.slice(starts, limits, strides)]


@ops.slice.set_shape_eval
def fn(x: ArrayShape, *, starts, limits, strides: Sequence[int]) -> List[ArrayShape]:
    if strides is None or tuple(strides) == (1,) * len(x.shape):
        shape = [
            limit if type(start) is int and start == 0 else limit - start
            for start, limit in zip(starts, limits)
        ]
        return [ArrayShape(shape, x.dtype)]
    else:
        # TODO: compute strided shape without numpy
        x = np.zeros_like(x.shape)
        x = x[tuple(slice(s, l, r) for s, l, r in zip(starts, limits, strides))]
        return [ArrayShape(x.shape, x.dtype)]


@ops.slice.set_T
def T(cts, x, *, starts, limits, strides):
    # TODO: compute tuple arithmetic numpy
    (z,) = cts
    x_shape = x.aval.shape
    assert isinstance(x, UndefPrimal)
    if strides is None or np.all(np.equal(strides, 1)):
        lo, hi, interior = (
            starts,
            np.subtract(x.aval.shape, limits),
            (0,) * len(starts),
        )
    else:
        real_limits = np.add(
            starts,
            np.where(
                np.array(x.shape) == 0,
                0,
                np.add(1, np.multiply(np.subtract(t.shape, 1), strides)),
            ),
        )
        lo, hi, interior = list_zip(
            starts, np.subtract(x_shape, real_limits), np.subtract(strides, 1)
        )
    res = z.pad(lo, hi, interior)
    assert res.shape == x_shape, f"{res.shape=} {x_shape=}"
    return [res]


ops.flip = Op.shape("flip")


@ops.flip.set_eval
def fn(x, *, axes):
    return [x.flip(axes)]


@ops.flip.set_vmap
def fn(axis_size, vals_in, dims_in, *, perm):
    raise NotImplementedError


@ops.flip.set_jvp
def fn(primals, tangents, *, axes):
    (x,), (x_dot,) = primals, tangents
    return [x.flip(axes)], [x_dot.flip(axes)]


@ops.flip.set_shape_eval
def fn(x: ArrayShape, *, padding: Sequence[int]) -> List[ArrayShape]:
    shape = [x.shape[i] for i in padding]
    return [ArrayShape(shape, x.dtype)]


@ops.flip.set_T
def T(cts, *, axes):
    (z,) = cts
    return [z.flip(axes)]


ops.concatenate = Op.shape("concatenate")


@ops.concatenate.set_eval
def fn(xs: Sequence[Any], *, axis):
    return [Array.concatenate(xs, axis=axis)]


@ops.concatenate.set_vmap
def fn(axis_size, vals_in, dims_in, *, perm):
    raise NotImplementedError


@ops.concatenate.set_jvp
def jvp(primals, tangents, *, axis):
    (xs,), (xs_dot,) = primals, tangents
    return [Array.concatenate(xs, axis=axis)], [Array.concatenate(xs_dot, axis=axis)]


@ops.concatenate.set_shape_eval
def fn(x: ArrayShape, idx, *, axis: Sequence[int]) -> List[ArrayShape]:
    shape = [x.shape[i] for i in axis]
    return [ArrayShape(shape, x.dtype)]


@ops.concatenate.set_T
def T(cts, xs, *, axis):
    (zs,) = cts
    return [zs.slice(x.shape) for x in xs]


# -----------------------
# LoadOps
# -----------------------

ops.constant = Op.load("constant")


@ops.constant.set_load_fn
def fn(*, val, dtype):
    return Array(val, dtype)


@ops.constant.set_shape_eval
def fn(*, val, dtype) -> List[ArrayShape]:
    # TODO: not using numpy to extract shape
    return [ArrayShape(np.array(val).shape, dtype)]


ops.full = Op.load("full")


@ops.full.set_load_fn
def fn(*, fill_value, shape, dtype):
    return Array.full(fill_value, shape, dtype)


@ops.full.set_shape_eval
def fn(*, fill_value, shape, dtype) -> List[ArrayShape]:
    return [ArrayShape(tuple(shape), dtype)]


ops.random_uniform = Op.load("random_uniform")


@ops.random_uniform.set_load_fn
def fn(*, shape, dtype):
    return Array.random_uniform(shape, dtype)


@ops.random_uniform.set_shape_eval
def fn(*, shape, dtype) -> List[ArrayShape]:
    return [ArrayShape(tuple(shape), dtype)]


ops.random_normal = Op.load("random_normal")


@ops.random_normal.set_load_fn
def fn(*, shape, dtype):
    return Array.random_normal(shape, dtype)


@ops.random_normal.set_shape_eval
def fn(*, shape, dtype) -> List[ArrayShape]:
    return [ArrayShape(tuple(shape), dtype)]


ops.arange = Op.load("arange")


@ops.arange.set_load_fn
def fn(*, start, stop, stride, dtype):
    return Array.arange(start, stop, stride, dtype)


@ops.arange.set_shape_eval
def fn(*, start, stop, stride, dtype):
    return [ArrayShape(len(tuple(slice(start, stop, stride))), dtype)]


ops.jit = Op("jit")


@ops.jit.set_eval
def fn(*args, hashable_prog, hashable_consts):
    jit_fn = RT.backend.callable(hashable_prog, hashable_consts)
    return [jit_fn(*args)]


# Functions
default_dtype = Array.default_dtype


@procs.add
@classmethod
def zeros(cls, shape, dtype=default_dtype, **kwargs):
    return cls.full(shape, 0.0, dtype, **kwargs)


@procs.add
@classmethod
def ones(cls, shape, dtype=default_dtype, **kwargs):
    return cls.full(shape, 1.0, dtype, **kwargs)


@procs.add
@classmethod
def full_like(cls, other, fill_value, **kwargs):
    return cls.full(other.shape, fill_value, dtype=other.dtype, **kwargs)


@procs.add
@classmethod
def zeros_like(cls, other, **kwargs):
    return cls.zeros(other.shape, dtype=other.dtype, **kwargs)


@procs.add
@classmethod
def ones_like(cls, other, **kwargs):
    return cls.full(other.shape, fill_value=1.0, dtype=other.dtype, **kwargs)


@procs.add
def where(self, trueval, falseval):
    cond = self != 0.0
    cond = cond.convert(trueval.dtype)  # TODO: type promotion logic
    return cond * trueval + (1.0 - cond) * falseval


@procs.add
def pow(self, y):
    assert type(y) is int
    if y == 0:
        return self.ones_like(x)
    is_reciprocal = y < 0
    if is_reciprocal:
        y = -y
    acc = None
    while y > 0:
        if y & 1:
            acc = x if acc is None else acc * x
        y >>= 1
        if y > 0:
            x = x * x
    ret = acc
    if is_reciprocal:
        ret = self.ones_like(acc) / acc
    return ret


@procs.add
def cross_entropy(x, y):
    return x * y.log()


@procs.add
def mse(x, y):
    return pow((x - y), 2)


@procs.add
def mean(self, axes=None, keepdims=False):
    out = self.sum(axes=axes, keepdim=keepdims)
    return out * (math.prod(out.shape) / math.prod(self.shape))


@procs.add
def minimum(self, other):
    return -self.maximum(-self, -other)


@procs.add
def min(self, axes=None, keepdims=False):
    return -((-self).max(self, axes, keepdims))


@procs.add
def flatten(self, start_dim=0):
    return self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))


@procs.add
@classmethod
def glorot_uniform(cls, *shape, **kwargs):
    return cls.rand(*shape, **kwargs).mul(
        (6 / (shape[0] + math.prod(shape[1:]))) ** 0.5
    )


@procs.add
def T(self):
    perm = list(range(self.ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return self.transpose(perm)


@procs.add
def _softmax(self, axes):
    m = self - self.max(axes, keepdims=True)
    e = m.exp()
    return m, e, e.sum(axes, keepdims=True)


@procs.add
def softmax(self, axes=-1):
    _, e, ss = self._softmax(axes)
    return e.div(ss)


@procs.add
def log_softmax(self, axes=-1):
    m, _, ss = self._softmax(axes)
    return m - ss.log()


@procs.add
def dot(self, w):
    x = self.reshape((*self.shape[0:-1], 1, self.shape[-1]))
    w = w.reshape((*w.shape[0:-2], 1, w.shape[-2], w.shape[-1])).T
    return (x * w).sum(-1).reshape((*x.shape[0:-2], -1))


@procs.add
def square(self):
    return self * self


@procs.add
def clip(self, min_, max_):
    return ((self - min_).relu() + min_) - (self - max_).relu()


@procs.add
def abs(self):
    return self.relu() + (-self).relu()


@procs.add
def sign(self):
    return self / (self.abs() + 1e-10)


@procs.add
def reciprocal(self):
    return 1.0 / self


numpy_backend = Backend("numpy")
for typ in [bool, int, float, np.ndarray, np.float64, np.float32]:
    numpy_backend.set_input_handler(typ, np.asarray)


@numpy_backend.set_compile
def fn(self, prog, consts, in_avals, name) -> List[Any]:
    safe_builtins = {"math": math, "np": np, "pickle": pickle}

    exec_locals = {}
    env: Dict[Var, Any] = {}
    ncs = 0
    nxs = 0
    nzs = 0
    inb_args = []
    inb_consts = []

    for inb in prog.in_binders:
        if type(inb.aval) is not ArrayShape:
            env[inb] = f"c{ncs}"
            inb_consts += [env[inb]]
            ncs += 1
        else:
            env[inb] = f"x{nxs}"
            inb_args += [env[inb]]
            nxs += 1
    code_lines = []
    code_lines += [f"def {name}({', '.join(inb_args)}):"]
    for inb_const, const in zip(inb_consts, consts):
        code_lines += [f"    {inb_const} = pickle.loads({pickle.dumps(const.val)})"]
    multiline_op_impl_set = set()
    multiline_op_impl_defs = []
    for eqn in prog.instrs:
        in_vals = list_map(lambda x: env[x], eqn.inputs)
        for outb in eqn.out_binders:
            env[outb] = f"z{nzs}"
            nzs += 1
        out_vals = list_map(lambda z: env[z], eqn.out_binders)
        assert not len(out_vals) > 1, "Op with >1 output not supported"
        # op_ir = self.op_impls[eqn.op.name].ir(*in_vals, **eqn.params, ret=out_vals[0])
        impl = RT.ops[eqn.op.name].impls[self.name]
        op_impl_code_lines = inspect.getsourcelines(impl)
        args_str = ", ".join(in_vals)
        kwargs_str = ", ".join([f"{k}={v}" for k, v in eqn.params.items()])
        if len(op_impl_code_lines) > 2:
            if eqn.op.name not in multiline_op_impl_set:
                multiline_op_impl_set.add(eqn.op.name)
                multiline_op_impl_defs += [op_impl_code_lines]
            code_line += f"{out_vals[0]} = {eqn.op.name}({args_str}, {kwargs_str})"
        else:
            argspec = inspect.getargspec(impl)
            op_str = op_impl_code_lines[1].replace("return", "").strip()
            for argname, arg in zip(argspec.args, in_vals):
                op_str.replace(argname, arg)
            for kwargname, kwarg in eqn.params.items():
                op_str.replace(kwargname, kwarg)
            code_line = f"{out_vals[0]} = {op_str}"

        code_line = "\n".join(["    " + line for line in code_line.strip().split("\n")])
        code_lines += [code_line]
        # out_vals = eqn.op.jit(in_avals, in_vals, **eqn.params)

    outs = list_map(lambda y: env[y], prog.outs)
    # ops_code += [f"    outs[0]}"]
    code_lines += [f"    return {', '.join(outs)}{',' if len(outs)==1 else ''}"]
    code_lines = multiline_op_impl_defs + code_lines
    code = "\n".join(code_lines)
    exec(compile(code, "<string>", "exec"), safe_builtins, exec_locals)
    fn = exec_locals[name]
    # exec('\n'.join(ops_code), safe_builtins, exec_locals)
    return JitFn(code, fn)


# _rng: np.random.Generator = np.random.default_rng()

# @classmethod
# def manual_seed(cls, seed=None):
#     cls._rng = np.random.default_rng(seed=seed)

# # control flow
# choose = select = lambda arr, *vals, idx: Array(np.choose(idx, *vals))
# where = lambda arr, trueval, falseval: Array(np.where(arr, trueval, falseval))

### Op Impls


@numpy_backend.set_impl(ops.convert)
def fn(
    x,
    *,
    dtype,
):
    return np.astype(x, dtype)


@numpy_backend.set_impl(ops.stop_gradient)
def fn(x, dtype):
    return x


@numpy_backend.set_impl(ops.neg)
def fn(
    x,
):
    return np.neg(x)


@numpy_backend.set_impl(ops.sqrt)
def fn(x):
    return np.sqrt(x)


@numpy_backend.set_impl(ops.exp)
def fn(x):
    return np.exp(x)


@numpy_backend.set_impl(ops.log)
def fn(x):
    return np.log(x)


@numpy_backend.set_impl(ops.add)
def fn(x, y):
    return np.add(x, y)


@numpy_backend.set_impl(ops.sub)
def fn(x, y):
    return np.subtract(x, y)


@numpy_backend.set_impl(ops.mul)
def fn(x, y):
    return np.multiply(x, y)


@numpy_backend.set_impl(ops.div)
def fn(x, y):
    return np.divide(x, y)


@numpy_backend.set_impl(ops.constant)
def fn(*, val, dtype):
    return np.array(val, dtype=dtype)


@numpy_backend.set_impl(ops.arange)
def fn(*, start, stop, stride, dtype):
    return np.arange(start, stop, stride, dtype=dtype)


@numpy_backend.set_impl(ops.full)
def fn(*, fill_value, dtype):
    return np.full(fill_value=fill_value, dtype=dtype)


@numpy_backend.set_impl(ops.random_uniform)
def fn(*, shape, dtype):
    return np.random.uniform(size=shape, dtype=dtype)


@numpy_backend.set_impl(ops.random_normal)
def fn(*, shape, dtype):
    return np.random.normal(size=shape, dtype=dtype)


@numpy_backend.set_impl(ops.broadcast)
def fn(x, *, shape, axes):
    ret = x
    if not axes is None:
        for a in sorted(axes):
            ret = np.expand_dims(ret, a)
    ret = np.broadcast_to(ret, shape)
    return ret


@numpy_backend.set_impl(ops.reshape)
def fn(x, *, shape):
    return np.reshape(x, shape)


@numpy_backend.set_impl(ops.pad)
def fn(x, *, lo, hi, interior, value):
    return np.pad({x}, list(zip(lo, hi)), constant_values={value})


@numpy_backend.set_impl(ops.slice)
def fn(x, *, starts, limits, strides):
    return x[[slice(s, l, st) for s, l, st in zip(starts, limits, strides)]]


@numpy_backend.set_impl(ops.concatenate)
def fn(xs, *, axes):
    return np.concatenate(xs, axes)


@numpy_backend.set_impl(ops.transpose)
def fn(x, *, axes):
    return np.transpose(x, axes)


@numpy_backend.set_impl(ops.flip)
def fn(x, *, axes):
    return np.flip(x, axes)


backend = numpy_backend


class PPrint:
    lines: List[Tuple[int, str]]

    def __init__(self, lines):
        self.lines = lines

    def indent(self, indent: int) -> "PPrint":
        return PPrint([(indent + orig_indent, s) for orig_indent, s in self.lines])

    def __add__(self, rhs: "PPrint") -> "PPrint":
        return PPrint(self.lines + rhs.lines)

    def __rshift__(self, rhs: "PPrint") -> "PPrint":
        if not rhs.lines:
            return self
        if not self.lines:
            return rhs
        indent, s = self.lines[-1]
        indented_block = rhs.indent(indent + len(s))
        common_line = s + " " * rhs.lines[0][0] + rhs.lines[0][1]
        return PPrint(
            self.lines[:-1] + [(indent, common_line)] + indented_block.lines[1:]
        )

    def __str__(self) -> str:
        return "\n".join(" " * indent + s for indent, s in self.lines)

    def __repr__(self):
        return str(self)

    @classmethod
    def pp(cls, s: Any):
        return cls([(0, line) for line in str(s).splitlines()])

    @classmethod
    def vcat(cls, ps: List["PPrint"]):
        return sum(ps, cls.pp(""))


# def mapped_aval(batch_dim, aval):
#     shape = list(aval.shape)
#     del shape[batch_dim]
#     return ArrayShape(tuple(shape), aval.dtype)

BatchAxis = Union[None, int]


class BatchTracerArray(TracerArray):
    def __init__(self, trace, val, batch_dim: BatchAxis):
        self._trace = trace
        self.val = val
        self.batch_dim = batch_dim

    @property
    def aval(self):
        aval = self.get_aval(self.val)
        if self.batch_dim is None:
            return aval
        else:
            shape = list(aval.shape)
            del shape[self.batch_dim]
            return ArrayShape(tuple(shape), aval.dtype)

    def full_lower(self):
        if self.batch_dim is None:
            return RT.full_lower(self.val)
        else:
            return self


class BatchTrace(Trace):
    pure = lift = lambda self, val: BatchTracerArray(self, val, None)

    def run_op(self, op, tracers, params):
        vals_in, bdims_in = unzip2((t.val, t.batch_dim) for t in tracers)
        val_outs, bdim_outs = op.vmap(self.axis_size, vals_in, bdims_in, **params)
        return [
            BatchTracerArray(self, x, bd) for x, bd in list_zip(val_outs, bdim_outs)
        ]

    @property
    def axis_size(self):
        return self.main.global_data


def move_batch_axis(axis_size, src, dst, x):
    if src is None:
        target_shape = list(x.shape)
        target_shape.insert(dst, axis_size)
        out_ndim = len(target_shape)
        if type(dst) in (tuple, list):
            out_ndim += 1
        reshape_shape = [1 if ax == dst else target_shape for ax in range(out_ndim)]
        x = x.reshape(reshape_shape)
        x = x.broadcast(target_shape)
        return x
    elif src == dst:
        return x
    else:
        perm = [i for i in range(len(x.shape)) if i != src]
        perm.insert(dst, src)
        return x.transpose(perm)


def vmap_flat(f, in_axes, *args):
    axis_set = {x.shape[ax] for x, ax in list_zip(args, in_axes) if ax is not None}
    assert len(axis_set) == 1
    (axis_size,) = axis_set
    with RT.new_main(BatchTrace, axis_size) as main:
        trace = BatchTrace(main)
        tracers_in = [
            BatchTracerArray(trace, x, ax) if ax is not None else x
            for x, ax in list_zip(args, in_axes)
        ]
        outs = f(*tracers_in)
        tracers_out = [RT.full_raise(trace, out) for out in outs]
        vals_out, bdims_out = unzip2((t.val, t.batch_dim) for t in tracers_out)
    outs_transposed = [
        move_batch_axis(axis_size, bdim, 0, val_out)
        for val_out, bdim in list_zip(vals_out, bdims_out)
    ]
    return outs_transposed


def vmap(f, in_axes):
    def batched_f(*args):
        args_flat, in_tree = tree_flatten(args)
        in_axes_flat, in_tree2 = tree_flatten(in_axes)
        if in_tree != in_tree2:
            raise TypeError(f"{in_tree}\n!=\n{in_tree2}")
        f_flat, out_tree = flatten_fun(f, in_tree)
        outs_flat = vmap_flat(f_flat, in_axes_flat, *args_flat)
        return tree_unflatten(out_tree(), outs_flat)

    return batched_f


class JVPTracerArray(TracerArray):
    def __init__(self, trace, primal, tangent):
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def aval(self):
        return self.get_aval(self.primal)


class JVPTrace(Trace):
    def pure(self, val):
        aval = TracerArray.get_aval(val)
        val = aval if not isinstance(aval, TracerArray) else val

        return JVPTracerArray(self, val, Array.zeros(aval.shape, aval.dtype))

    lift = pure

    def run_op(self, op, tracers, params):
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        primal_outs, tangent_outs = op.jvp(primals_in, tangents_in, **params)
        return [
            JVPTracerArray(self, x, t) for x, t in list_zip(primal_outs, tangent_outs)
        ]


def jvp_flat(f, primals, tangents):
    with RT.new_main(JVPTrace) as main:
        trace = JVPTrace(main)
        tracers_in = [
            JVPTracerArray(trace, x, t) for x, t in list_zip(primals, tangents)
        ]
        outs = f(*tracers_in)
        tracers_out = [RT.full_raise(trace, out) for out in outs]
        primals_out, tangents_out = unzip2((t.primal, t.tangent) for t in tracers_out)
    return primals_out, tangents_out


def jvp(f, primals, tangents):
    primals_flat, in_tree = tree_flatten(primals)
    tangents_flat, in_tree2 = tree_flatten(tangents)
    if in_tree != in_tree2:
        raise TypeError
    f, out_tree = flatten_fun(f, in_tree)
    primals_out_flat, tangents_out_flat = jvp_flat(f, primals_flat, tangents_flat)
    primals_out = tree_unflatten(out_tree(), primals_out_flat)
    tangents_out = tree_unflatten(out_tree(), tangents_out_flat)
    return primals_out, tangents_out


def jacfwd(f, x):
    pushfwd = lambda v: jvp(f, (x,), (v,))[1]
    vecs_in = Array.eye(math.prod(x.shape)).reshape(x.shape * 2)
    return vmap(pushfwd, (0,))(vecs_in)


class Var:
    val = None
    aval: ArrayShape

    def __init__(self, aval):
        self.aval = aval


class Lit:
    val: Any
    aval: ArrayShape

    def __init__(self, val):
        self.aval = aval = ArrayShape.like(TracerArray.get_aval(val))
        self.val = np.array(val, aval.dtype)


Atom = Union[Var, Lit]


class ProgEqn(NamedTuple):
    op: Op
    inputs: List[Atom]
    params: Dict[str, Any]
    out_binders: List[Atom]


class Prog(NamedTuple):
    in_binders: Any
    instrs: List[ProgEqn]
    outs: Any

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        namegen = (
            "".join(s)
            for r in itertools.count(1)
            for s in itertools.permutations(string.ascii_lowercase, r)
        )
        names = defaultdict(lambda: next(namegen))
        in_binders = ", ".join(self.var_str(names, x) for x in self.in_binders)
        instrs = PPrint.vcat([self.pp_instr(names, e) for e in self.instrs])
        outs = ", ".join(
            names[v] if isinstance(v, Var) else str(v.val) for v in self.outs
        )
        return str(
            PPrint.pp(f"{{ lambda {in_binders} .")
            + ((PPrint.pp("let ") >> instrs) + PPrint.pp(f"in ( {outs} ) }}")).indent(2)
        )

    def pp_instr(self, names: DefaultDict[Var, str], instr: ProgEqn) -> PPrint:
        # rule = instr.op.pprint
        # if rule() is not None:
        #     return rule(names, instr)
        # else:
        lhs = PPrint.pp(" ".join(self.var_str(names, v) for v in instr.out_binders))
        rhs = (
            PPrint.pp(repr(instr.op.__class__))
            >> self.pp_params(instr.params)
            >> PPrint.pp(
                " ".join(
                    names[x] if isinstance(x, Var) else str(x.val) for x in instr.inputs
                )
            )
        )
        return lhs >> PPrint.pp(" = ") >> rhs

    def pp_params(self, params: Dict[str, Any]) -> PPrint:
        items = sorted(params.items())
        if items:
            return (
                PPrint.pp(" [ ")
                >> PPrint.vcat([PPrint.pp(f"{k}={v}") for k, v in items])
                >> PPrint.pp(" ] ")
            )
        else:
            return PPrint.pp(" ")

    def var_str(self, names: DefaultDict[Var, str], v) -> str:
        return f"{names[v]}:{v.aval.str_short()}"


class ProgType(NamedTuple):
    in_types: List[ArrayShape]
    out_types: List[ArrayShape]

    def __repr__(self):
        in_types = ", ".join(aval.str_short() for aval in self.in_types)
        out_types = ", ".join(aval.str_short() for aval in self.out_types)
        return f"({in_types}) -> ({out_types})"


def typecheck_prog(prog: Prog) -> ProgType:
    env: Set[Var] = set()

    for v in prog.in_binders:
        if v in env:
            raise TypeError
        env.add(v)

    for instr in prog.instrs:
        in_types = [typecheck_atom(env, x) for x in instr.inputs]
        out_types = instr.op.shape_eval(*in_types, **instr.params)
        for out_binder, out_type in list_zip(instr.out_binders, out_types):
            if not out_type == out_binder.aval:
                raise TypeError
        for out_binder in instr.out_binders:
            if out_binder in env:
                raise TypeError
            env.add(out_binder)

    in_types = [v.aval for v in prog.in_binders]
    out_types = [typecheck_atom(env, x) for x in prog.outs]
    return ProgType(in_types, out_types)


def typecheck_atom(env: Set[Var], x: Atom) -> ArrayShape:
    if isinstance(x, Var):
        if x not in env:
            raise TypeError("unbound variable")
        return x.aval
    elif isinstance(x, Lit):
        return TracerArray.get_aval(x.val)
    else:
        assert False


def eval_prog(prog: Prog, args: List[Any]) -> List[Any]:
    env: Dict[Var, Any] = {}

    def read(x: Atom) -> Any:
        return env[x] if type(x) is Var else x.val

    def write(v: Var, val: Any) -> None:
        assert v not in env  # single-assignment
        env[v] = val

    list_map(write, prog.in_binders, args)
    for instr in prog.instrs:
        in_vals = list_map(read, instr.inputs)
        outs = RT.bind(instr.op, *in_vals, **instr.params)
        list_map(write, instr.out_binders, outs)
    return list_map(read, prog.outs)


def prog_as_fun(prog: Prog):
    return lambda *args: eval_prog(prog, args)


class ProgTracerArray(TracerArray):
    __slots__ = ["aval"]
    aval: ArrayShape

    def __init__(self, trace, aval):
        self._trace = trace
        self.aval = aval


class ProgTrace(Trace):
    def new_arg(self, aval) -> ProgTracerArray:
        aval = ArrayShape.like(aval)
        tracer = self.builder.new_tracer(self, aval)
        self.builder.tracer_to_var[id(tracer)] = Var(aval)

        return tracer

    def get_or_make_const_tracer(self, val: Any) -> ProgTracerArray:
        tracer = self.builder.const_tracers.get(id(val))
        if tracer is None:
            tracer = self.builder.new_tracer(self, TracerArray.get_aval(val))
            self.builder.add_const(tracer, val)
        return tracer

    pure = lift = get_or_make_const_tracer

    def run_op(self, op, tracers, params):
        avals_in = [t.aval for t in tracers]
        avals_out = op.shape_eval(*avals_in, **params)
        out_tracers = [self.builder.new_tracer(self, a) for a in avals_out]
        inputs = [self.builder.getvar(t) for t in tracers]
        outvars = [self.builder.add_var(t) for t in out_tracers]
        self.builder.add_instr(ProgEqn(op, inputs, params, outvars))
        return out_tracers

    @property
    def builder(self):
        return self.main.global_data


class ProgBuilder:
    instrs: List[ProgEqn]
    tracer_to_var: Dict[int, Var]
    const_tracers: Dict[int, TracerArray]
    constvals: Dict[Var, Any]
    tracers: List[ProgTracerArray]

    def __init__(self):
        self.instrs = []
        self.tracer_to_var = {}
        self.const_tracers = {}
        self.constvals = {}
        self.tracers = []

    def new_tracer(self, trace: ProgTrace, aval: ArrayShape) -> ProgTracerArray:
        tracer = ProgTracerArray(trace, aval)
        self.tracers.append(tracer)
        return tracer

    def add_instr(self, instr: ProgEqn) -> None:
        self.instrs.append(instr)

    def add_var(self, tracer: ProgTracerArray) -> Var:
        assert id(tracer) not in self.tracer_to_var
        var = self.tracer_to_var[id(tracer)] = Var(tracer.aval)
        return var

    def getvar(self, tracer: ProgTracerArray) -> Var:
        var = self.tracer_to_var.get(id(tracer))
        assert var is not None
        return var

    def add_const(self, tracer: ProgTracerArray, val: Any) -> Var:
        var = self.add_var(tracer)
        self.const_tracers[id(val)] = tracer
        self.constvals[var] = val
        return var

    def build(self, in_tracers: Any, out_tracers: Any) -> Tuple[Prog, List[Any]]:
        constvars, constvals = unzip2(self.constvals.items())
        t2v = lambda t: self.tracer_to_var[id(t)]
        in_binders = constvars + [t2v(t) for t in in_tracers]
        out_vars = [t2v(t) for t in out_tracers]
        prog = Prog(in_binders, self.instrs, out_vars)
        typecheck_prog(prog)
        prog, constvals = self._inline_literals(prog, constvals)
        return prog, constvals

    def _inline_literals(self, prog: Prog, consts: List[Any]) -> Tuple[Prog, List[Any]]:
        const_binders, other_binders = split_list(prog.in_binders, len(consts))
        scalars = [
            type(x) in TracerArray.TYPES and not TracerArray.get_aval(x).shape
            for x in consts
        ]
        new_const_binders, lit_binders = partition_list(scalars, const_binders)
        new_consts, lit_vals = partition_list(scalars, consts)
        literals = dict(zip(lit_binders, list_map(Lit, lit_vals)))
        new_instrs = [
            ProgEqn(
                instr.op,
                [literals.get(x, x) for x in instr.inputs],
                instr.params,
                instr.out_binders,
            )
            for instr in prog.instrs
        ]
        new_outs = [literals.get(x, x) for x in prog.outs]
        new_prog = Prog(new_const_binders + other_binders, new_instrs, new_outs)
        typecheck_prog(new_prog)
        return new_prog, new_consts


@lru_cache()
def make_prog(
    f: Callable,
    *avals_in: ArrayShape,
) -> Tuple[Prog, List[Any], PyTreeDef]:
    avals_in, in_tree = tree_flatten(avals_in)
    f, out_tree = flatten_fun(f, in_tree)

    builder = ProgBuilder()
    with RT.new_main(ProgTrace, builder) as main:
        with RT.new_dynamic(main):
            trace = ProgTrace(main)
            tracers_in = [trace.new_arg(aval) for aval in avals_in]
            outs = f(*tracers_in)
            tracers_out = [RT.full_raise(trace, out) for out in outs]
            prog, consts = builder.build(tracers_in, tracers_out)
    return prog, consts, out_tree()


def linearize_flat(f, *primals_in):
    pvals_in = [PartialVal.known(x) for x in primals_in] + [
        PartialVal.unknown(ArrayShape.like(TracerArray.get_aval(x))) for x in primals_in
    ]

    def f_jvp(*primals_tangents_in):
        primals_out, tangents_out = jvp(f, *split_half(primals_tangents_in))
        return [*primals_out, *tangents_out]

    prog, pvals_out, consts = partial_eval_flat(f_jvp, pvals_in)
    primal_pvals, _ = split_half(pvals_out)
    assert all(pval.is_known for pval in primal_pvals)
    primals_out = [pval.const for pval in primal_pvals]
    f_lin = lambda *tangents: eval_prog(prog, [*consts, *tangents])
    return primals_out, f_lin


def linearize(f, *primals_in):
    primals_in_flat, in_tree = tree_flatten(primals_in)
    f, out_tree = flatten_fun(f, in_tree)
    primals_out_flat, f_lin_flat = linearize_flat(f, *primals_in_flat)
    primals_out = tree_unflatten(out_tree(), primals_out_flat)

    def f_lin(*tangents_in):
        tangents_in_flat, in_tree2 = tree_flatten(tangents_in)
        if in_tree != in_tree2:
            raise TypeError
        tangents_out_flat = f_lin_flat(*tangents_in_flat)
        return tree_unflatten(out_tree(), tangents_out_flat)

    return primals_out, f_lin


class PartialVal(NamedTuple):
    aval: ArrayShape
    const: Optional[Any]

    @classmethod
    def known(cls, val: Any):
        return PartialVal(TracerArray.get_aval(val), val)

    @classmethod
    def unknown(cls, aval: ArrayShape):
        return PartialVal(aval, None)

    is_known = property(lambda self: self.const is not None)
    is_unknown = property(lambda self: self.const is None)


def partial_eval_flat(
    f: Callable, pvals_in: List[PartialVal]
) -> Tuple[Prog, List[PartialVal], List[Any]]:
    with RT.new_main(PartialEvalTrace) as main:
        trace = PartialEvalTrace(main)
        tracers_in = [trace.new_arg(pval) for pval in pvals_in]
        outs = f(*tracers_in)
        tracers_out = [RT.full_raise(trace, out) for out in outs]
        pvals_out = [t.pval for t in tracers_out]
        unk_tracers_in = [t for t in tracers_in if t.pval.is_unknown]
        unk_tracers_out = [t for t in tracers_out if t.pval.is_unknown]
        prog, consts = tracers_to_prog(unk_tracers_in, unk_tracers_out)

    return prog, pvals_out, consts


class LambdaBindingProto(NamedTuple):
    pass


class ConstProto(NamedTuple):
    val: Any


class ProgEqnProto(NamedTuple):
    prim: Op
    tracers_in: List["PartialEvalTracerArray"]
    params: Dict[str, Any]
    avals_out: List[ArrayShape]
    # tracer_refs_out: List[weakref.ReferenceType["PartialEvalTracerArray"]]
    tracer_refs_out: List


ProgProto = Union[LambdaBindingProto, ConstProto, ProgEqnProto]


class PartialEvalTracerArray(TracerArray):
    pval: PartialVal
    proto: Optional[ProgProto]

    def __init__(self, trace, pval, proto):
        self._trace = trace
        self.pval = pval
        self.proto = proto

    aval = property(lambda self: self.pval.aval)

    def full_lower(self):
        if self.pval.is_known:
            return RT.full_lower(self.pval.const)
        return self


class PartialEvalTrace(Trace):
    def new_arg(self, pval: PartialVal) -> Any:
        return PartialEvalTracerArray(self, pval, LambdaBindingProto())

    def lift(self, val: Any) -> PartialEvalTracerArray:
        val = Array(val)
        return PartialEvalTracerArray(self, PartialVal.known(val), None)

    pure = lift

    def instantiate_const(
        self, tracer: PartialEvalTracerArray
    ) -> PartialEvalTracerArray:
        if tracer.pval.is_unknown:
            return tracer
        else:
            pval = PartialVal.unknown(ArrayShape.like(tracer.aval))
            return PartialEvalTracerArray(self, pval, ConstProto(tracer.pval.const))

    def run_op(self, op, tracers, params):
        if all(t.pval.is_known for t in tracers):
            return RT.bind(op, *map(RT.full_lower, tracers), **params)
        tracers_in = [self.instantiate_const(t) for t in tracers]
        avals_in = [t.aval for t in tracers_in]
        avals_out = op.shape_eval(*avals_in, **params)
        tracers_out = [
            PartialEvalTracerArray(self, PartialVal.unknown(aval), None)
            for aval in avals_out
        ]
        instr = ProgEqnProto(
            op, tracers_in, params, avals_out, map(weakref.ref, tracers_out)
        )
        for t in tracers_out:
            t.proto = instr
        return tracers_out


def tracers_to_prog(
    tracers_in: List[PartialEvalTracerArray], tracers_out: List[PartialEvalTracerArray]
):
    tracer_to_var: Dict[int, Var] = {
        id(t): Var(ArrayShape.like(t.aval)) for t in tracers_in
    }
    constvar_to_val: Dict[int, Any] = {}
    constid_to_var: Dict[int, Var] = {}
    processed_instrs: Set[int] = set()
    instrs: List[ProgEqn] = []
    for t in toposort(tracers_out, tracer_parents):
        if isinstance(t.proto, LambdaBindingProto):
            assert id(t) in set(list_map(id, tracers_in))
        elif isinstance(t.proto, ConstProto):
            val = t.proto.val
            var = constid_to_var.get(id(val))
            if var is None:
                aval = ArrayShape.like(TracerArray.get_aval(val))
                var = constid_to_var[id(val)] = Var(aval)
                constvar_to_val[var] = val
            tracer_to_var[id(t)] = var
        elif isinstance(t.proto, ProgEqnProto):
            if id(t.proto) not in processed_instrs:
                instrs.append(proto_to_instr(tracer_to_var, t.proto))
                processed_instrs.add(id(t.proto))
        else:
            raise TypeError(t.proto)

    constvars, constvals = unzip2(constvar_to_val.items())
    in_binders = constvars + [tracer_to_var[id(t)] for t in tracers_in]
    out_vars = [tracer_to_var[id(t)] for t in tracers_out]
    prog = Prog(in_binders, instrs, out_vars)
    typecheck_prog(prog)
    return prog, constvals


def proto_to_instr(tracer_to_var: Dict[int, Var], proto: ProgEqnProto) -> ProgEqn:
    inputs = [tracer_to_var[id(t)] for t in proto.tracers_in]
    out_binders = [Var(aval) for aval in proto.avals_out]
    for t_ref, var in zip(proto.tracer_refs_out, out_binders):
        if t_ref() is not None:
            tracer_to_var[id(t_ref())] = var
    return ProgEqn(proto.prim, inputs, proto.params, out_binders)


def tracer_parents(t: PartialEvalTracerArray) -> List[PartialEvalTracerArray]:
    return t.proto.tracers_in if isinstance(t.proto, ProgEqnProto) else []


def toposort(out_nodes: List[Any], parents: Callable[[Any], List[Any]]):
    if not out_nodes:
        return []
    out_nodes = remove_duplicates(out_nodes)

    child_counts = {}
    stack = list(out_nodes)
    while stack:
        node = stack.pop()
        if id(node) in child_counts:
            child_counts[id(node)] += 1
        else:
            child_counts[id(node)] = 1
            stack.extend(parents(node))
    for node in out_nodes:
        child_counts[id(node)] -= 1

    sorted_nodes = []
    childless_nodes = [node for node in out_nodes if not child_counts[id(node)]]
    while childless_nodes:
        node = childless_nodes.pop()
        sorted_nodes.append(node)
        for parent in parents(node):
            if child_counts[id(parent)] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[id(parent)] -= 1

    sorted_nodes = sorted_nodes[::-1]
    check_toposort(sorted_nodes, parents)
    return sorted_nodes


def remove_duplicates(lst):
    seen = set()
    return [x for x in lst if id(x) not in seen and not seen.add(id(x))]


def check_toposort(nodes: List[Any], parents: Callable[[Any], List[Any]]):
    seen = set()
    for node in nodes:
        assert all(id(parent) in seen for parent in parents(node))
        seen.add(id(node))


def vjp_flat(f, *primals_in):
    pvals_in = [PartialVal.known(x) for x in primals_in] + [
        PartialVal.unknown(ArrayShape.like(TracerArray.get_aval(x))) for x in primals_in
    ]
    primal_pvals_in, tangent_pvals_in = split_half(pvals_in)

    def f_jvp(*primals_tangents_in):
        primals_out, tangents_out = jvp(f, *split_half(primals_tangents_in))
        return [*primals_out, *tangents_out]

    prog, pvals_out, consts = partial_eval_flat(f_jvp, pvals_in)  # linearize
    primal_pvals, _ = split_half(pvals_out)
    assert all(pval.is_known for pval in primal_pvals)
    primals_out = [pval.const for pval in primal_pvals]
    transpose_inputs = consts + [UndefPrimal(p.aval) for p in tangent_pvals_in]
    f_vjp = lambda *cts: eval_prog_transposed(prog, transpose_inputs, cts)
    return primals_out, f_vjp


def vjp(f, *primals_in):
    primals_in_flat, in_tree = tree_flatten(primals_in)
    f, out_tree = flatten_fun(f, in_tree)
    primals_out_flat, f_vjp_flat = vjp_flat(f, *primals_in_flat)
    primals_out = tree_unflatten(out_tree(), primals_out_flat)

    def f_vjp(*cotangents_out):
        cotangents_out_flat, _ = tree_flatten(cotangents_out)
        cotangents_in_flat = f_vjp_flat(*cotangents_out_flat)
        return tree_unflatten(in_tree, cotangents_in_flat)

    return primals_out, f_vjp


class UndefPrimal(NamedTuple):
    aval: ArrayShape


# NB: the analogous function in JAX is called 'backward_pass'
def eval_prog_transposed(
    prog: Prog, args: List[Any], cotangents: List[Any]
) -> List[Any]:
    primal_env: Dict[Var, Any] = {}
    ct_env: Dict[Var, Any] = {}

    def read_primal(x: Atom) -> Any:
        return primal_env.get(x, UndefPrimal(x.aval)) if type(x) is Var else x.val

    def write_primal(v: Var, val: Any) -> None:
        if type(val) is not UndefPrimal:
            primal_env[v] = val

    def read_cotangent(v: Var) -> Any:
        return ct_env.pop(v, Array.zeros(v.aval.shape, v.aval.dtype))

    def write_cotangent(x: Atom, val: Any):
        if type(x) is Var and val is not None:
            ct_env[x] = ct_env[x] + val if x in ct_env else val

    list_map(write_primal, prog.in_binders, args)
    list_map(write_cotangent, prog.outs, cotangents)
    # print(len(prog.instrs))
    # for i, instr in enumerate(prog.instrs[::-1]):
    # print(i, instr)
    for instr in prog.instrs[::-1]:
        primals_in = list_map(read_primal, instr.inputs)
        cts_in = list_map(read_cotangent, instr.out_binders)
        cts_out = instr.op.T(cts_in, *primals_in, **instr.params)
        list_map(write_cotangent, instr.inputs, cts_out)
    ret = [
        read_cotangent(v)
        for v, x in zip(prog.in_binders, args)
        if type(x) is UndefPrimal
    ]
    return ret


def grad(f):
    def gradfun(x, *xs):
        y, f_vjp = vjp(f, x, *xs)
        if np.shape(y) != ():
            raise TypeError
        # out = f_vjp(Array.ones(np.shape(y)))
        out = f_vjp(Array.ones(shape=()))
        return y, out

    return gradfun


def jit(f):
    hashable_prog = None
    hashable_consts = None

    def get_jit_fn():
        nonlocal hashable_consts, hashable_prog
        if hashable_prog is None and hashable_consts is None:
            print("Run with an input first to get jit_fn")
            return None
        return RT.backend.callable(hashable_prog, hashable_consts)

    def f_jitted(*args):
        avals_in = [ArrayShape.like(TracerArray.get_aval(x)) for x in args]
        prog, consts, out_tree = make_prog(f, *avals_in)
        nonlocal hashable_consts, hashable_prog
        hashable_consts = tuple(map(IDHashable, consts))
        hashable_prog = IDHashable(prog)
        outs = base_ops.Jit.do(
            *args, hashable_prog=hashable_prog, hashable_consts=hashable_consts
        )
        return tree_unflatten(out_tree, outs)

    f_jitted.get_jit_fn = get_jit_fn
    return f_jitted
