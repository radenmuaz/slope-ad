from dataclasses import dataclass
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
    Final,
)
import weakref
import types
from contextlib import contextmanager
import itertools
import weakref
from collections import defaultdict
from enum import Enum, auto
import operator as operator_py
import string
import numpy as np
import math
import inspect
from functools import partial

import slope as sp
import importlib

max_ = max
sum_ = sum
slice_ = slice
zip_ = zip
map_ = map


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


class Hable:
    val: Any

    def __init__(self, val):
        self.val = val

    def __hash__(self) -> int:
        # return hash(id(self.val))
        return hash((self.val,))

    def __eq__(self, other):
        if isinstance(other, Hable):
            return self.data == other.data
            # return id(self.val) == id(other.val)
        return False


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
        self.rt_ref = None
        self.args_fixer = lambda *args, **params: (args, params)

    def set_rt(self, rt):
        self.rt_ref = weakref.ref(rt)

    @property
    def rt(self):
        return self.rt_ref()

    def __repr__(self) -> str:
        return f"Op <{self.name}>"

    def pack_args(self, args, params):
        sig = inspect.signature(self.eval)
        args_strs = [
            k
            for k, v in sig.parameters.items()
            if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and k != "self"
        ]
        params_strs = [
            k
            for k, v in sig.parameters.items()
            if v.kind == inspect.Parameter.KEYWORD_ONLY and k != "self"
        ]

        if len(args) > len(args_strs) and len(args) != 0:  # and len(args_strs) != 0:
            args_ = args
            args, rest = args[: len(args_strs)], args[len(args_strs) :]
            new_params = {}
            for i, rest_arg in enumerate(rest):
                k = params_strs[i]
                assert k not in params.keys()

                new_params[k] = rest_arg
            params = {**new_params, **params}
        elif len(args) <= len(args_strs):
            args = list(args)
            for i, k in enumerate(args_strs):
                if k in params.keys():
                    args.insert(i, params[k])
                    del params[k]
            assert len(args) == len(args_strs)
        return args, params

    def impl(self, *args, **params):
        args, params = self.pack_args(args, params)
        args, params = self.args_fixer(*args, **params)
        return self.rt.backend.run_impl(self, *args, **params)

    def __call__(self, *args, **params):
        args, params = self.pack_args(args, params)
        args, params = self.args_fixer(*args, **params)
        return self.rt.bind1(self, *args, **params)

    def args_fixer(self, *args, **params):
        raise NotImplementedError

    def eval(self, *args, **params):
        raise NotImplementedError

    def partial_eval(self, trace, tracers, **params):
        tracers_in = [trace.instantiate_const(t) for t in tracers]
        avals_in = [t.aval for t in tracers_in]
        avals_out = self.shape_eval(*avals_in, **params)
        tracers_out = [
            PartialEvalTracerArray(
                self.rt, trace, self.rt.make_unknown_pval(aval), None
            )
            for aval in avals_out
        ]
        instr = InstrProto(
            self, tracers_in, params, avals_out, list_map(weakref.ref, tracers_out)
        )
        for t in tracers_out:
            t.proto = instr
        return tracers_out

    def partial_eval_instr(
        self,
        instr,
        read,
        write,
        env,
        residuals,
        instrs1,
        instrs2,
    ):
        in_unknowns = list_map(partial(read, env), instr.inputs)
        if any(in_unknowns):

            def new_res(x: Atom) -> Atom:
                if type(x) is Var:
                    residuals.add(x)
                return x

            inputs = [
                v if unk else new_res(v) for unk, v in zip(in_unknowns, instr.inputs)
            ]
            instrs2 += [Instr(instr.op, inputs, instr.params, instr.out_binders)]
            list_map(partial(write, env, True), instr.out_binders)
        else:
            instrs1 += [instr]
            list_map(partial(write, env, False), instr.out_binders)

    def jvp(self, *args, **params):
        raise NotImplementedError

    def T(self, *args, **params):
        raise NotImplementedError

    def vmap(self, *args, **params):
        raise NotImplementedError

    def shape_eval(self, *args, **params):
        raise NotImplementedError

    def set_partial_eval_instr(self, fn):
        self.partial_eval_instr = types.MethodType(fn, self)

    def set_args_fixer(self, fn):
        self.args_fixer = types.MethodType(fn, self)

    def set_partial_eval(self, fn):
        self.partial_eval = types.MethodType(fn, self)

    def set_eval(self, fn):
        self.eval = types.MethodType(fn, self)

    def set_jvp(self, fn):
        self.jvp = types.MethodType(fn, self)

    def set_vmap(self, fn):
        self.vmap = types.MethodType(fn, self)

    def set_T(self, fn):
        self.T = types.MethodType(fn, self)

    def set_shape_eval(self, fn):
        self.shape_eval = types.MethodType(fn, self)

    @classmethod
    def unary(cls, name):
        op = cls(name, OpType.Unary)

        @op.set_vmap
        def f(self, x, *, axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            return [self(x, **params)], [x_bdim]

        @op.set_shape_eval
        def f(self, x, **params):
            return [VoidArray(x.shape, x.dtype)]

        @op.set_jvp
        def f(self, primals, tangents, **params):
            (x,), (x_dot,) = primals, tangents
            return [self(x, **params)], [self(x_dot, **params)]

        return op

    @classmethod
    def binary(cls, name):
        op = cls(name, OpType.Binary)

        @op.set_args_fixer
        def f(self, x, y, **params):
            if type(x) is UndefPrimal and type(y) is UndefPrimal:
                assert x.aval.shape == y.aval.shape
                return (x, y), params
            elif type(x) is UndefPrimal:
                assert x.aval.shape == y.shape
                return (x, y), params
            elif type(y) is UndefPrimal:
                assert y.aval.shape == x.shape
                return (x, y), params

            if type(x) in [bool, int, float]:
                x = self.rt.array(x, dtype=y.dtype)
            elif type(y) in [bool, int, float]:
                y = self.rt.array(y, dtype=x.dtype)

            if type(x) is Array and isinstance(y, TracerArray):
                x = y._trace.pure(x)
            elif type(y) is Array and isinstance(x, TracerArray):
                y = x._trace.pure(y)

            if getattr(x, "shape", None) == getattr(y, "shape", None):
                return (x, y), params
            if x.ndim == 0:
                shape_ret = y.shape
                bx = tuple(range(y.ndim))
                x = x.broadcast(shape=shape_ret, axes=bx)
            elif y.ndim == 0:
                shape_ret = x.shape
                by = tuple(range(x.ndim))
                y = y.broadcast(shape=shape_ret, axes=by)
            else:
                bx = tuple(range((max_(x.ndim, y.ndim) - x.ndim)))
                by = tuple(range((max_(x.ndim, y.ndim) - y.ndim)))
                bx = bx if len(bx) > 0 else None
                by = by if len(by) > 0 else None
                shape_ret = tuple(max_(sx, sy) for sx, sy in list_zip(x.shape, y.shape))
                x = x.broadcast(shape=shape_ret, axes=bx)
                y = y.broadcast(shape=shape_ret, axes=by)

            return (x, y), params

        @op.set_vmap
        def f(self, axis_size, vals_in, dims_in, **params):
            (x, y), (x_bdim, y_bdim) = vals_in, dims_in
            if x_bdim != y_bdim:
                if x_bdim is None:
                    x = BatchTrace.move_batch_axis(axis_size, x_bdim, y_bdim, x)
                    x_bdim = y_bdim
                else:
                    y = BatchTrace.move_batch_axis(axis_size, y_bdim, x_bdim, y)
            return [self(x, y, **params)], [x_bdim]

        @op.set_shape_eval
        def f(self, x: VoidArray, y: VoidArray, **params) -> List[VoidArray]:
            if not type(x) in (Array, VoidArray) or not type(x) in (Array, VoidArray):
                raise TypeError
            if VoidArray.like(x) != VoidArray.like(y):
                raise TypeError(f"{x} != {y}")
            return [VoidArray(x.shape, x.dtype)]

        @op.set_jvp
        def f(self, primals, tangents, **params):
            (x,), (x_dot,) = primals, tangents
            return [self(x, **params)], [self(x_dot, **params)]

        return op

    @classmethod
    def reduce(cls, name):
        op = cls(name, OpType.Reduce)

        @op.set_args_fixer
        def f(self, x, axes=None, keepdims=False):
            if axes is None:
                axes = tuple(range(x.ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            axes = tuple(a if a >= 0 else a + len(x.shape) for a in axes)
            return (x,), dict(axes=axes, keepdims=keepdims)

        @op.set_vmap
        def f(self, axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            axes = list(params["axes"])
            axes = tuple(a + (x_bdim <= a) for a in axes)
            out_bdim = x_bdim - sum(a < x_bdim for a in axes)
            params["axes"] = tuple(axes)
            return [cls.do(x, **params)], [out_bdim]

        @op.set_shape_eval
        def f(self, x: VoidArray, axes=None, keepdims=False) -> List[VoidArray]:
            axes = [a + len(x.shape) if a < 0 else a for a in axes]
            axes_ = set(axes)
            if keepdims:
                new_shape = [d if i not in axes_ else 1 for i, d in enumerate(x.shape)]
            else:
                new_shape = [d for i, d in enumerate(x.shape) if i not in axes_]
            return [VoidArray(tuple(new_shape), x.dtype)]

        return op

    @classmethod
    def shape(cls, name):
        op = cls(name, OpType.Shape)
        return op

    @classmethod
    def load(cls, name):
        op = cls(name, OpType.Load)
        return op


# ========================
# Ops
# ========================


class OpsDir:
    def register(self, op):
        setattr(self, op.name, op)

    def alias(self, op, name):
        assert op.name in vars(self)
        setattr(self, name, getattr(self, op.name))


class ProcsDir:
    def register(self, fn):
        setattr(self, fn.__name__, fn)
        return fn


@dataclass
class Opset:
    ops_dir: OpsDir
    procs_dir: ProcsDir
    backends: dict


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
        # assert len(arg) == n
        try:
            assert len(arg) == n
        except:
            breakpoint()
            raise
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


class VoidArray:
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
        raise Exception("VoidArray can't be unambiguously converted to bool")

    @staticmethod
    def _nonzero(tracer):
        raise Exception("VoidArray can't be unambiguously converted to bool")

    def str_short(self):
        return f'{str(self.dtype)}[{",".join(str(d) for d in self.shape)}]'

    def __hash__(self):
        return hash((self.shape, self.dtype))

    def __eq__(self, other):
        if type(self) != type(other):
            return other == self
        return tuple(self.shape) == tuple(other.shape) and self.dtype == other.dtype

    def __repr__(self):
        return f"VoidArray(shape={self.shape}, dtype={self.dtype})"


class Var:
    val = None
    aval: VoidArray

    def __init__(self, aval):
        self.aval = aval


class Lit:
    val: Any
    aval: VoidArray

    def __init__(self, val):
        self.aval = aval = VoidArray.like(self.get_aval(val))
        self.val = np.array(val, aval.dtype)


Atom = Union[Var, Lit]


class Instr(NamedTuple):
    op: Op
    inputs: List[Atom]
    params: Dict[str, Any]
    out_binders: List[Atom]


class Prog(NamedTuple):
    in_binders: Any
    instrs: Tuple[Instr]
    outs: Any

    def __hash__(self):
        return hash(repr(self))

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

    def pp_instr(self, names: DefaultDict[Var, str], instr: Instr) -> PPrint:
        lhs = PPrint.pp(" ".join(self.var_str(names, v) for v in instr.out_binders))
        rhs = (
            PPrint.pp(repr(instr.op.name))
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
    in_types: Tuple[VoidArray]
    out_types: Tuple[VoidArray]

    def __repr__(self):
        in_types = ", ".join(aval.str_short() for aval in self.in_types)
        out_types = ", ".join(aval.str_short() for aval in self.out_types)
        return f"({in_types}) -> ({out_types})"


class MainTrace(NamedTuple):
    rt: "Runtime"
    level: int
    trace_type: Type["Trace"]
    global_data: Optional[Any]


class Trace:
    main: MainTrace

    def __init__(self, rt, main: MainTrace) -> None:
        self.rt_ref = weakref.ref(rt)
        self.main = main

    @property
    def rt(self):
        return self.rt_ref()

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

    def __init__(self, rt):
        self.rt_ref = weakref.ref(rt)

    @property
    def rt(self):
        return self.rt_ref()

    def is_int(self) -> bool:
        return self.dtype in (self.int8, self.uint8, self.int32, self.int64)

    def is_float(self) -> bool:
        return self.dtype in (self.float16, self.float32)

    def is_unsigned(self) -> bool:
        return self.dtype is self.uint8

    def __getattr__(self, attr):
        raise NotImplementedError

    def __getitem__(self, idx):
        if None in idx:
            self.broadcast(self.shape, idx)

    def __setitem__(self, idx, item):
        raise NotImplementedError

    def str_short(self):
        return f'{str(self.dtype)}[{",".join(str(d) for d in self.shape)}]'

    __neg__ = lambda self: self.neg()
    __add__ = lambda self, other: self.add(other)
    __radd__ = lambda self, other: self.add(other)
    __sub__ = lambda self, other: self.sub(other)
    __rsub__ = lambda self, other: self.sub.func(other, self)
    __mul__ = lambda self, other: self.mul(other)
    __rmul__ = lambda self, other: self.mul(other)
    __div__ = lambda self, other: self.div(other)
    __rdiv__ = lambda self, other: self.div.func(other, self)
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
    __array_priority__ = 1000

    def __init__(self, rt, val: ArrayBuffer):
        super().__init__(rt)
        assert isinstance(val, ArrayBuffer)
        self.buf = val

    def __hash__(self):
        return id(self.val)

    val = property(lambda self: self.buf.val)
    dtype = property(lambda self: self.buf.val.dtype)
    shape = property(lambda self: self.buf.val.shape)
    ndim = property(lambda self: self.buf.val.ndim)

    def __getattr__(self, attr):
        if attr in self.__dict__.keys():
            return self.__dict__[attr]
        if attr in vars(self.rt.ops).keys():
            op = getattr(self.rt.ops, attr)
            return partial(op.impl, self)
        elif attr in vars(self.rt.procs).keys():
            proc = getattr(self.rt.procs, attr)
            assert not isinstance(
                proc, classmethod
            ), f"use rt.{attr} instead of Array.{attr}"
            return partial(proc, self)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {attr}")

    def __repr__(self):
        return f"{self.__class__.__name__}: {repr(self.val)[6:-1] if self.val.ndim > 0 else self.val}"

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
    __array_priority__ = 2000

    _trace: "Trace"

    # @property
    # def aval(self):
    #     return self.get_aval(self.val)

    aval = property(lambda self: self.get_aval(self.val))
    dtype = property(lambda self: self.val.dtype)
    shape = property(lambda self: self.val.shape)
    ndim = property(lambda self: self.val.ndim)

    def full_lower(self):
        return self.val

    # def __repr__(self):
    #     return f"{self.__class__.__name__}: {repr(self.aval)}"

    def __str__(self):
        return repr(self)

    def full_lower(self):
        return self

    def __getattr__(self, attr):
        if attr in vars(self.rt.ops).keys():
            op = getattr(self.rt.ops, attr)
            return partial(op, self)
        elif attr in vars(self.rt.procs).keys():
            proc = getattr(self.rt.procs, attr)
            assert not isinstance(
                proc, classmethod
            ), f"Access this proc by Array.{attr}"
            return partial(proc, self)
        try:
            return getattr(self.aval, attr)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {attr}")

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return f"{self.__class__.__name__}: {repr(self.aval)[6:-1] if self.aval.ndim > 0 else self.aval}"


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

    # def __repr__(self):
    # return f"tree {self.node_type.name})\n\tmetadata:{self.node_metadata}\n\tchildren:{self.child_treedefs}"


class Leaf:
    pass


leaf = Leaf()

jit_op = Op("jit_op")
jit_op.pack_args = lambda args, params: (args, params)


@jit_op.set_eval
def f(self, *args, prog, num_consts):
    hable_prog = Hable(prog)
    consts, args = args[:num_consts], args[num_consts:]
    hable_consts = tuple(map(Hable, consts))
    jit_fn = self.rt.backend.callable(hable_prog, hable_consts)
    return jit_fn(*args, *consts)


@jit_op.set_jvp
def f(self, primals, tangents, *, prog, num_consts):
    new_prog, new_consts = self.rt.jvp_prog(prog)
    outs = self.rt.bind(
        self,
        *new_consts,
        *primals,
        *tangents,
        prog=new_prog,
        num_consts=len(new_consts),
    )
    n = len(outs) // 2
    primals_out, tangents_out = outs[:n], outs[n:]
    return primals_out, tangents_out


@jit_op.set_shape_eval
def f(self, *in_types, prog, num_consts):
    prog_type = self.rt.typecheck_prog(prog)
    if not all(t1 == t2 for t1, t2 in zip(prog_type.in_types, in_types)):
        raise TypeError
    return prog_type.out_types


@jit_op.set_T
def f(self, cts, *invals, prog, num_consts):
    undef_primals = [type(x) is UndefPrimal for x in invals]
    transposed_prog, new_consts = self.rt.transpose_prog(prog, tuple(undef_primals))
    residuals, _ = partition_list(undef_primals, invals)
    outs = self.rt.bind(
        self,
        *new_consts,
        *residuals,
        *cts,
        prog=transposed_prog,
        num_consts=len(new_consts),
    )
    outs = iter(outs)
    return [next(outs) if undef else None for undef in undef_primals]


@jit_op.set_partial_eval
def f(self, trace, tracers, *, prog, num_consts):
    in_unknowns = [not t.pval.is_known for t in tracers]
    prog1, prog2, out_unknowns, num_res = self.rt.partial_eval_prog(prog, in_unknowns)
    known_tracers, unknown_tracers = partition_list(in_unknowns, tracers)
    known_vals = [t.pval.const for t in known_tracers]
    outs1_res = self.rt.bind(jit_op, *known_vals, prog=prog1, num_consts=0)
    outs1, res = split_list(outs1_res, len(prog1.outs) - num_res)
    res_tracers = [trace.instantiate_const(self.rt.full_raise(trace, x)) for x in res]
    outs2 = [
        PartialEvalTracerArray(self.rt, trace, self.rt.make_unknown_pval(v.aval), None)
        for v in prog2.outs
    ]
    instr = InstrProto(
        self,
        res_tracers + unknown_tracers,
        dict(prog=prog2, num_consts=0),
        [v.aval for v in prog2.outs],
        list_map(weakref.ref, outs2),
    )
    for t in outs2:
        t.proto = instr

    return merge_lists(out_unknowns, outs1, outs2)


@jit_op.set_partial_eval_instr
def f(
    self, instr: Instr, read, write, env, residuals, instrs1, instrs2
) -> Tuple[Instr, Instr, List[bool], List[Var]]:
    in_unknowns = list_map(partial(read, env), instr.inputs)
    prog = instr.params["prog"]
    prog1, prog2, out_unknowns, num_res = self.rt.partial_eval_prog(prog, in_unknowns)
    ins1, ins2 = partition_list(in_unknowns, instr.inputs)
    out_binders1, out_binders2 = partition_list(out_unknowns, instr.out_binders)
    res = [Var(v.aval) for v in prog2.in_binders[:num_res]]
    instr1 = Instr(self, ins1, dict(prog=prog1, num_consts=0), out_binders1 + res)
    instr2 = Instr(self, res + ins2, dict(prog=prog2, num_consts=0), out_binders2)
    instrs1.append(instr1)
    instrs2.append(instr2)
    residuals.update(res)
    list_map(partial(write, env), out_unknowns, instr.out_binders)


class Runtime:
    def __init__(
        self,
        opset,
        default_backend="numpy",
    ):
        self.trace_stack: List[MainTrace] = []
        self.dynamic_trace: Optional[MainTrace] = None
        self.trace_stack += [MainTrace(self, 0, EvalTrace, None)]
        self.node_types = dict()
        self.register_pytree_node(tuple, lambda t: (None, t), lambda _, xs: tuple(xs))
        self.register_pytree_node(list, lambda l: (None, l), lambda _, xs: list(xs))
        self.register_pytree_node(
            dict,
            lambda d: list_map(tuple, unzip2(sorted(d.items()))),
            lambda keys, vals: dict(list_zip(keys, vals)),
        )
        self.register_pytree_node(
            UndefPrimal, lambda u: (u.aval, ()), lambda aval, _: UndefPrimal(aval)
        )

        self.opset = opset
        self.ops.register(jit_op)
        for op_name in vars(self.ops):
            getattr(self.ops, op_name).set_rt(self)
        for _, backend in self.backends.items():
            backend.set_rt(self)
        self.backend = self.backends[default_backend]

    def make_known_pval(self, val: Any):
        return PartialVal(self.get_aval(val), val)

    def make_unknown_pval(self, aval: VoidArray):
        return PartialVal(aval, None)

    def get_aval(self, x):
        if isinstance(x, TracerArray):
            return x.aval
        elif type(x) in TracerArray.TYPES:
            return self.array(np.asarray(x))
        elif isinstance(x, Array):
            return x
        elif isinstance(x, VoidArray):
            breakpoint()
            return x
        else:
            raise TypeError(x)

    def array(
        self,
        val: Union[list, tuple, np.ndarray, ArrayBuffer] = None,
        dtype: Optional[Any] = BaseArray.default_dtype,
    ):
        return (
            Array(self, val)
            if isinstance(val, ArrayBuffer)
            else self.backend.run_impl(self.ops.constant, val=val, dtype=dtype)
        )

    @property
    def ops(self):
        return self.opset.ops_dir

    @property
    def procs(self):
        return self.opset.procs_dir

    @property
    def backends(self):
        return self.opset.backends

    def tree_flatten(self, x: Any) -> Any:
        def _tree_flatten(x_: Any) -> Tuple[Iterable, Union[PyTreeDef, Leaf]]:
            node_type = self.node_types.get(type(x_))

            if node_type:
                node_metadata, children = node_type.to_iterable(x_)
                children_flat, child_trees = unzip2(list_map(_tree_flatten, children))
                flattened = itertools.chain.from_iterable(children_flat)
                return flattened, PyTreeDef(
                    node_type, node_metadata, tuple(child_trees)
                )
            else:
                return [x_], leaf

        children_iter, treedef = _tree_flatten(x)
        return list(children_iter), treedef

    def tree_unflatten(self, treedef: PyTreeDef, xs: List[Any]) -> Any:
        def _tree_unflatten(treedef_: PyTreeDef, xs_: Iterator) -> Any:
            if treedef_ is leaf:
                return next(xs_)
            else:
                children = (_tree_unflatten(t, xs_) for t in treedef_.child_treedefs)
                return treedef_.node_type.from_iterable(
                    treedef_.node_metadata, children
                )

        return _tree_unflatten(treedef, iter(xs))

    def flatten_fun(self, f, in_tree):
        store = Store()

        def flat_fun(*args_flat):
            pytree_args = self.tree_unflatten(in_tree, args_flat)
            out = f(*pytree_args)
            out_flat, out_tree = self.tree_flatten(out)
            store.set_value(out_tree)
            return out_flat

        return flat_fun, store

    def register_pytree_node(
        self, ty: Type, to_iter: Callable, from_iter: Callable
    ) -> None:
        self.node_types[ty] = NodeType(str(ty), to_iter, from_iter)

    def tree_map(self, f: Callable[..., Any], tree: Any) -> Any:
        leaves, treedef = self.tree_flatten(tree)
        return self.tree_unflatten(treedef, [f(leaf) for leaf in leaves])

    @contextmanager
    def new_main(self, trace_type: Type["Trace"], global_data=None):
        level = len(self.trace_stack)
        main = MainTrace(self, level, trace_type, global_data)
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
        top_main = max_(
            (x._trace.main for x in xs if isinstance(x, TracerArray)),
            default=self.trace_stack[0],
            key=operator_py.attrgetter("level"),
        )
        if self.dynamic_trace and self.dynamic_trace.level > top_main.level:
            top_main = self.dynamic_trace
        return top_main.trace_type(self, top_main)

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

    def typecheck_prog(self, prog: Prog) -> ProgType:
        env: Set[Var] = set()

        for v in prog.in_binders:
            if v in env:
                raise TypeError
            env.add(v)

        for instr in prog.instrs:
            in_types = [self.typecheck_atom(env, x) for x in instr.inputs]
            out_types = instr.op.shape_eval(*in_types, **instr.params)
            for out_binder, out_type in list_zip(instr.out_binders, out_types):
                if not out_type == out_binder.aval:
                    raise TypeError
            for out_binder in instr.out_binders:
                if out_binder in env:
                    raise TypeError
                env.add(out_binder)

        in_types = [v.aval for v in prog.in_binders]
        out_types = [self.typecheck_atom(env, x) for x in prog.outs]
        return ProgType(tuple(in_types), tuple(out_types))

    def typecheck_atom(self, env: Set[Var], x: Atom) -> VoidArray:
        if isinstance(x, Var):
            if x not in env:
                raise TypeError("unbound variable")
            return x.aval
        elif isinstance(x, Lit):
            return self.get_aval(x.val)
        else:
            assert False

    def eval_prog(self, prog: Prog, args: List[Any]) -> List[Any]:
        env: Dict[Var, Any] = {}

        def read(x: Atom) -> Any:
            return env[x] if type(x) is Var else x.val

        def write(v: Var, val: Any) -> None:
            assert v not in env  # single-assignment
            env[v] = val

        list_map(write, prog.in_binders, args)
        for instr in prog.instrs:
            in_vals = list_map(read, instr.inputs)
            outs = self.bind(instr.op, *in_vals, **instr.params)
            list_map(write, instr.out_binders, outs)
        return list_map(read, prog.outs)

    def prog_as_fun(self, prog: Prog):
        return lambda *args: self.eval_prog(prog, args)

    def vmap_flat(self, f, in_axes, *args):
        axis_set = {x.shape[ax] for x, ax in list_zip(args, in_axes) if ax is not None}
        assert len(axis_set) == 1
        (axis_size,) = axis_set
        with self.new_main(BatchTrace, axis_size) as main:
            trace = BatchTrace(main)
            tracers_in = [
                BatchTracerArray(self, trace, x, ax) if ax is not None else x
                for x, ax in list_zip(args, in_axes)
            ]
            outs = f(*tracers_in)
            tracers_out = [self.full_raise(trace, out) for out in outs]
            vals_out, bdims_out = unzip2((t.val, t.batch_dim) for t in tracers_out)
        outs_transposed = [
            BatchTrace.move_batch_axis(axis_size, bdim, 0, val_out)
            for val_out, bdim in list_zip(vals_out, bdims_out)
        ]
        return outs_transposed

    def vmap(self, f, in_axes):
        def batched_f(*args):
            args_flat, in_tree = self.tree_flatten(args)
            in_axes_flat, in_tree2 = self.tree_flatten(in_axes)
            if in_tree != in_tree2:
                raise TypeError(f"{in_tree}\n!=\n{in_tree2}")
            f_flat, out_tree = self.flatten_fun(f, in_tree)
            outs_flat = self.vmap_flat(f_flat, in_axes_flat, *args_flat)
            return self.tree_unflatten(out_tree(), outs_flat)

        return batched_f

    def jvp_flat(self, f, primals, tangents):
        with self.new_main(JVPTrace) as main:
            trace = JVPTrace(self, main)
            tracers_in = [
                JVPTracerArray(self, trace, x, t)
                for x, t in list_zip(primals, tangents)
            ]
            outs = f(*tracers_in)
            tracers_out = [self.full_raise(trace, out) for out in outs]
            primals_out, tangents_out = unzip2(
                (t.primal, t.tangent) for t in tracers_out
            )
        return primals_out, tangents_out

    def jvp(self, f, primals, tangents):
        primals_flat, in_tree = self.tree_flatten(primals)
        tangents_flat, in_tree2 = self.tree_flatten(tangents)
        if in_tree != in_tree2:
            raise TypeError
        f, out_tree = self.flatten_fun(f, in_tree)
        primals_out_flat, tangents_out_flat = self.jvp_flat(
            f, primals_flat, tangents_flat
        )
        primals_out = self.tree_unflatten(out_tree(), primals_out_flat)
        tangents_out = self.tree_unflatten(out_tree(), tangents_out_flat)
        return primals_out, tangents_out

    def jacfwd(self, f, x):
        pushfwd = lambda v: self.jvp(f, (x,), (v,))[1]
        vecs_in = self.procs.eye(math.prod(x.shape)).reshape(x.shape * 2)
        return self.vmap(pushfwd, (0,))(vecs_in)

    def make_prog(
        self,
        f: Callable,
        *avals_in: VoidArray,
    ) -> Tuple[Prog, List[Any], PyTreeDef]:
        oavals_in = avals_in
        avals_in, in_tree = self.tree_flatten(avals_in)
        f, out_tree = self.flatten_fun(f, in_tree)

        builder = ProgBuilder(self)
        with self.new_main(ProgTrace, builder) as main:
            with self.new_dynamic(main):
                trace = ProgTrace(self, main)
                tracers_in = [trace.new_arg(aval) for aval in avals_in]
                outs = f(*tracers_in)
                tracers_out = [self.full_raise(trace, out) for out in outs]
                prog, consts = builder.build(tracers_in, tracers_out)
        return prog, consts, out_tree()

    def jvp_prog(self, prog: Prog) -> Tuple[Prog, List[Any]]:
        def jvp_traceable(*primals_and_tangents):
            n = len(primals_and_tangents) // 2
            primals, tangents = primals_and_tangents[:n], primals_and_tangents[n:]
            return self.jvp(self.prog_as_fun(prog), primals, tangents)

        in_avals = self.tree_map(lambda v: v.aval, prog.in_binders)
        new_prog, new_consts, _ = self.make_prog(jvp_traceable, *in_avals, *in_avals)
        return new_prog, new_consts

    def partial_eval_flat(
        self, f: Callable, pvals_in: List["PartialVal"]
    ) -> Tuple[Prog, List["PartialVal"], List[Any]]:
        with self.new_main(PartialEvalTrace) as main:
            trace = PartialEvalTrace(self, main)
            tracers_in = [trace.new_arg(pval) for pval in pvals_in]
            outs = f(*tracers_in)
            tracers_out = [self.full_raise(trace, out) for out in outs]
            pvals_out = [t.pval for t in tracers_out]
            unk_tracers_in = [t for t in tracers_in if t.pval.is_unknown]
            unk_tracers_out = [t for t in tracers_out if t.pval.is_unknown]
            prog, consts = self.tracers_to_prog(unk_tracers_in, unk_tracers_out)

        return prog, pvals_out, consts

    def partial_eval_prog(
        self,
        prog: Prog,
        in_unknowns: List[bool],
        instantiate: Optional[List[bool]] = None,
    ) -> Tuple[Prog, Prog, List[bool], int]:
        env: Dict[Var, bool] = {}
        residuals: Set[Var] = set()

        def read(env, x: Atom) -> bool:
            return type(x) is Var and env[x]

        def write(env, unk: bool, v: Var) -> None:
            env[v] = unk

        instrs1, instrs2 = [], []
        list_map(partial(write, env), in_unknowns, prog.in_binders)

        for instr in prog.instrs:
            instr.op.partial_eval_instr(
                instr, read, write, env, residuals, instrs1, instrs2
            )
        out_unknowns = list_map(partial(read, env), prog.outs)
        if instantiate is not None:
            for v, uk, inst in zip(prog.outs, out_unknowns, instantiate):
                if inst and not uk:
                    if type(v) is Var:
                        residuals.add(v)
            out_unknowns = list_map(operator_py.or_, out_unknowns, instantiate)

        residuals, num_res = list(residuals), len(residuals)
        assert all(type(v) is Var for v in residuals), residuals

        ins1, ins2 = partition_list(in_unknowns, prog.in_binders)
        outs1, outs2 = partition_list(out_unknowns, prog.outs)

        prog1 = Prog(ins1, instrs1, outs1 + residuals)
        prog2 = Prog(residuals + ins2, instrs2, outs2)
        self.typecheck_partial_eval_prog(prog, in_unknowns, out_unknowns, prog1, prog2)

        return prog1, prog2, out_unknowns, num_res

    def typecheck_partial_eval_prog(
        self, prog, in_unknowns, out_unknowns, prog1, prog2
    ):
        progty = self.typecheck_prog(prog)  # (a1,  a2) -> (b1, b2 )
        prog1ty = self.typecheck_prog(prog1)  #  a1       -> (b1, res)
        prog2ty = self.typecheck_prog(prog2)  # (res, a2) -> b2

        a1, a2 = partition_list(in_unknowns, progty.in_types)
        b1, b2 = partition_list(out_unknowns, progty.out_types)
        b1_, res = split_list(prog1ty.out_types, len(b1))
        res_, a2_ = split_list(prog2ty.in_types, len(res))
        b2_ = prog2ty.out_types

        a1 = tuple(a1)
        a2, a2_ = tuple(a2), tuple(a2_)
        b1, b1_ = tuple(b1), tuple(b1_)
        b2, b2_ = tuple(b2), tuple(b2_)
        res, res_ = tuple(res), tuple(res_)

        if prog1ty.in_types != a1:
            raise TypeError
        if prog2ty.out_types != b2:
            raise TypeError
        if b1 != b1_:
            raise TypeError
        if res != res_:
            raise TypeError
        if a2 != a2_:
            raise TypeError
        if b2 != b2_:
            raise TypeError

    def linearize_flat(self, f, *primals_in):
        pvals_in = [self.make_known_pval(x) for x in primals_in] + [
            self.make_unknown_pval(VoidArray.like(self.get_aval(x))) for x in primals_in
        ]

        def f_jvp(*primals_tangents_in):
            primals_out, tangents_out = self.jvp(f, *split_half(primals_tangents_in))
            return [*primals_out, *tangents_out]

        prog, pvals_out, consts = self.partial_eval_flat(f_jvp, pvals_in)
        primal_pvals, _ = split_half(pvals_out)
        assert all(pval.is_known for pval in primal_pvals)
        primals_out = [pval.const for pval in primal_pvals]
        f_lin = lambda *tangents: self.eval_prog(prog, [*consts, *tangents])
        return primals_out, f_lin

    def linearize(self, f, *primals_in):
        primals_in_flat, in_tree = self.tree_flatten(primals_in)
        f, out_tree = self.flatten_fun(f, in_tree)
        primals_out_flat, f_lin_flat = self.linearize_flat(f, *primals_in_flat)
        primals_out = self.tree_unflatten(out_tree(), primals_out_flat)

        def f_lin(*tangents_in):
            tangents_in_flat, in_tree2 = self.tree_flatten(tangents_in)
            if in_tree != in_tree2:
                raise TypeError
            tangents_out_flat = f_lin_flat(*tangents_in_flat)
            return self.tree_unflatten(out_tree(), tangents_out_flat)

        return primals_out, f_lin

    def tracers_to_prog(
        self,
        tracers_in: List["PartialEvalTracerArray"],
        tracers_out: List["PartialEvalTracerArray"],
    ):
        def tracer_parents(t: PartialEvalTracerArray) -> List[PartialEvalTracerArray]:
            return t.proto.tracers_in if isinstance(t.proto, InstrProto) else []

        def proto_to_instr(tracer_to_var: Dict[int, Var], proto: InstrProto) -> Instr:
            inputs = [tracer_to_var[id(t)] for t in proto.tracers_in]
            out_binders = [Var(aval) for aval in proto.avals_out]
            for t_ref, var in list_zip(proto.tracer_refs_out, out_binders):
                if t_ref() is not None:
                    tracer_to_var[id(t_ref())] = var
            return Instr(proto.prim, inputs, proto.params, out_binders)

        tracer_to_var: Dict[int, Var] = {
            id(t): Var(VoidArray.like(t.aval)) for t in tracers_in
        }
        constvar_to_val: Dict[int, Any] = {}
        constid_to_var: Dict[int, Var] = {}
        processed_instrs: Set[int] = set()
        instrs: List[Instr] = []
        for t in self.toposort(tracers_out, tracer_parents):
            if isinstance(t.proto, LambdaBindingProto):
                assert id(t) in set(list_map(id, tracers_in))
            elif isinstance(t.proto, ConstProto):
                val = t.proto.val
                var = constid_to_var.get(id(val))
                if var is None:
                    aval = VoidArray.like(self.get_aval(val))
                    var = constid_to_var[id(val)] = Var(aval)
                    constvar_to_val[var] = val
                tracer_to_var[id(t)] = var
            elif isinstance(t.proto, InstrProto):
                if id(t.proto) not in processed_instrs:
                    instrs.append(proto_to_instr(tracer_to_var, t.proto))
                    processed_instrs.add(id(t.proto))
            else:
                raise TypeError(t.proto)

        constvars, constvals = unzip2(constvar_to_val.items())
        in_binders = constvars + [tracer_to_var[id(t)] for t in tracers_in]
        out_vars = [tracer_to_var[id(t)] for t in tracers_out]
        prog = Prog(tuple(in_binders), tuple(instrs), tuple(out_vars))
        self.typecheck_prog(prog)
        return prog, constvals

    def toposort(self, out_nodes: List[Any], parents: Callable[[Any], List[Any]]):
        def check_toposort(nodes: List[Any], parents: Callable[[Any], List[Any]]):
            seen = set()
            for node in nodes:
                assert all(id(parent) in seen for parent in parents(node))
                seen.add(id(node))

        def remove_duplicates(lst):
            seen = set()
            return [x for x in lst if id(x) not in seen and not seen.add(id(x))]

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

    def vjp_flat(self, f, *primals_in):
        pvals_in = [self.make_known_pval(x) for x in primals_in] + [
            self.make_unknown_pval(VoidArray.like(self.get_aval(x))) for x in primals_in
        ]
        primal_pvals_in, tangent_pvals_in = split_half(pvals_in)

        def f_jvp(*primals_tangents_in):
            primals_out, tangents_out = self.jvp(f, *split_half(primals_tangents_in))
            return [*primals_out, *tangents_out]

        prog, pvals_out, consts = self.partial_eval_flat(f_jvp, pvals_in)  # linearize
        primal_pvals, _ = split_half(pvals_out)
        assert all(pval.is_known for pval in primal_pvals)
        primals_out = [pval.const for pval in primal_pvals]
        transpose_inputs = consts + [UndefPrimal(p.aval) for p in tangent_pvals_in]
        f_vjp = lambda *cts: self.eval_prog_transposed(prog, transpose_inputs, cts)
        return primals_out, f_vjp

    def vjp(self, f, *primals_in):
        primals_in_flat, in_tree = self.tree_flatten(primals_in)
        f, out_tree = self.flatten_fun(f, in_tree)
        primals_out_flat, f_vjp_flat = self.vjp_flat(f, *primals_in_flat)
        primals_out = self.tree_unflatten(out_tree(), primals_out_flat)

        def f_vjp(*cotangents_out):
            cotangents_out_flat, _ = self.tree_flatten(cotangents_out)
            cotangents_in_flat = f_vjp_flat(*cotangents_out_flat)

            return self.tree_unflatten(in_tree, cotangents_in_flat)

        return primals_out, f_vjp

    def eval_prog_transposed(
        self, prog: Prog, args: List[Any], cotangents: List[Any]
    ) -> List[Any]:
        primal_env: Dict[Var, Any] = {}
        ct_env: Dict[Var, Any] = {}

        def read_primal(x: Atom) -> Any:
            return primal_env.get(x, UndefPrimal(x.aval)) if type(x) is Var else x.val

        def write_primal(v: Var, val: Any) -> None:
            if type(val) is not UndefPrimal:
                primal_env[v] = val

        def read_cotangent(v: Var) -> Any:
            return ct_env.pop(v, self.procs.zeros(v.aval.shape, v.aval.dtype))

        def write_cotangent(x: Atom, val: Any):
            if type(x) is Var and val is not None:
                ct_env[x] = ct_env[x] + val if x in ct_env else val

        list_map(write_primal, prog.in_binders, args)
        list_map(write_cotangent, prog.outs, cotangents)
        # print(len(prog.instrs))
        # for i, instr in enumerate(prog.instrs[::-1]):
        #     print(i, instr)
        for instr in prog.instrs[::-1]:
            primals_in = list_map(read_primal, instr.inputs)
            cts_in = list_map(read_cotangent, instr.out_binders)
            inp, params = primals_in, instr.params
            inp, params = instr.op.pack_args(inp, params)
            inp, params = instr.op.args_fixer(*inp, **params)
            cts_out = instr.op.T(cts_in, *inp, **params)
            list_map(write_cotangent, instr.inputs, cts_out)

        ret = [
            read_cotangent(v)
            for v, x in list_zip(prog.in_binders, args)
            if type(x) is UndefPrimal
        ]

        return ret

    def transpose_prog(
        self, prog: Prog, undef_primals: tuple[bool, ...]
    ) -> tuple[Prog, list[Any]]:
        avals_in, avals_out = self.typecheck_prog(prog)
        traceable = partial(self.eval_prog_transposed, prog)
        args = [UndefPrimal(a) if u else a for a, u in zip(avals_in, undef_primals)]
        trans_prog, consts, _ = self.make_prog(traceable, tuple(args), tuple(avals_out))
        self.typecheck_prog(trans_prog)

        return trans_prog, consts

    def grad(self, f):
        def gradfun(x, *xs):
            y, f_vjp = self.vjp(f, x, *xs)
            if np.shape(y) != ():
                raise TypeError
            out = f_vjp(self.procs.ones(()))
            return y, out

        return gradfun

    def jit(self, f):
        hable_prog = None
        hable_consts = None
        in_tree = None
        out_tree = None

        def get_jit_fn():
            jit_fn = self.backend.jit_fns.get(hash((hable_prog, hable_consts)), None)
            if jit_fn is None:
                print(f"jit not run for {f.__name__} yet")
            return jit_fn

        def f_jitted(*args):
            nonlocal hable_consts, hable_prog, in_tree, out_tree
            if hable_prog is None:
                avals_in = self.tree_map(
                    lambda x: VoidArray.like(self.get_aval(x)), args
                )
                prog, consts, out_tree = self.make_prog(f, *avals_in)

                hable_consts = tuple(list_map(Hable, consts))
                hable_prog = Hable(prog)
            else:
                consts = tuple(c.val for c in hable_consts)
            args, in_tree = self.tree_flatten(args)
            outs = self.bind(
                jit_op,
                *consts,
                *args,
                prog=hable_prog.val,
                num_consts=len(hable_consts),
            )
            return self.tree_unflatten(out_tree, outs)

        f_jitted.get_jit_fn = get_jit_fn
        return f_jitted

    def jit_partial_eval(self, trace, tracers, *, prog, num_consts):
        del num_consts  # Unused
        in_unknowns = [not t.pval.is_known for t in tracers]
        prog1, prog2, out_unknowns, num_res = self.partial_eval_prog(prog, in_unknowns)
        known_tracers, unknown_tracers = partition_list(in_unknowns, tracers)
        known_vals = [t.pval.const for t in known_tracers]
        outs1_res = jit_op(*known_vals, prog=prog1, num_consts=0)
        outs1, res = split_list(outs1_res, len(prog1.outs) - num_res)
        res_tracers = [trace.instantiate_const(self.full_raise(trace, x)) for x in res]
        outs2 = [
            PartialEvalTracerArray(trace, PartialVal.unknown(v.aval), None)
            for v in prog2.outs
        ]
        proto = InstrProto(
            jit_op,
            res_tracers + unknown_tracers,
            dict(prog=prog2, num_consts=0),
            [v.aval for v in prog2.outs],
            map(weakref.ref, outs2),
        )
        for t in outs2:
            t.proto = proto
        return merge_lists(out_unknowns, outs1, outs2)


class Backend:
    def __init__(
        self, name, default_dtype=BaseArray.float32, deps=("numpy as np", "math")
    ):
        self.name = name
        self.default_dtype = default_dtype
        self.impls = dict()
        self.dtype_map = dict()
        self.rt_ref: Runtime = None
        self.jit_fns = dict()
        self.deps_dict = dict()
        for dep in deps:
            if " as " in dep:  # e.g. "numpy as np"
                dep, _, dep_alias = dep.split(" ")
                self.deps_dict[dep] = importlib.import_module(dep)
                self.deps_dict[dep_alias] = self.deps_dict[dep]
            else:
                self.deps_dict[dep] = importlib.import_module(dep)

    def set_rt(self, rt):
        self.rt_ref = weakref.ref(rt)

    @property
    def rt(self):
        return self.rt_ref()

    def callable(
        self,
        hable_prog: Hable,
        hable_consts: Tuple[Hable, ...],
    ):
        key = hash((hable_prog, hable_consts))
        if (compiled := self.jit_fns.get(key, None)) is not None:
            return compiled
        prog: Prog = hable_prog.val
        self.rt.typecheck_prog(prog)
        consts = [x.val for x in hable_consts]
        in_avals = [v.aval for v in prog.in_binders[len(consts) :]]
        fn_name = f"{self.name.lower()}_fn"
        codegen_out = self.codegen(
            prog,
            consts + in_avals
            # consts=tuple([v for v in in_avals if type(v) is not VoidArray])
            # in_avals=tuple([v.aval for v in prog.outs])
        )
        fn = self.compile(prog, codegen_out, fn_name)
        compiled = sp.core.JitFn(self.rt, codegen_out, fn, consts)
        self.jit_fns[key] = compiled
        return compiled

    def codegen(self, prog, args, in_avals, name: str):
        "Returns IR from the Prog"
        raise NotImplementedError

    def compile(self, prog, args, in_avals, name: str):
        "Compiles IR to a Python callable function"
        raise NotImplementedError

    def set_dtype_map(self, dtype_map):
        self.dtype_map = dtype_map

    def set_codegen(self, fn):
        self.codegen = types.MethodType(fn, self)

    def set_compile(self, fn):
        self.compile = types.MethodType(fn, self)

    def set_impl(self, op):
        def set_impl_(fn):
            self.impls[op] = fn

        return set_impl_

    def run_impl(self, op, *args, **params):
        def process_arg(a):
            return (
                a.val
                if isinstance(a, BaseArray)
                else self.dtype_map[a]
                if isinstance(a, DType)
                else a
            )

        args = tuple([process_arg(a) for a in args])
        params = {k: process_arg(v) for k, v in params.items()}
        val = self.impls[op](*args, **params)
        return Array(self.rt, ArrayBuffer(val))

    def set_input_handler(self, typ, fn):
        self.input_handlers[typ] = fn


class JitFn:
    def __init__(self, rt, code, fn, consts):
        super().__init__()
        self.rt_ref = weakref.ref(rt)
        self.code = code
        self.fn = fn
        self.consts = consts

    @property
    def rt(self):
        return self.rt_ref()

    def __call__(self, *args, **params):
        # args = [a.val if isinstance(a, Array) else a for a in args]
        args = self.rt.tree_map(lambda a: a.val if isinstance(a, Array) else a, args)
        args, in_tree = self.rt.tree_flatten(args)
        try:
            # outs = self.fn(*self.consts, *args, **params)
            outs = self.fn(*args, **params)
        except Exception as e:
            print(self.code)
            print(e)

            raise
        return [self.rt.array(ArrayBuffer(o)) for o in outs]


BatchAxis = Union[None, int]


class BatchTracerArray(TracerArray):
    def __init__(self, rt, trace, val, batch_dim: BatchAxis):
        super().__init__(rt)
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
            return VoidArray(tuple(shape), aval.dtype)

    def full_lower(self):
        if self.batch_dim is None:
            return self.rt.full_lower(self.val)
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

    @staticmethod
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


class JVPTracerArray(TracerArray):
    def __init__(self, rt, trace, primal, tangent):
        super().__init__(rt)
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def aval(self):
        return self.rt.get_aval(self.primal)


class JVPTrace(Trace):
    pure = lift = lambda self, val: JVPTracerArray(
        self.rt, self, val, self.rt.procs.zeros_like(val)
    )
    # def pure(self, val):
    #     aval = self.rt.get_aval(val)
    #     val = aval if not isinstance(aval, TracerArray) else val

    #     return JVPTracerArray(
    #         self.rt, self, val, sp.rt.procs.zeros(aval.shape, aval.dtype)
    #     )

    # lift = pure

    def run_op(self, op, tracers, params):
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        primal_outs, tangent_outs = op.jvp(primals_in, tangents_in, **params)
        return [
            JVPTracerArray(self.rt, self, x, t)
            for x, t in list_zip(primal_outs, tangent_outs)
        ]


class ProgTracerArray(TracerArray):
    __slots__ = ["aval"]
    aval: VoidArray

    def __init__(self, rt, trace, aval):
        super().__init__(rt)
        self._trace = trace
        self.aval = aval


class ProgTrace(Trace):
    def new_arg(self, aval) -> ProgTracerArray:
        aval = VoidArray.like(aval)
        tracer = self.builder.new_tracer(self, aval)
        self.builder.tracer_to_var[id(tracer)] = Var(aval)

        return tracer

    def get_or_make_const_tracer(self, val: Any) -> ProgTracerArray:
        tracer = self.builder.const_tracers.get(id(val))
        if tracer is None:
            tracer = self.builder.new_tracer(self, self.rt.get_aval(val))
            self.builder.add_const(tracer, val)
        return tracer

    pure = lift = get_or_make_const_tracer

    def run_op(self, op, tracers, params):
        avals_in = [t.aval for t in tracers]
        avals_out = op.shape_eval(*avals_in, **params)
        out_tracers = [self.builder.new_tracer(self, a) for a in avals_out]
        inputs = [self.builder.getvar(t) for t in tracers]
        outvars = [self.builder.add_var(t) for t in out_tracers]
        self.builder.add_instr(Instr(op, inputs, params, outvars))
        return out_tracers

    @property
    def builder(self):
        return self.main.global_data


class ProgBuilder:
    instrs: List[Instr]
    tracer_to_var: Dict[int, Var]
    const_tracers: Dict[int, TracerArray]
    constvals: Dict[Var, Any]
    tracers: List[ProgTracerArray]

    def __init__(self, rt):
        self.rt_ref = weakref.ref(rt)
        self.instrs = []
        self.tracer_to_var = {}
        self.const_tracers = {}
        self.constvals = {}
        self.tracers = []

    @property
    def rt(self):
        return self.rt_ref()

    def new_tracer(self, trace: ProgTrace, aval: VoidArray) -> ProgTracerArray:
        tracer = ProgTracerArray(self.rt, trace, aval)
        self.tracers.append(tracer)
        return tracer

    def add_instr(self, instr: Instr) -> None:
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
        self.rt.typecheck_prog(prog)
        prog, constvals = self._inline_literals(prog, constvals)
        return prog, constvals

    def _inline_literals(self, prog: Prog, consts: List[Any]) -> Tuple[Prog, List[Any]]:
        const_binders, other_binders = split_list(prog.in_binders, len(consts))
        scalars = [
            type(x) in TracerArray.TYPES and not self.get_aval(x).shape for x in consts
        ]
        new_const_binders, lit_binders = partition_list(scalars, const_binders)
        new_consts, lit_vals = partition_list(scalars, consts)
        literals = dict(list_zip(lit_binders, list_map(Lit, lit_vals)))
        new_instrs = [
            Instr(
                instr.op,
                [literals.get(x, x) for x in instr.inputs],
                instr.params,
                instr.out_binders,
            )
            for instr in prog.instrs
        ]
        new_outs = [literals.get(x, x) for x in prog.outs]
        new_prog = Prog(new_const_binders + other_binders, new_instrs, new_outs)
        self.rt.typecheck_prog(new_prog)
        return new_prog, new_consts


class UndefPrimal(NamedTuple):
    aval: VoidArray

    @property
    def shape(self):
        return self.aval.shape

    @property
    def dtype(self):
        return self.aval.dtype


class PartialVal(NamedTuple):
    aval: VoidArray
    const: Optional[Any]

    is_known = property(lambda self: self.const is not None)
    is_unknown = property(lambda self: self.const is None)


class LambdaBindingProto(NamedTuple):
    pass


class ConstProto(NamedTuple):
    val: Any


class InstrProto(NamedTuple):
    prim: Op
    tracers_in: List["PartialEvalTracerArray"]
    params: Dict[str, Any]
    avals_out: List[VoidArray]
    tracer_refs_out: List[weakref.ReferenceType["PartialEvalTracerArray"]]


ProgProto = Union[LambdaBindingProto, ConstProto, InstrProto]


class PartialEvalTracerArray(TracerArray):
    pval: PartialVal
    proto: Optional[ProgProto]

    def __init__(self, rt, trace, pval, proto):
        super().__init__(rt)
        self._trace = trace
        self.pval = pval
        self.proto = proto

    aval = property(lambda self: self.pval.aval)

    def full_lower(self):
        if self.pval.is_known:
            return self.rt.full_lower(self.pval.const)
        return self


class PartialEvalTrace(Trace):
    def new_arg(self, pval: PartialVal) -> Any:
        return PartialEvalTracerArray(self.rt, self, pval, LambdaBindingProto())

    def lift(self, val: Any) -> PartialEvalTracerArray:
        return PartialEvalTracerArray(self.rt, self, self.rt.make_known_pval(val), None)

    pure = lift

    def instantiate_const(
        self, tracer: PartialEvalTracerArray
    ) -> PartialEvalTracerArray:
        if tracer.pval.is_unknown:
            return tracer
        else:
            pval = self.rt.make_unknown_pval(VoidArray.like(tracer.aval))
            return PartialEvalTracerArray(
                self.rt, self, pval, ConstProto(tracer.pval.const)
            )

    def run_op(self, op, tracers, params):
        if all(t.pval.is_known for t in tracers):
            return self.rt.bind(op, *list_map(self.rt.full_lower, tracers), **params)
        return op.partial_eval(self, tracers, **params)
