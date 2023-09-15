from dataclasses import dataclass
from pathlib import Path
import os
import json
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
    BinaryIO,
)
from collections.abc import Callable as abc_Callable
import weakref
import types
from contextlib import contextmanager
import itertools
import weakref
from collections import defaultdict, OrderedDict
from enum import Enum, auto
import operator as operator_py
import string
import numpy as np
import math
import inspect
from functools import partial, lru_cache

import slope
import importlib
import copy
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


class Hashed:
    val: Any

    def __init__(self, val):
        self.val = val

    def __hash__(self) -> int:
        return hash((self.val,))

    def __eq__(self, other):
        if isinstance(other, Hashed):
            return self.val == other.val
        return False


class OperatorType(Enum):
    Unary = auto()
    Binary = auto()
    Reduce = auto()
    Shape = auto()
    Load = auto()
    Other = auto()


class Operator:
    def __init__(self, name, op_type=OperatorType.Other):
        self.name = name
        self.op_type = op_type
        self.impls = dict()
        self.args_fixer = lambda *args, **params: (args, params)

    def __repr__(self) -> str:
        return f"Operator <{self.name}>"

    def reorg_args(self, args, params):
        sig = inspect.signature(self.run)
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
        args, params = self.reorg_args(args, params)
        args, params = self.args_fixer(*args, **params)
        return slope.M().backend.run_impl(self, *args, **params)

    def __call__(self, *args, **params):
        args, params = self.reorg_args(args, params)
        args, params = self.args_fixer(*args, **params)
        return slope.M().bind1(self, *args, **params)

    def args_fixer(self, *args, **params):
        raise NotImplementedError

    def run(self, *args, **params):
        raise NotImplementedError

    def partial_run(self, trace, tracers, **params):
        tracers_in = [trace.instantiate_const(t) for t in tracers]
        avals_in = [t.aval for t in tracers_in]
        avals_out = self.void_run(*avals_in, **params)
        tracers_out = [
            PartialEvalTracerArray(trace, slope.M().make_unknown_pval(aval), None)
            for aval in avals_out
        ]
        instruction = InstructionProto(
            self, tracers_in, params, avals_out, list_map(weakref.ref, tracers_out)
        )
        for t in tracers_out:
            t.proto = instruction
        return tracers_out

    def partial_run_instruction(self, unks_in, instruction):
        if any(unks_in):
            instruction1 = None
            instruction2 = Instruction(
                instruction.op,
                instruction.inputs,
                instruction.params,
                instruction.out_binders,
            )
            unks_out = [True for i in instruction.out_binders]
            res = [
                v
                for unk, v in zip(unks_in, instruction.inputs)
                if ((not unk) and type(v) is Var)
            ]
        else:
            instruction1 = instruction
            instruction2 = None
            unks_out = [False for i in instruction.out_binders]
            res = None

        return instruction1, instruction2, unks_out, res

    def jvp(self, *args, **params):
        raise NotImplementedError

    def T(self, *args, **params):
        raise NotImplementedError

    def vmap(self, *args, **params):
        raise NotImplementedError

    def void_run(self, *args, **params):
        raise NotImplementedError

    def set_partial_run_instruction(self, fn):
        self.partial_run_instruction = types.MethodType(fn, self)

    def set_args_fixer(self, fn):
        self.args_fixer = types.MethodType(fn, self)

    def set_partial_run(self, fn):
        self.partial_run = types.MethodType(fn, self)

    def set_run(self, fn):
        self.run = types.MethodType(fn, self)

    def set_jvp(self, fn):
        self.jvp = types.MethodType(fn, self)

    def set_vmap(self, fn):
        self.vmap = types.MethodType(fn, self)

    def set_T(self, fn):
        self.T = types.MethodType(fn, self)

    def set_void_run(self, fn):
        self.void_run = types.MethodType(fn, self)

    @classmethod
    def unary(cls, name):
        op = cls(name, OperatorType.Unary)

        @op.set_vmap
        def f(self, x, *, axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            return [self(x, **params)], [x_bdim]

        @op.set_void_run
        def f(self, x, **params):
            return [VoidArray(x.shape, x.dtype)]

        @op.set_jvp
        def f(self, primals, tangents, **params):
            (x,), (x_dot,) = primals, tangents
            return [self(x, **params)], [self(x_dot, **params)]

        return op

    @classmethod
    def binary(cls, name):
        op = cls(name, OperatorType.Binary)

        # binary op has many edge cases
        def impl(self, *args, **params):
            assert len(args) == 2
            if isinstance(args[0], Array) and isinstance(args[1], TracerArray):
                return op(args[0], args[1])
            else:
                assert isinstance(args[0], Array) or isinstance(args[1], Array)
                args, params = self.reorg_args(args, params)
                args, params = self.args_fixer(*args, **params)
                return slope.M().backend.run_impl(self, *args, **params)

        op.impl = types.MethodType(impl, op)

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
                x = slope.environment.array(x, dtype=y.dtype)
            elif type(y) in [bool, int, float]:
                y = slope.environment.array(y, dtype=x.dtype)

            if type(x) is Array and isinstance(y, TracerArray):
                x = y._trace.pure(x)
            elif type(y) is Array and isinstance(x, TracerArray):
                y = x._trace.pure(y)

            if (xshape := x.shape) == (yshape := y.shape):
                return (x, y), params
            shape_delta = len(xshape) - len(yshape)
            if shape_delta > 0:
                y = y.reshape((1,) * shape_delta + yshape)
            elif shape_delta < 0:
                x = x.reshape((1,) * -shape_delta + xshape)
            if (xshape := x.shape) == (yshape := y.shape):
                return (x, y), params

            shape_ret = tuple([max(x, y) for x, y in zip(xshape, yshape)])
            if xshape != shape_ret:
                x = x.broadcast_to(shape_ret)
            if yshape != shape_ret:
                y = y.broadcast_to(shape_ret)
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

        @op.set_void_run
        def f(self, x: VoidArray, y: VoidArray, **params) -> List[VoidArray]:
            if not type(x) in (Array, VoidArray) or not type(x) in (Array, VoidArray):
                raise TypeError
            if VoidArray.like(x) != VoidArray.like(y):
                breakpoint()
                raise TypeError(f"{x} != {y}")
            return [VoidArray(x.shape, x.dtype)]

        @op.set_jvp
        def f(self, primals, tangents, **params):
            (x,), (x_dot,) = primals, tangents
            return [self(x, **params)], [self(x_dot, **params)]

        return op

    @classmethod
    def reduce(cls, name):
        op = cls(name, OperatorType.Reduce)

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

        @op.set_void_run
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
        op = cls(name, OperatorType.Shape)
        return op

    @classmethod
    def load(cls, name):
        op = cls(name, OperatorType.Load)
        return op


# ========================
# Ops
# ========================


class OperatorSet:
    def register(self, op):
        setattr(self, op.name, op)

    def alias(self, op, name):
        assert op.name in vars(self)
        setattr(self, name, getattr(self, op.name))


class ProcedureSet:
    def register(self, fn):
        setattr(self, fn.__name__, fn)
        return fn

    def alias(self, fn, name):
        assert fn in vars(self)
        setattr(self, name, fn)


class DType(NamedTuple):
    priority: int
    itemsize: int
    name: str
    np: type

    def __repr__(self):
        return f"DType<{self.name}>"


class BaseArray:
    bool: Final[DType] = DType(0, 1, "bool", bool)
    float16: Final[DType] = DType(0, 2, "half", np.float16)
    float32: Final[DType] = DType(4, 4, "float", np.float32)
    int8: Final[DType] = DType(0, 1, "char", np.int8)
    int32: Final[DType] = DType(1, 4, "int", np.int32)
    int64: Final[DType] = DType(2, 8, "int64", np.int64)
    uint8: Final[DType] = DType(0, 1, "uchar", np.uint8)

    safe_dtypes = {
        "F16": float16,
        "F32": float32,
        "U8": uint8,
        "I8": int8,
        "I32": int32,
        "I64": int64,
    }
    safe_dtypes_inv = {v: k for k, v in safe_dtypes.items()}

    @property
    def default_dtype(self):
        return slope.M().backend.default_dtype

    def is_int(self) -> bool:
        return self.dtype in (self.int8, self.uint8, self.int32, self.int64)

    def is_float(self) -> bool:
        return self.dtype in (self.float16, self.float32)

    def is_unsigned(self) -> bool:
        return self.dtype is self.uint8

    def __getattr__(self, attr):
        raise NotImplementedError

    def __getitem__(self, idx):
        # if None in idx:
        #     self = self.broadcast_in_dim(self.shape, idx)
        self.getitem(idx)

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
    dtype: DType

    @classmethod
    def like(cls, aval):
        shape = aval.shape
        if isinstance(aval, Array):
            dtype = slope.M().backend.dtype_map_inv[aval.buf.val.dtype]
        else:
            dtype = aval.dtype
        return cls(shape, dtype)

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
            return False
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


class Instruction(NamedTuple):
    op: Operator
    inputs: List[Atom]
    params: Dict[str, Any]
    out_binders: List[Atom]


class Program(NamedTuple):
    in_binders: Any
    instructions: Tuple[Instruction]
    outs: Any

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        namegen = (
            # "z"+repr(r)
            # for r in itertools.count()
            "".join(s)
            for r in itertools.count(1)
            for s in itertools.permutations(string.ascii_lowercase, r)
        )
        names = defaultdict(lambda: next(namegen))
        in_binders = ", ".join(self.var_str(names, x) for x in self.in_binders)
        instructions = PPrint.vcat(
            [self.pp_instruction(names, e) for e in self.instructions]
        )
        outs = [names[v] if isinstance(v, Var) else str(v.val) for v in self.outs]
        outs = ", ".join(outs)
        # outs = ', '.join(sorted(outs))
        ret = str(
            PPrint.pp(f"{{ lambda {in_binders} .")
            + (
                (PPrint.pp("let ") >> instructions) + PPrint.pp(f"in ( {outs} ) }}")
            ).indent(2)
        )
        # print(ret)
        # print('program outs: ', outs)
        return ret

    def pp_instruction(
        self, names: DefaultDict[Var, str], instruction: Instruction
    ) -> PPrint:
        lhs = PPrint.pp(
            " ".join(self.var_str(names, v) for v in instruction.out_binders)
        )
        rhs = (
            PPrint.pp(repr(instruction.op.name))
            >> self.pp_params(instruction.params)
            >> PPrint.pp(
                " ".join(
                    names[x] if isinstance(x, Var) else str(x.val)
                    for x in instruction.inputs
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


class ProgramType(NamedTuple):
    in_types: Tuple[VoidArray]
    out_types: Tuple[VoidArray]

    def __repr__(self):
        in_types = ", ".join(aval.str_short() for aval in self.in_types)
        out_types = ", ".join(aval.str_short() for aval in self.out_types)
        return f"({in_types}) -> ({out_types})"


class MainTrace(NamedTuple):
    rt: "Machine"
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
        return op.run(*tracers, **params)


class ArrayBuffer:
    def __init__(self, val):
        self.val = val


class Array(BaseArray):
    def __init__(self, val: ArrayBuffer):
        assert isinstance(val, ArrayBuffer)
        self.buf = val

    def __hash__(self):
        return id(self.val)

    val = property(lambda self: self.buf.val)

    @property
    def dtype(self):
        return slope.M().backend.dtype_map_inv[self.buf.val.dtype]

    @property
    def device(self):
        return slope.M().backend.device_of(self)

    def numpy(self):
        return slope.M().backend.numpy_of(self)

    shape = property(lambda self: self.buf.val.shape)
    ndim = property(lambda self: self.buf.val.ndim)

    def __getattr__(self, attr):
        if attr in self.__dict__.keys():
            return self.__dict__[attr]
        if attr in vars(slope.environment.operator_set).keys():
            op = getattr(slope.environment.operator_set, attr)
            return partial(op.impl, self)
        elif attr in vars(slope.environment.procedure_set).keys():
            procedure = getattr(slope.environment.procedure_set, attr)
            assert not isinstance(
                procedure, classmethod
            ), f"use machine.{attr} instead of Array.{attr}"
            return partial(procedure, self)
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


@dataclass
class Environment:
    operator_set: OperatorSet
    procedure_set: ProcedureSet
    backends: dict

    def __getattr__(self, attr):
        try:
            # print(f"looking {attr} in operator_set")
            return getattr(self.operator_set, attr)
        except:
            pass
        try:
            # print(f"looking {attr} in procedure_set")
            return getattr(self.procedure_set, attr)
        except:
            pass
        # print(f"fallback to default getattribute")
        super().__getattribute__(attr)

    def array(
        self,
        val: Union[list, tuple, np.ndarray, "ArrayBuffer"] = None,
        dtype: Optional[Any] = BaseArray.float32,
    ):
        return (
            Array(val)
            if isinstance(val, ArrayBuffer)
            else slope.M().backend.run_impl(
                self.operator_set.constant, val=val, dtype=dtype
            )
        )

    def save(arr: "Array", filename: str):
        arr_np = arr.numpy()

    def safe_load(self, fn: Union[Array, str]) -> Dict[str, Array]:
        t = (
            fn
            if isinstance(fn, Array)
            else Array.empty(
                os.stat(fn).st_size, dtype=BaseArray.uint8, device=f"disk:{fn}"
            )
        )
        json_len = t[0:1].cast(BaseArray.int64).numpy()[0]
        metadata = json.loads(t[8 : 8 + json_len].numpy().tobytes())
        return {
            k: t[8 + json_len + v["data_offsets"][0] :]
            .cast(BaseArray.safe_dtypes[v["dtype"]])[: math.prod(v["shape"])]
            .reshape(v["shape"])
            for k, v in metadata.items()
            if k != "__metadata__"
        }

    def safe_save(self, Arrays: Dict[str, Array], fn: str):
        metadata, offset = {}, 0
        for k, v in Arrays.items():
            metadata[k] = {
                "dtype": BaseArray.safe_dtypes_inv[v.dtype],
                "shape": list(v.shape),
                "data_offsets": [offset, offset + v.nbytes()],
            }
            offset += v.nbytes()
        j = json.dumps(metadata, separators=(",", ":"))
        j += "\x20" * ((8 - len(j) % 8) % 8)
        Path(fn).unlink(missing_ok=True)
        t = Array.empty(8 + len(j) + offset, dtype=BaseArray.uint8, device=f"disk:{fn}")
        t[0:1].cast(BaseArray.int64).assign([len(j)])
        t[8 : 8 + len(j)].assign(
            Array(list(j.encode("utf-8")), dtype=BaseArray.uint8, device="cpu")
        )
        for k, v in self.safe_load(t).items():
            v.assign(Arrays[k])


#   def assign(self, x) -> Tensor:
#     # TODO: this is a hack for writing to DISK
#     if self.device.startswith("DISK"):
#       if x.__class__ is not Tensor: x = Tensor(x, device="CPU", dtype=self.dtype)
#       self.lazydata.realize().realized._copyin(x.numpy())  # type: ignore
#       return self
#     if x.__class__ is not Tensor: x = Tensor(x, device=self.device, dtype=self.dtype)
#     assert self.shape == x.shape and self.device == x.device, f"assign shape mismatch {self.shape} != {x.shape} or device mismatch {self.device} != {x.device}"
#     assert not x.requires_grad  # self requires_grad is okay?
#     if DEBUG >= 4: print(f"assign {self.lazydata} <- {x.lazydata}")
#     if self.lazydata.realized is not None and not getenv("DISALLOW_ASSIGN"): x.lazydata.output_buffer = self.lazydata.realized
#     self.lazydata = x.lazydata
#     return self


class TracerArray(BaseArray):
    TYPES = {
        bool,
        int,
        float,
    }
    _trace: "Trace"

    aval = property(lambda self: self.get_aval(self.val))
    dtype = property(lambda self: self.aval.dtype)
    shape = property(lambda self: self.aval.shape)

    @property
    def val(self):
        raise NotImplementedError

    # def __repr__(self):
    #     return f"{self.__class__.__name__}: {repr(self.aval)}"

    def __str__(self):
        return repr(self)

    def full_lower(self):
        return self

    def __getattr__(self, attr):
        if attr in vars(slope.environment.operator_set).keys():
            op = getattr(slope.environment.operator_set, attr)
            return partial(op, self)
        elif attr in vars(slope.environment.procedure_set).keys():
            procedure = getattr(slope.environment.procedure_set, attr)
            assert not isinstance(
                procedure, classmethod
            ), f"Access this procedure by Array.{attr}"
            return partial(procedure, self)
        return self.__getattribute__(attr)
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
    to_seq: Callable
    from_iterable: Callable


class PyTreeDef(NamedTuple):
    node_type: NodeType
    node_metadata: Hashable
    child_treedefs: Tuple["PyTreeDef", ...]

    def __repr__(self):
        return f"tree {self.node_type.name})\n\tchildren:{self.child_treedefs}"


class Leaf:
    def __repr__(self):
        return "<Leaf>"


leaf = Leaf()

jit_op = Operator("jit_op")
jit_op.reorg_args = lambda args, params: (args, params)


@jit_op.set_run
def f(self, *args, program, num_consts):
    hashed_program = Hashed(program)
    consts, args = args[:num_consts], args[num_consts:]
    hashed_consts = tuple(map(Hashed, consts))
    # print(args, hash(hashed_program), hash(hashed_consts))
    # print([o.aval for o in program.outs])
    
    jit_fn = slope.M().backend.callable(hashed_program, hashed_consts)
    return jit_fn(*consts, *args)
    # return jit_fn(*args, *consts)


@jit_op.set_jvp
def f(self, primals, tangents, *, program, num_consts):
    del num_consts
    new_program, new_consts = slope.M().jvp_program(program)
    outs = slope.M().bind(
        self,
        *new_consts,
        *primals,
        *tangents,
        program=new_program,
        num_consts=len(new_consts),
    )
    n = len(outs) // 2
    primals_out, tangents_out = outs[:n], outs[n:]
    return primals_out, tangents_out


@jit_op.set_void_run
def f(self, *in_types, program, num_consts):
    program_type = slope.M().typecheck_program(program)
    if not all(t1 == t2 for t1, t2 in zip(program_type.in_types, in_types)):
        raise TypeError
    return program_type.out_types


@jit_op.set_T
def f(self, cts, *invals, program, num_consts):
    undef_primals = [type(x) is UndefPrimal for x in invals]
    transposed_program, new_consts = slope.M().transpose_program(
        program, tuple(undef_primals)
    )
    residuals, _ = partition_list(undef_primals, invals)
    outs = slope.M().bind(
        self,
        *new_consts,
        *residuals,
        *cts,
        program=transposed_program,
        num_consts=len(new_consts),
    )
    outs = iter(outs)
    return [next(outs) if undef else None for undef in undef_primals]


@jit_op.set_partial_run
def f(self, trace, tracers, *, program, num_consts):
    in_unknowns = [not t.pval.is_known for t in tracers]
    program1, program2, out_unknowns, num_res = slope.M().partial_run_program(
        program, in_unknowns
    )
    known_tracers, unknown_tracers = partition_list(in_unknowns, tracers)
    known_vals = [t.pval.const for t in known_tracers]
    outs1_res = slope.M().bind(jit_op, *known_vals, program=program1, num_consts=0)
    outs1, res = split_list(outs1_res, len(program1.outs) - num_res)
    res_tracers = [trace.instantiate_const(slope.M().full_raise(trace, x)) for x in res]
    outs2 = [
        PartialEvalTracerArray(
            slope.M, trace, slope.M().make_unknown_pval(v.aval), None
        )
        for v in program2.outs
    ]
    instruction = InstructionProto(
        self,
        res_tracers + unknown_tracers,
        dict(program=program2, num_consts=0),
        [v.aval for v in program2.outs],
        list_map(weakref.ref, outs2),
    )
    for t in outs2:
        t.proto = instruction

    return merge_lists(out_unknowns, outs1, outs2)


@jit_op.set_partial_run_instruction
def f(
    self, unks_in, instruction
) -> Tuple[Instruction, Instruction, List[bool], List[Var]]:
    program = instruction.params["program"]
    program1, program2, out_unknowns, num_res = slope.M().partial_run_program(
        program, unks_in
    )
    ins1, ins2 = partition_list(unks_in, instruction.inputs)
    out_binders1, out_binders2 = partition_list(out_unknowns, instruction.out_binders)
    res = [Var(v.aval) for v in program2.in_binders[:num_res]]
    instruction1 = Instruction(
        self, ins1, dict(program=program1, num_consts=0), out_binders1 + res
    )
    instruction2 = Instruction(
        self, res + ins2, dict(program=program2, num_consts=0), out_binders2
    )
    return instruction1, instruction2, out_unknowns, res

class Module:
    @staticmethod
    def get_array_attrs(mod, prefix=""):
        def find(obj, prefix):
            if isinstance(obj, Array):
                return {prefix.strip("."):obj}
            if isinstance(obj, Module):
                return find(obj.__dict__, prefix)
            array_attrs = {}
            if isinstance(obj, dict):
                for k, v in obj.items():
                    array_attrs.update(find(v, f"{prefix}{str(k)}."))
            return array_attrs
        return tuple(find(mod, "").keys())
    
    def init_attr_paths(self):
        self.array_attrs = self.get_array_attrs()

    def to_seq(self):
        arrays = tuple(operator_py.attrgetter(attr)(self) for attr in self.array_attrs)
        return self, arrays
   
    @staticmethod
    def from_seq(mod, arrays):
        def set_nested_attr(obj, attr, value):
            nested_attrs = attr.split('.')
            target_obj = obj
            for a in nested_attrs[:-1]:
                target_obj = getattr(target_obj, a)
            setattr(target_obj, nested_attrs[-1], value)
        mod = copy.copy(mod)
        try:
            mod = copy.deepcopy(mod)
        except TypeError as e:
            pass
        for attr, array in list_zip(mod.array_attrs, arrays):
            set_nested_attr(mod, attr, array)
        return mod

def as_module(cls):
    original_init = cls.__init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.array_attrs = Module.get_array_attrs(self)
    cls.__init__ = new_init
    cls = type(cls.__name__, (cls, Module,), {})
    slope.M().register_node(cls, cls.to_seq, cls.from_seq)
    return cls

class Machine:
    def __init__(
        self,
        environment,
        default_backend="numpy",
    ):
        self.trace_stack: List[MainTrace] = []
        self.dynamic_trace: Optional[MainTrace] = None
        self.trace_stack += [MainTrace(self, 0, EvalTrace, None)]
        self.node_types = dict()
        self.register_node(tuple, lambda t: (None, t), lambda _, xs: tuple(xs))
        self.register_node(list, lambda l: (None, l), lambda _, xs: list(xs))
        self.register_node(
            dict,
            lambda d: list_map(tuple, unzip2(sorted(d.items()))),
            lambda keys, vals: dict(list_zip(keys, vals)),
        )
        self.register_node(
            UndefPrimal, lambda u: (u.aval, ()), lambda aval, _: UndefPrimal(aval)
        )

        self.environment = environment
        self.environment.operator_set.register(jit_op)
        self.backend = self.environment.backends[default_backend]

    """

def load_state_dict(model, state_dict, strict=True):
  with Timing("loaded weights in ", lambda et_ns: f", {GlobalCounters.mem_used/1e9:.2f} GB loaded at {GlobalCounters.mem_used/et_ns:.2f} GB/s"):
    model_state_dict = get_state_dict(model)
    if DEBUG >= 1 and len(state_dict) > len(model_state_dict): print("WARNING: unused weights in state_dict", sorted(list(state_dict.keys() - model_state_dict.keys())))
    for k,v in (t := tqdm(model_state_dict.items(), disable=CI)):
      t.set_description(f"ram used: {GlobalCounters.mem_used/1e9:5.2f} GB, {k:50s}")
      if k not in state_dict and not strict:
        if DEBUG >= 1: print(f"WARNING: not loading {k}")
        continue
      v.assign(state_dict[k].to(v.device)).realize()
"""

    def pprint_trace_stack(self):
        for trace in self.trace_stack:
            print(f"{trace.level}: {trace.trace_type.__name__}\t{trace.global_data=}")

    def make_known_pval(self, val: Any):
        return PartialVal(self.get_aval(val), val)

    def make_unknown_pval(self, aval: VoidArray):
        return PartialVal(aval, None)

    def get_aval(self, x):
        if isinstance(x, TracerArray):
            return x.aval
        elif type(x) in TracerArray.TYPES:
            return self.environment.array(x)
        elif isinstance(x, Array):
            return x
        elif isinstance(x, VoidArray):
            breakpoint()
            return x
        else:
            raise TypeError(type(x))

    def tree_to_seq(self, x: Any) -> Any:
        def _tree_to_seq(x_: Any) -> Tuple[Iterable, Union[PyTreeDef, Leaf]]:
            node_type = self.node_types.get(type(x_))
            if node_type:
                node_metadata, children = node_type.to_seq(x_)
                children_iterable, child_trees = unzip2(
                    list_map(_tree_to_seq, children)
                )
                to_seqed = itertools.chain.from_iterable(children_iterable)
                return to_seqed, PyTreeDef(
                    node_type, node_metadata, tuple(child_trees)
                )
            else:
                return [x_], leaf

        children_iter, treedef = _tree_to_seq(x)
        return list(children_iter), treedef

    def tree_from_seq(self, treedef: PyTreeDef, xs: List[Any]) -> Any:
        def _tree_from_seq(treedef_: PyTreeDef, xs_: Iterator) -> Any:
            if treedef_ is leaf:
                return next(xs_)
            else:
                children = (_tree_from_seq(t, xs_) for t in treedef_.child_treedefs)
                return treedef_.node_type.from_iterable(
                    treedef_.node_metadata, children
                )

        return _tree_from_seq(treedef, iter(xs))

    def seq_fn_wrap(self, f, in_tree):
        store = Store()

        def seq_fun(*args_iterable):
            pytree_args = self.tree_from_seq(in_tree, args_iterable)
            out = f(*pytree_args)
            out_iterable, out_tree = self.tree_to_seq(out)
            store.set_value(out_tree)
            return out_iterable

        return seq_fun, store

    def register_node(self, ty: Type, to_iter: Callable, from_iter: Callable) -> None:
        self.node_types[ty] = NodeType(str(ty), to_iter, from_iter)

    def tree_map(self, f: Callable[..., Any], tree: Any) -> Any:
        leaves, treedef = self.tree_to_seq(tree)
        # ret =self.tree_from_seq(treedef, tuple(f(leaf) for leaf in leaves));breakpoint()
        return self.tree_from_seq(treedef, tuple(f(leaf) for leaf in leaves))

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

    def typecheck_program(self, program: Program) -> ProgramType:
        environment: Set[Var] = set()

        for v in program.in_binders:
            if v in environment:
                raise TypeError
            environment.add(v)

        for instruction in program.instructions:
            in_types = [self.typecheck_atom(environment, x) for x in instruction.inputs]
            out_types = instruction.op.void_run(*in_types, **instruction.params)
            for out_binder, out_type in list_zip(instruction.out_binders, out_types):
                if not out_type == out_binder.aval:
                    raise TypeError
            for out_binder in instruction.out_binders:
                if out_binder in environment:
                    raise TypeError
                environment.add(out_binder)

        in_types = [v.aval for v in program.in_binders]
        out_types = [self.typecheck_atom(environment, x) for x in program.outs]
        return ProgramType(tuple(in_types), tuple(out_types))

    def typecheck_atom(self, environment: Set[Var], x: Atom) -> VoidArray:
        if isinstance(x, Var):
            if x not in environment:
                raise TypeError("unbound variable")
            return x.aval
        elif isinstance(x, Lit):
            return self.get_aval(x.val)
        else:
            assert False

    def run_program(self, program: Program, args: List[Any]) -> List[Any]:
        environment: Dict[Var, Any] = {}

        def read(x: Atom) -> Any:
            return environment[x] if type(x) is Var else x.val

        def write(v: Var, val: Any) -> None:
            assert v not in environment  # single-assignment
            environment[v] = val

        list_map(write, program.in_binders, args)
        for instruction in program.instructions:
            in_vals = list_map(read, instruction.inputs)
            outs = self.bind(instruction.op, *in_vals, **instruction.params)
            list_map(write, instruction.out_binders, outs)
        return list_map(read, program.outs)

    def program_as_fun(self, program: Program):
        return lambda *args: self.run_program(program, args)

    def vmap_iterable(self, f, in_axes, *args):
        axi_set = {x.shape[ax] for x, ax in list_zip(args, in_axes) if ax is not None}
        assert len(axi_set) == 1
        (axis_size,) = axi_set
        with self.new_main(BatchTrace, axis_size) as main:
            trace = BatchTrace(main)
            tracers_in = [
                BatchTracerArray(trace, x, ax) if ax is not None else x
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
            args_iterable, in_tree = self.tree_to_seq(args)
            in_axes_iterable, in_tree2 = self.tree_to_seq(in_axes)
            if in_tree != in_tree2:
                raise TypeError(f"{in_tree}\n!=\n{in_tree2}")
            f_iterable, out_tree_store = self.seq_fn_wrap(f, in_tree)
            outs_iterable = self.vmap_iterable(f_iterable, in_axes_iterable, *args_iterable)
            return self.tree_mnis(out_tree_store(), outs_iterable)

        return batched_f

    def jvp_iterable(self, f, primals, tangents):
        with self.new_main(JVPTrace) as main:
            trace = JVPTrace(main)
            tracers_in = [
                JVPTracerArray(trace, x, t) for x, t in list_zip(primals, tangents)
            ]
            outs = f(*tracers_in)
            tracers_out = [self.full_raise(trace, out) for out in outs]
            primals_out, tangents_out = unzip2(
                (t.primal, t.tangent) for t in tracers_out
            )
        return primals_out, tangents_out

    def jvp(self, f, primals, tangents):
        primals_iterable, in_tree = self.tree_to_seq(primals)
        tangents_iterable, in_tree2 = self.tree_to_seq(tangents)
        if in_tree != in_tree2:
            raise TypeError
        f, out_tree_store = self.seq_fn_wrap(f, in_tree)
        primals_out_iterable, tangents_out_iterable = self.jvp_iterable(
            f, primals_iterable, tangents_iterable
        )
        primals_out = self.tree_from_seq(out_tree_store(), primals_out_iterable)
        tangents_out = self.tree_from_seq(out_tree_store(), tangents_out_iterable)
        return primals_out, tangents_out

    def jacfwd(self, f, x):
        pushfwd = lambda v: self.jvp(f, (x,), (v,))[1]
        vecs_in = self.environment.eye(math.prod(x.shape)).reshape(x.shape * 2)
        return self.vmap(pushfwd, (0,))(vecs_in)

    @lru_cache
    def make_program(
        self,
        f: Callable,
        *avals_in: VoidArray,
    ) -> Tuple[Program, List[Any], PyTreeDef]:
        avals_in, in_tree = self.tree_to_seq(avals_in)
        f, out_tree_store = self.seq_fn_wrap(f, in_tree)

        builder = ProgramBuilder()
        with self.new_main(ProgramTrace, builder) as main:
            with self.new_dynamic(main):
                trace = ProgramTrace(main)
                tracers_in = [trace.new_arg(aval) for aval in avals_in]
                outs = f(*tracers_in)
                tracers_out = [self.full_raise(trace, out) for out in outs]
                program, consts = builder.build(tracers_in, tracers_out)
        return program, consts, out_tree_store()

    @lru_cache
    def jvp_program(self, program: Program) -> Tuple[Program, List[Any]]:
        def jvp_traceable(*primals_and_tangents):
            n = len(primals_and_tangents) // 2
            primals, tangents = primals_and_tangents[:n], primals_and_tangents[n:]
            return self.jvp(self.program_as_fun(program), primals, tangents)

        in_avals = self.tree_map(lambda v: v.aval, program.in_binders)
        new_program, new_consts, _ = self.make_program(
            jvp_traceable, *in_avals, *in_avals
        )
        return new_program, new_consts

    def partial_run_iterable(
        self, f: Callable, pvals_in: List["PartialVal"]
    ) -> Tuple[Program, List["PartialVal"], List[Any]]:
        with self.new_main(PartialEvalTrace) as main:
            trace = PartialEvalTrace(main)
            tracers_in = [trace.new_arg(pval) for pval in pvals_in]
            outs = f(*tracers_in)
            tracers_out = [self.full_raise(trace, out) for out in outs]
            pvals_out = [t.pval for t in tracers_out]
            unk_tracers_in = [t for t in tracers_in if t.pval.is_unknown]
            unk_tracers_out = [t for t in tracers_out if t.pval.is_unknown]
            program, consts = self.tracers_to_program(unk_tracers_in, unk_tracers_out)

        return program, pvals_out, consts

    def partial_run_program(
        self,
        program: Program,
        in_unknowns: List[bool],
        instantiate: Optional[List[bool]] = None,
    ) -> Tuple[Program, Program, List[bool], int]:
        environment: Dict[Var, bool] = {}
        residuals: Set[Var] = set()

        def read(x: Atom) -> bool:
            return type(x) is Var and environment[x]

        def write(unk: bool, v: Var) -> None:
            environment[v] = unk

        def new_res(x: Atom) -> Atom:
            if type(x) is Var:
                residuals.add(x)
            return x

        instructions1, instructions2 = [], []
        list_map(write, in_unknowns, program.in_binders)

        for instruction in program.instructions:
            unks_in = list_map(read, instruction.inputs)
            (
                instruction1,
                instruction2,
                unks_out,
                res,
            ) = instruction.op.partial_run_instruction(unks_in, instruction)
            if instruction1 is not None:
                instructions1.append(instruction1)
            if instruction2 is not None:
                instructions2.append(instruction2)
            if res is not None:
                residuals.update(res)
            list_map(write, unks_out, instruction.out_binders)
            # if any(unks_in):
            #     inputs = [v if unk else new_res(v) for unk, v in zip(unks_in, instruction.inputs)]
            #     instructions2.append(InstructionProto(instruction.primitive, inputs, instruction.params, instruction.out_binders))
            #     map(partial(write, True), instruction.out_binders)
            # else:
            #     instructions1.append(instruction)
            #     map(partial(write, False), instruction.out_binders)

        out_unknowns = list_map(read, program.outs)
        if instantiate is not None:
            for v, uk, inst in zip(program.outs, out_unknowns, instantiate):
                if inst and not uk:
                    if type(v) is Var:
                        residuals.add(v)
            out_unknowns = list_map(operator_py.or_, out_unknowns, instantiate)

        residuals, num_res = list(residuals), len(residuals)
        assert all(type(v) is Var for v in residuals), residuals

        ins1, ins2 = partition_list(in_unknowns, program.in_binders)
        outs1, outs2 = partition_list(out_unknowns, program.outs)

        program1 = Program(ins1, instructions1, outs1 + residuals)
        program2 = Program(residuals + ins2, instructions2, outs2)
        self.typecheck_partial_run_program(
            program, in_unknowns, out_unknowns, program1, program2
        )

        return program1, program2, out_unknowns, num_res

    def typecheck_partial_run_program(
        self, program, in_unknowns, out_unknowns, program1, program2
    ):
        programty = self.typecheck_program(program)  # (a1,  a2) -> (b1, b2 )
        program1ty = self.typecheck_program(program1)  #  a1       -> (b1, res)
        program2ty = self.typecheck_program(program2)  # (res, a2) -> b2

        a1, a2 = partition_list(in_unknowns, programty.in_types)
        b1, b2 = partition_list(out_unknowns, programty.out_types)
        b1_, res = split_list(program1ty.out_types, len(b1))
        res_, a2_ = split_list(program2ty.in_types, len(res))
        b2_ = program2ty.out_types

        a1 = tuple(a1)
        a2, a2_ = tuple(a2), tuple(a2_)
        b1, b1_ = tuple(b1), tuple(b1_)
        b2, b2_ = tuple(b2), tuple(b2_)
        res, res_ = tuple(res), tuple(res_)

        if program1ty.in_types != a1:
            raise TypeError
        if program2ty.out_types != b2:
            raise TypeError
        if b1 != b1_:
            raise TypeError
        if res != res_:
            raise TypeError
        if a2 != a2_:
            raise TypeError
        if b2 != b2_:
            raise TypeError

    def linearize_iterable(self, f, *primals_in):
        pvals_in = [self.make_known_pval(x) for x in primals_in] + [
            self.make_unknown_pval(VoidArray.like(self.get_aval(x))) for x in primals_in
        ]

        def f_jvp(*primals_tangents_in):
            primals_out, tangents_out = self.jvp(f, *split_half(primals_tangents_in))
            return [*primals_out, *tangents_out]

        program, pvals_out, consts = self.partial_run_iterable(f_jvp, pvals_in)
        primal_pvals, _ = split_half(pvals_out)
        assert all(pval.is_known for pval in primal_pvals)
        primals_out = [pval.const for pval in primal_pvals]
        f_lin = lambda *tangents: self.run_program(program, [*consts, *tangents])
        return primals_out, f_lin

    def linearize(self, f, *primals_in):
        primals_in_iterable, in_tree = self.tree_to_seq(primals_in)
        f, out_tree_store = self.seq_fn_wrap(f, in_tree)
        primals_out_iterable, f_lin_iterable = self.linearize_iterable(f, *primals_in_iterable)
        primals_out = self.tree_from_seq(out_tree_store(), primals_out_iterable)

        def f_lin(*tangents_in):
            tangents_in_iterable, in_tree2 = self.tree_to_seq(tangents_in)
            if in_tree != in_tree2:
                raise TypeError
            tangents_out_iterable = f_lin_iterable(*tangents_in_iterable)
            return self.tree_from_seq(out_tree_store(), tangents_out_iterable)

        return primals_out, f_lin

    def tracers_to_program(
        self,
        tracers_in: List["PartialEvalTracerArray"],
        tracers_out: List["PartialEvalTracerArray"],
    ):
        def tracer_parents(t: PartialEvalTracerArray) -> List[PartialEvalTracerArray]:
            return t.proto.tracers_in if isinstance(t.proto, InstructionProto) else []

        def proto_to_instruction(
            tracer_to_var: Dict[int, Var], proto: InstructionProto
        ) -> Instruction:
            inputs = [tracer_to_var[id(t)] for t in proto.tracers_in]
            out_binders = [Var(aval) for aval in proto.avals_out]
            for t_ref, var in list_zip(proto.tracer_refs_out, out_binders):
                if t_ref() is not None:
                    tracer_to_var[id(t_ref())] = var
            return Instruction(proto.prim, inputs, proto.params, out_binders)

        tracer_to_var: Dict[int, Var] = {
            id(t): Var(VoidArray.like(t.aval)) for t in tracers_in
        }
        constvar_to_val: Dict[int, Any] = {}
        constid_to_var: Dict[int, Var] = {}
        procedureessed_instructions: Set[int] = set()
        instructions: List[Instruction] = []
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
            elif isinstance(t.proto, InstructionProto):
                if id(t.proto) not in procedureessed_instructions:
                    instructions.append(proto_to_instruction(tracer_to_var, t.proto))
                    procedureessed_instructions.add(id(t.proto))
            else:
                raise TypeError(t.proto)

        constvars, constvals = unzip2(constvar_to_val.items())
        in_binders = constvars + [tracer_to_var[id(t)] for t in tracers_in]
        out_vars = [tracer_to_var[id(t)] for t in tracers_out]
        program = Program(tuple(in_binders), tuple(instructions), tuple(out_vars))
        self.typecheck_program(program)
        return program, constvals

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

    def vjp_iterable(self, f, *primals_in):
        pvals_in = [self.make_known_pval(x) for x in primals_in] + [
            self.make_unknown_pval(VoidArray.like(self.get_aval(x))) for x in primals_in
        ]
        primal_pvals_in, tangent_pvals_in = split_half(pvals_in)

        def f_jvp(*primals_tangents_in):
            primals_out, tangents_out = self.jvp(f, *split_half(primals_tangents_in))
            return [*primals_out, *tangents_out]

        program, pvals_out, consts = self.partial_run_iterable(f_jvp, pvals_in)  # linearize
        primal_pvals, _ = split_half(pvals_out)
        assert all(pval.is_known for pval in primal_pvals)
        primals_out = [pval.const for pval in primal_pvals]
        transpose_inputs = consts + [UndefPrimal(p.aval) for p in tangent_pvals_in]
        f_vjp = lambda *cts: self.run_program_transposed(program, transpose_inputs, cts)
        return primals_out, f_vjp

    def vjp(self, f, *primals_in):
        primals_in_iterable, in_tree = self.tree_to_seq(primals_in)
        f, out_tree_store = self.seq_fn_wrap(f, in_tree)
        primals_out_iterable, f_vjp_iterable = self.vjp_iterable(f, *primals_in_iterable)
        primals_out = self.tree_from_seq(out_tree_store(), primals_out_iterable)

        def f_vjp(*cotangents_out):
            cotangents_out_iterable, _ = self.tree_to_seq(cotangents_out)
            cotangents_in_iterable = f_vjp_iterable(*cotangents_out_iterable)

            return self.tree_from_seq(in_tree, cotangents_in_iterable)

        return primals_out, f_vjp

    def run_program_transposed(
        self, program: Program, args: List[Any], cotangents: List[Any]
    ) -> List[Any]:
        primal_environment: Dict[Var, Any] = {}
        ct_environment: Dict[Var, Any] = {}

        def read_primal(x: Atom) -> Any:
            return (
                primal_environment.get(x, UndefPrimal(x.aval))
                if type(x) is Var
                else x.val
            )

        def write_primal(v: Var, val: Any) -> None:
            if type(val) is not UndefPrimal:
                primal_environment[v] = val

        def read_cotangent(v: Var) -> Any:
            return ct_environment.pop(
                v, self.environment.zeros(v.aval.shape, v.aval.dtype)
            )

        def write_cotangent(x: Atom, val: Any):
            if type(x) is Var and val is not None:
                ct_environment[x] = (
                    ct_environment[x] + val if x in ct_environment else val
                )

        list_map(write_primal, program.in_binders, args)
        list_map(write_cotangent, program.outs, cotangents)
        # print(len(program.instructions))
        # for i, instruction in enumerate(program.instructions[::-1]):
        #     print(i, instruction)
        for instruction in program.instructions[::-1]:
            primals_in = list_map(read_primal, instruction.inputs)
            cts_in = list_map(read_cotangent, instruction.out_binders)
            inp, params = primals_in, instruction.params
            inp, params = instruction.op.reorg_args(inp, params)
            inp, params = instruction.op.args_fixer(*inp, **params)
            cts_out = instruction.op.T(cts_in, *inp, **params)
            list_map(write_cotangent, instruction.inputs, cts_out)

        ret = [
            read_cotangent(v)
            for v, x in list_zip(program.in_binders, args)
            if type(x) is UndefPrimal
        ]

        return ret

    @lru_cache
    def transpose_program(
        self, program: Program, undef_primals: tuple[bool, ...]
    ) -> tuple[Program, list[Any]]:
        avals_in, avals_out = self.typecheck_program(program)
        traceable = partial(self.run_program_transposed, program)
        args = [UndefPrimal(a) if u else a for a, u in zip(avals_in, undef_primals)]
        trans_program, consts, _ = self.make_program(
            traceable, tuple(args), tuple(avals_out)
        )
        self.typecheck_program(trans_program)

        return trans_program, consts

    def grad(self, f, *, ret_fval=False):
        def gradfun(x, *xs):
            y, f_vjp = self.vjp(f, x, *xs)
            if np.shape(y) != ():
                raise TypeError
            if ret_fval:
                out = f_vjp(self.environment.ones(()))
                return y, out
            else:
                x_bar, *_ = f_vjp(self.environment.ones(()))
                return x_bar

        if f.__qualname__ == "Machine.jit.<locals>.f_jitted":
            # unjit then jit back
            f = f.__closure__[0].cell_contents
            return self.jit(gradfun)
        else:
            return gradfun

    def jit(self, f):
        def f_jitted(*args):
            args_ = args
            avals_in = self.tree_map(lambda x: VoidArray.like(self.get_aval(x)), args)
            program, consts, out_tree = self.make_program(f, *avals_in)

            args, in_tree = self.tree_to_seq(args)
            outs = self.bind(
                jit_op,
                *consts,
                *args,
                program=program,
                num_consts=len(consts),
            )
            return self.tree_from_seq(out_tree, outs)

        return f_jitted

    def jit_partial_run(self, trace, tracers, *, program, num_consts):
        del num_consts  # Unused
        in_unknowns = [not t.pval.is_known for t in tracers]
        program1, program2, out_unknowns, num_res = self.partial_run_program(
            program, in_unknowns
        )
        known_tracers, unknown_tracers = partition_list(in_unknowns, tracers)
        known_vals = [t.pval.const for t in known_tracers]
        outs1_res = jit_op(*known_vals, program=program1, num_consts=0)
        outs1, res = split_list(outs1_res, len(program1.outs) - num_res)
        res_tracers = [trace.instantiate_const(self.full_raise(trace, x)) for x in res]
        outs2 = [
            PartialEvalTracerArray(trace, PartialVal.unknown(v.aval), None)
            for v in program2.outs
        ]
        proto = InstructionProto(
            jit_op,
            res_tracers + unknown_tracers,
            dict(program=program2, num_consts=0),
            [v.aval for v in program2.outs],
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
        self.dtype_map_inv = dict()
        self.deps_dict = dict()
        self.codegen_depth = 0
        self.codegen_idx = 0
        for dep in deps:
            if " as " in dep:  # e.g. "numpy as np"
                dep, _, dep_alias = dep.split(" ")
                self.deps_dict[dep] = importlib.import_module(dep)
                self.deps_dict[dep_alias] = self.deps_dict[dep]
            else:
                self.deps_dict[dep] = importlib.import_module(dep)

    def device_of(self, array):
        raise NotImplementedError

    def numpy_of(self, array):
        raise NotImplementedError

    @lru_cache
    def callable(
        self,
        hashed_program: Hashed,
        hashed_consts: Tuple[Hashed, ...],
    ):
        program: Program = hashed_program.val
        slope.M().typecheck_program(program)
        consts = [x.val for x in hashed_consts]
        in_avals = [v.aval for v in program.in_binders[len(consts) :]]
        fn_name = f"{self.name.lower()}_fn"
        codegen_out = self.codegen(
            program,
            consts + in_avals
            # consts=tuple([v for v in in_avals if type(v) is not VoidArray])
            # in_avals=tuple([v.aval for v in program.outs])
        )
        fn, code = self.compile(program, codegen_out, fn_name)
        compiled = JitFn(code, fn, consts)
        return compiled

    def codegen(self, program, args, in_avals, name: str):
        "Returns IR from the Program"
        raise NotImplementedError

    def compile(self, program, args, in_avals, name: str):
        "Compiles IR to a Python callable function"
        raise NotImplementedError

    def set_dtype_map(self, dtype_map):
        self.dtype_map = dtype_map
        self.dtype_map_inv = {v: k for k, v in dtype_map.items()}

    def set_device_of(self, fn):
        self.device_of = types.MethodType(fn, self)

    def set_numpy_of(self, fn):
        self.numpy_of = types.MethodType(fn, self)

    def set_codegen(self, fn):
        self.codegen = types.MethodType(fn, self)

    def set_compile(self, fn):
        self.compile = types.MethodType(fn, self)

    def set_impl(self, op):
        def set_impl_(fn):
            self.impls[op] = types.MethodType(fn, self)

        return set_impl_

    def run_impl(self, op, *args, **params):
        def procedureess_arg(a):
            return (
                a.val
                if isinstance(a, Array)
                else self.dtype_map[a]
                if isinstance(a, DType)
                else a
            )

        args = tuple([procedureess_arg(a) for a in args])
        params = {k: procedureess_arg(v) for k, v in params.items()}
        val = self.impls[op](*args, **params)
        return Array(ArrayBuffer(val))

    def set_input_handler(self, typ, fn):
        self.input_handlers[typ] = fn


class JitFn:
    def __init__(self, code, fn, consts):
        super().__init__()
        self.code = code
        self.fn = fn
        self.consts = consts

    def __call__(self, *args, **params):
        # args = [a.val if isinstance(a, Array) else a for a in args]
        args = slope.M().tree_map(lambda a: a.val if isinstance(a, Array) else a, args)
        args, in_tree = slope.M().tree_to_seq(args)
        try:
            # outs = self.fn(*self.consts, *args, **params)
            outs = self.fn(*args, **params)
        except Exception as e:
            print(self.code)
            breakpoint()
            raise
        return [slope.environment.array(ArrayBuffer(o)) for o in outs]


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
            return VoidArray(tuple(shape), aval.dtype)

    def full_lower(self):
        if self.batch_dim is None:
            return slope.M().full_lower(self.val)
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
            x = x.broadcast_in_dim(target_shape)
            return x
        elif src == dst:
            return x
        else:
            perm = [i for i in range(len(x.shape)) if i != src]
            perm.insert(dst, src)
            return x.transpose(perm)


class JVPTracerArray(TracerArray):
    def __init__(self, trace, primal, tangent):
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def aval(self):
        return slope.M().get_aval(self.primal)

    @property
    def val(self):
        return self.primal

    @property
    def dtype(self):
        return self.primal.dtype


class JVPTrace(Trace):
    # pure = lift = lambda self, val: JVPTracerArray(self, val, slope.environment.zeros_like(val))
    def pure(self, val):
        if isinstance(val, PartialEvalTrace):
            val = val.pval.const
        # elif isinstance(val, Array):
        # val = val
        return JVPTracerArray(self, val, slope.environment.zeros_like(val))

    lift = pure

    def run_op(self, op, tracers, params):
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        primal_outs, tangent_outs = op.jvp(primals_in, tangents_in, **params)
        return [
            JVPTracerArray(self, x, t) for x, t in list_zip(primal_outs, tangent_outs)
        ]


class ProgramTracerArray(TracerArray):
    __slots__ = ["aval"]
    aval: VoidArray

    def __init__(self, trace, aval):
        self._trace = trace
        self.aval = aval


class ProgramTrace(Trace):
    def new_arg(self, aval) -> ProgramTracerArray:
        aval = VoidArray.like(aval)
        tracer = self.builder.new_tracer(self, aval)
        self.builder.tracer_to_var[id(tracer)] = Var(aval)

        return tracer

    def get_or_make_const_tracer(self, val: Any) -> ProgramTracerArray:
        tracer = self.builder.const_tracers.get(id(val))
        if tracer is None:
            tracer = self.builder.new_tracer(self, slope.M().get_aval(val))
            self.builder.add_const(tracer, val)
        return tracer

    pure = lift = get_or_make_const_tracer

    def run_op(self, op, tracers, params):
        avals_in = [t.aval for t in tracers]
        avals_out = op.void_run(*avals_in, **params)
        out_tracers = [self.builder.new_tracer(self, a) for a in avals_out]
        inputs = [self.builder.getvar(t) for t in tracers]
        outvars = [self.builder.add_var(t) for t in out_tracers]
        self.builder.add_instruction(Instruction(op, inputs, params, outvars))
        return out_tracers

    @property
    def builder(self):
        return self.main.global_data


class ProgramBuilder:
    instructions: List[Instruction]
    tracer_to_var: Dict[int, Var]
    const_tracers: Dict[int, TracerArray]
    constvals: Dict[Var, Any]
    tracers: List[ProgramTracerArray]

    def __init__(self):
        self.instructions = []
        self.tracer_to_var = {}
        self.const_tracers = {}
        self.constvals = {}
        self.tracers = []

    def new_tracer(self, trace: ProgramTrace, aval: VoidArray) -> ProgramTracerArray:
        tracer = ProgramTracerArray(trace, aval)
        self.tracers.append(tracer)
        return tracer

    def add_instruction(self, instruction: Instruction) -> None:
        self.instructions.append(instruction)

    def add_var(self, tracer: ProgramTracerArray) -> Var:
        assert id(tracer) not in self.tracer_to_var
        var = self.tracer_to_var[id(tracer)] = Var(tracer.aval)
        return var

    def getvar(self, tracer: ProgramTracerArray) -> Var:
        var = self.tracer_to_var.get(id(tracer))
        assert var is not None
        return var

    def add_const(self, tracer: ProgramTracerArray, val: Any) -> Var:
        var = self.add_var(tracer)
        self.const_tracers[id(val)] = tracer
        self.constvals[var] = val
        return var

    def build(self, in_tracers: Any, out_tracers: Any) -> Tuple[Program, List[Any]]:
        constvars, constvals = unzip2(self.constvals.items())
        t2v = lambda t: self.tracer_to_var[id(t)]
        in_binders = constvars + [t2v(t) for t in in_tracers]
        out_vars = [t2v(t) for t in out_tracers]
        program = Program(in_binders, self.instructions, out_vars)
        slope.M().typecheck_program(program)
        program, constvals = self._inline_literals(program, constvals)
        return program, constvals

    def _inline_literals(
        self, program: Program, consts: List[Any]
    ) -> Tuple[Program, List[Any]]:
        const_binders, other_binders = split_list(program.in_binders, len(consts))
        scalars = [
            type(x) in TracerArray.TYPES and not slope.M().get_aval(x).shape
            for x in consts
        ]
        new_const_binders, lit_binders = partition_list(scalars, const_binders)
        new_consts, lit_vals = partition_list(scalars, consts)
        literals = dict(list_zip(lit_binders, list_map(Lit, lit_vals)))
        new_instructions = [
            Instruction(
                instruction.op,
                [literals.get(x, x) for x in instruction.inputs],
                instruction.params,
                instruction.out_binders,
            )
            for instruction in program.instructions
        ]
        new_outs = [literals.get(x, x) for x in program.outs]
        new_program = Program(
            new_const_binders + other_binders, new_instructions, new_outs
        )
        slope.M().typecheck_program(new_program)
        return new_program, tuple(new_consts)


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


class InstructionProto(NamedTuple):
    prim: Operator
    tracers_in: List["PartialEvalTracerArray"]
    params: Dict[str, Any]
    avals_out: List[VoidArray]
    tracer_refs_out: List[weakref.ReferenceType["PartialEvalTracerArray"]]


ProgramProto = Union[LambdaBindingProto, ConstProto, InstructionProto]


class PartialEvalTracerArray(TracerArray):
    def __init__(self, trace, pval, proto):
        self._trace = trace
        self.pval = pval
        self.proto = proto

    aval = property(lambda self: self.pval.aval)
    val = property(lambda self: self.pval.const)

    def full_lower(self):
        if self.pval.is_known:
            return slope.M().full_lower(self.pval.const)
        return self


class PartialEvalTrace(Trace):
    def new_arg(self, pval: PartialVal) -> Any:
        return PartialEvalTracerArray(self, pval, LambdaBindingProto())

    def lift(self, val: Any) -> PartialEvalTracerArray:
        return PartialEvalTracerArray(self, slope.M().make_known_pval(val), None)

    pure = lift

    def instantiate_const(
        self, tracer: PartialEvalTracerArray
    ) -> PartialEvalTracerArray:
        if tracer.pval.is_unknown:
            return tracer
        else:
            pval = slope.M().make_unknown_pval(VoidArray.like(tracer.aval))
            return PartialEvalTracerArray(self, pval, ConstProto(tracer.pval.const))

    def run_op(self, op, tracers, params):
        if all(t.pval.is_known for t in tracers):
            return slope.M().bind(
                op, *list_map(slope.M().full_lower, tracers), **params
            )
        return op.partial_run(self, tracers, **params)


# serialization
