from pathlib import Path
import os
import json
from typing import (
    Callable,
    NamedTuple,
    Dict,
    Hashable,
    List,
    Any,
    Iterable,
    Sequence,
    Iterator,
    Type,
    Tuple,
    Optional,
    Union,
    Dict,
    Set,
    DefaultDict,
    Final,
)
import weakref
import types
from contextlib import contextmanager, ContextDecorator
import itertools
import weakref
import operator as operator_py
import numpy as np
import math
import inspect
from functools import partial, lru_cache
import mmap
import traceback
import importlib
import time
import cProfile
import pstats

# =================================
#   Utils
# =================================


class Timing(ContextDecorator):
    def __init__(self, prefix="", on_exit=None, enabled=True):
        self.prefix, self.on_exit, self.enabled = prefix, on_exit, enabled

    def __enter__(self):
        self.st = time.perf_counter_ns()

    def __exit__(self, *exc):
        self.et = time.perf_counter_ns() - self.st
        if self.enabled:
            print(f"{self.prefix}{self.et*1e-6:6.2f} ms" + (self.on_exit(self.et) if self.on_exit else ""))


def colored(st, color: Optional[str], background=False):
    return (
        f"\u001b[{10*background+60*(color.upper() == color)+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color.lower())}m{st}\u001b[0m"
        if color is not None
        else st
    )  # replace the termcolor library with one line  # noqa: E501


def _format_fcn(fcn):
    return f"{fcn[0]}:{fcn[1]}:{fcn[2]}"


class Profiling(ContextDecorator):
    def __init__(self, enabled=True, sort="cumtime", frac=0.2, fn=None, ts=1):
        self.enabled, self.sort, self.frac, self.fn, self.time_scale = enabled, sort, frac, fn, 1e3 / ts

    def __enter__(self):
        self.pr = cProfile.Profile()
        if self.enabled:
            self.pr.enable()

    def __exit__(self, *exc):
        if self.enabled:
            self.pr.disable()
            if self.fn:
                self.pr.dump_stats(self.fn)
            stats = pstats.Stats(self.pr).strip_dirs().sort_stats(self.sort)
            for fcn in stats.fcn_list[0 : int(len(stats.fcn_list) * self.frac)]:  # type: ignore[attr-defined]
                (_primitive_calls, num_calls, tottime, cumtime, callers) = stats.stats[fcn]  # type: ignore[attr-defined]
                scallers = sorted(callers.items(), key=lambda x: -x[1][2])
                print(
                    f"n:{num_calls:8d}  tm:{tottime*self.time_scale:7.2f}ms  tot:{cumtime*self.time_scale:7.2f}ms",
                    colored(_format_fcn(fcn), "yellow") + " " * (50 - len(_format_fcn(fcn))),
                    colored(f"<- {(scallers[0][1][2]/tottime)*100:3.0f}% {_format_fcn(scallers[0][0])}", "BLACK") if len(scallers) else "",
                )


def dblog(*msg, enable=True):
    if enable:
        print(*msg)


def unzip2(pairs) -> Tuple[List[Any], List[Any]]:
    lst1, lst2 = [], []
    for i1, i2 in pairs:
        lst1 += [i1]
        lst2 += [i2]
    return lst1, lst2


def list_map(f: Callable, *xs: Iterable) -> List[Any]:
    return list(map(f, *xs))


def list_zip(*args: List[Any]) -> List[Any]:
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


# def partition_list(bs: List[bool], l: List[Any]) -> Tuple[List[Any], List[Any]]:
#     assert len(bs) == len(l)
#     lists = lst1, lst2 = [], []
#     for b, x in list_zip(bs, l):
#         lst = lists[int(b)]
#         lst += [x]
#     breakpoint()
#     return lst1, lst2


def split_list(lst: List[Any], n: int) -> Tuple[List[Any], List[Any]]:
    assert 0 <= n <= len(lst)
    return lst[:n], lst[n:]


def partition_list(bs: List[bool], l: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert len(bs) == len(l)
    lists = lst1, lst2 = [], []
    for b, x in zip(bs, l):
        lists[b].append(x)
    return lst1, lst2


def lru_cache_verbose(
    maxsize: int = 100,
    typed: bool = False,
    tb_start: int = -12,
    tb_end: int = -7,
):
    def decorator(fn: Callable):
        @lru_cache(maxsize=maxsize, typed=typed)
        def wrapper(*args, **kwargs) -> Callable:
            return fn(*args, **kwargs)

        def decorated_function(*args, **kwargs) -> Any:
            result = wrapper(*args, **kwargs)
            cache_info = wrapper.cache_info()

            dblog(
                f"{fn.__name__}.{cache_info} {args.__hash__()}",
                enable=backend.LOG_LRU,
            )
            tb = "".join(traceback.format_list(traceback.extract_stack())[tb_start:tb_end]).replace("\n    ", ":\t") + "-" * 20 + "\n"
            dblog(f"{tb}", enable=backend.LOG_LRU)

            return result

        decorated_function.cache_info = wrapper.cache_info
        decorated_function.fn = fn
        return decorated_function

    return decorator


def cuda_is_available():
    try:
        import subprocess
        import platform

        cmd = f"nvidia-smi{'.exe' if platform.system == 'Windows' else ''}"
        result = subprocess.run([cmd], stdout=subprocess.PIPE)
        output = result.stdout.decode("utf-8")
        return True if "NVIDIA-SMI" in output else False
    except FileNotFoundError:
        return False


class Hashed:
    val: Any

    def __init__(self, val):
        self.val = val

    def __hash__(self) -> int:
        return hash((self.val,))

    def __eq__(self, other):
        if isinstance(other, Hashed):
            if isinstance(self.val, Tensor) and isinstance(other.val, Tensor):
                # because Tensor.__eq__ already for Tensor.equal
                return id(self.val) == id(other.val)
            return self.val == other.val
        return False

    def __repr__(self):
        return f"Hashed: {repr(self.val)}"


# =================================
#   Tensors
# =================================


class DType(NamedTuple):
    priority: int
    itemsize: int
    name: str
    mlir: str
    numpy: type

    @property
    def format_code(self):
        return f"slope.{self.name}"

    def __repr__(self):
        return f"<DType: {self.name}>"


class dtypes:
    float32: Final[DType] = DType(4, 4, "float32", "f32", np.float32)
    uint8: Final[DType] = DType(0, 1, "uint8", "u8", np.uint8)
    int8: Final[DType] = DType(0, 1, "int8", "i8", np.int8)
    bool: Final[DType] = DType(0, 1, "bool", "i1", bool)
    int32: Final[DType] = DType(1, 4, "int32", "i32", np.int32)
    int64: Final[DType] = DType(2, 8, "int64", "i64", np.int64)
    uint64: Final[DType] = DType(2, 8, "uint64", "ui64", np.uint64)
    float16: Final[DType] = DType(0, 2, "float16", "f16", np.float16)

    all_dtypes = (bool, float16, float32, int8, int32, int64, uint8, uint64)
    name_dtype_map = {k.name: k for k in all_dtypes}
    name_dtype_map_inv = {v: k for k, v in name_dtype_map.items()}
    mlir_dtype_map = {k.mlir: k for k in all_dtypes}
    mlir_dtype_map_inv = {v: k for k, v in mlir_dtype_map.items()}

    @classmethod
    def is_int(cls, dtype):
        return dtype in (cls.uint8, cls.int8, cls.int32, cls.uint64, cls.int64)

    @classmethod
    def is_float(cls, dtype):
        return dtype in (cls.float16, cls.float32)


class Device(NamedTuple):
    name: str
    idx: int

    @property
    def format_code(self):
        return f"'{self.name}:{self.idx}'"

    def __repr__(self):
        return f"<Device: {self.format_code}>"


class devices:
    cpu: Final[Device] = Device("cpu", 0)
    metal: Final[Device] = Device("metal", 0)
    cuda0: Final[Device] = Device("cuda", 0)
    # TODO: programmatically define this class attrs to support other setup
    cuda = cuda0
    all_devices = (cpu, metal, cuda0)
    name_idx_device_map = {f"{k.name}:{k.idx}": k for k in all_devices}
    name_idx_device_map_inv = {v: k for k, v in name_idx_device_map.items()}


class TensorBuffer:
    def __init__(self, val):
        self.val = val


class Tensor:
    def __init__(self, val: TensorBuffer):
        assert isinstance(val, TensorBuffer)
        self.buf = val

    @property
    def symval(self):
        return SymbolicTensor.like(self)

    @property
    def default_dtype(self):
        return backend.default_dtype

    def is_int(self) -> bool:
        return self.dtype in (
            dtypes.int8,
            dtypes.uint8,
            dtypes.uint64,
            dtypes.int32,
            dtypes.int64,
        )

    def is_float(self) -> bool:
        return self.dtype in (dtypes.float16, dtypes.float32)

    def is_unsigned(self) -> bool:
        return self.dtype is dtypes.uint8

    def to_bool(self):
        return self.cast(dtypes.bool)

    def short(self):
        return self.cast(dtypes.int8)

    def int(self):
        return self.cast(dtypes.int32)

    def long(self):
        return self.cast(dtypes.int64)

    def half(self):
        return self.cast(dtypes.float16)

    def float(self):
        return self.cast(dtypes.float32)

    def __getattr__(self, attr):
        if attr in vars(backend.operator_set).keys():
            op = getattr(backend.operator_set, attr)
            return partial(op, self)
        elif attr in vars(backend.procedure_set).keys():
            procedure = getattr(backend.procedure_set, attr)
            assert not isinstance(procedure, classmethod), f"use {attr} instead of self.{attr}"
            return partial(procedure, self)
        else:
            return self.__getattribute__(attr)

    def __getitem__(self, idx):
        return self.getitem(idx)

    def __setitem__(self, idx, item):
        raise NotImplementedError

    def str_short(self):
        return f"<Tensor: shape={self.shape}, dtype={self.dtype}>"

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
    __pow__ = lambda self, other: self.pow(other)
    __rpow__ = lambda self, other: self.pow.func(other, self)
    __matmul__ = lambda self, other: self.matmul(other)
    __rmatmul__ = lambda self, other: self.matmul.func(other, self)
    __invert__ = lambda self: self.invert()
    __eq__ = lambda self, other: self.equal(other)
    __ne__ = lambda self, other: self.not_equal(other)
    __ge__ = lambda self, other: self.greater_equal(other)
    __le__ = lambda self, other: self.less_equal(other)
    __gt__ = lambda self, other: self.greater(other)
    __lt__ = lambda self, other: self.less(other)

    def __hash__(self):
        return id(self.val)

    val = property(lambda self: self.buf.val)

    def size(self, i):
        return self.shape[i]

    @property
    def dtype(self):
        return backend.dtype_of(self)

    @property
    def device(self):
        return backend.device_of(self)

    def numpy(self, memmap=False):
        return backend.numpy_of(self, memmap)

    @property
    def shape(self):
        return backend.shape_of(self)

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return self.dtype.itemsize

    def nbytes(self):
        return self.numel() * self.element_size()

    def __repr__(self):
        return f"<Tensor: val=\n{self.numpy()}\nshape={self.shape}, dtype={self.dtype.name}, device={self.device.format_code}>"


class SymbolicTensor(Tensor):
    def __init__(self, shape, dtype, device):
        assert isinstance(dtype, DType)
        self._shape = tuple(int(i) for i in shape)
        self._dtype = dtype
        self._device = device

    @property
    def symval(self):
        return self

    @property
    def val(self):
        raise RuntimeError(f"SymbolicTensor actually has no val, from {trace_stack[-1]=}, ")

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    def like(self, **overrides):
        shape = overrides.get("shape", self.shape)
        dtype = overrides.get("dtype", self.dtype)
        device = overrides.get("device", self.device)
        return SymbolicTensor(shape, dtype, device)

    def str_short(self):
        return f'{str(self.dtype)}[{",".join(str(d) for d in self.shape)}]'

    def __hash__(self):
        return hash((self.shape, self.dtype))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return (self.shape == other.shape) and (self.dtype == other.dtype)

    def __repr__(self):
        return f"<SymbolicTensor: shape={self.shape}, dtype={self.dtype.name}, device={self.device}>"


# =================================
#   Operator
# =================================


class Operator:
    def __init__(self, name, variadic_inputs=False, nary_outputs=False):
        self.name = name
        self.variadic_inputs = variadic_inputs
        self.nary_outputs = nary_outputs
        if self.variadic_inputs:
            self.reorg_args = self.reorg_args_nary

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Operator):
            return False
        return self.name == other.name

    def args_fixer(self, *args, **params):
        return args, params

    def __call__(self, *args, **params):
        args, params = self.reorg_args(args, params)
        args, params = self.args_fixer(*args, **params)
        ret = bind(self, *args, **params)
        if not self.nary_outputs:
            ret = ret[0]
        return ret

    def __repr__(self) -> str:
        return f"<{self.name}>"

    def typecheck(self, *args, **params):
        raise NotImplementedError

    def jvp(self, *args, **params):
        raise NotImplementedError

    def T(self, *args, **params):
        raise NotImplementedError

    def vmap(self, *args, **params):
        raise NotImplementedError

    def reorg_args(self, args, params):
        sig = inspect.signature(self.typecheck)
        args_strs = [k for k, v in sig.parameters.items() if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and k != "self"]
        params_strs = [k for k, v in sig.parameters.items() if v.kind == inspect.Parameter.KEYWORD_ONLY and k != "self"]

        if args:
            if len(args) > len(args_strs):
                args, rest = args[: len(args_strs)], args[len(args_strs) :]
                if params_strs:
                    new_params = {k: rest_arg for k, rest_arg in zip(params_strs, rest) if k not in params}
                    params = {**new_params, **params}
            else:
                args = tuple([params[k] if k in params else arg for k, arg in zip(args_strs, args)])
                assert len(args) == len(args_strs)
        return args, params

    def reorg_args_nary(self, args, params):
        return args, params

    def partial_run(self, trace, tracers, **params):
        tracers_in = [trace.instantiate_const(t) for t in tracers]
        symvals_in = [t.symval for t in tracers_in]
        symvals_out = self.typecheck(*symvals_in, **params)
        tracers_out = [PartialRunTraceTensor(trace, make_unknown_pval(symval), None) for symval in symvals_out]
        instruction = InstructionDraft(
            self,
            tracers_in,
            params,
            symvals_out,
            list_map(weakref.ref, tracers_out),
        )
        for t in tracers_out:
            t.draft = instruction
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
            res = [v for unk, v in zip(unks_in, instruction.inputs) if ((not unk) and type(v) is Var)]
        else:
            instruction1 = instruction
            instruction2 = None
            unks_out = [False for i in instruction.out_binders]
            res = None

        return instruction1, instruction2, unks_out, res


class MetaOperator(Operator):
    def meta_impl(self, *args, **kwargs):
        raise NotImplementedError


class UnaryOperator(Operator):
    def vmap(self, x, *, dim_size, vals_in, dims_in, **params):
        (x,), (x_bdim,) = vals_in, dims_in
        return [self(x, **params)], [x_bdim]

    def typecheck(self, x, **params):
        return [SymbolicTensor.like(x)]

    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [self(x, **params)], [self(x_dot, **params)]


class BinaryOperator(Operator):
    boolean_output = False

    def args_fixer(self, x, w, **params):
        if isinstance(x, UndefinedPrimal) or type(w) is UndefinedPrimal:
            assert x.shape == w.shape
            return (x, w), params

        if type(x) in TraceTensor.PYTHON_TYPES:
            x = backend.full(shape=(), fill_value=x, dtype=w.dtype)
        elif type(w) in TraceTensor.PYTHON_TYPES:
            w = backend.full(shape=(), fill_value=w, dtype=x.dtype)

        shape_delta = x.ndim - w.ndim
        if shape_delta > 0:
            w = w.reshape((1,) * shape_delta + w.shape)
        elif shape_delta < 0:
            x = x.reshape((1,) * -shape_delta + x.shape)

        shape_ret = tuple([max(x, w) for x, w in zip(x.shape, w.shape)])
        if x.shape != shape_ret:
            x = x.expand(shape_ret)
        if w.shape != shape_ret:
            w = w.expand(shape_ret)

        if type(x) is Tensor and isinstance(w, TraceTensor):
            x = w._trace.pure(x)
        elif type(w) is Tensor and isinstance(x, TraceTensor):
            w = x._trace.pure(w)
        # TODO: https://jax.readthedocs.io/en/latest/type_promotion.html
        if x.dtype != w.dtype:
            # {int, bool} -> float
            if dtypes.is_float(x.dtype) ^ dtypes.is_float(w.dtype):
                if dtypes.is_float(w.dtype):
                    x = x.cast(w.dtype)
                elif dtypes.is_float(x.dtype):
                    w = w.cast(x.dtype)
            # bool -> int
            elif dtypes.is_int(x.dtype) ^ dtypes.is_int(w.dtype):
                if dtypes.is_int(w.dtype):
                    x = x.cast(w.dtype)
                elif dtypes.is_int(x.dtype):
                    w = w.cast(x.dtype)
            else:  # TODO: fine-grained type promotions
                raise NotImplementedError("No other type promotion rules")

        return (x, w), params

    def vmap(self, dim_size, vals_in, dims_in, **params):
        (x, w), (x_bdim, w_bdim) = vals_in, dims_in
        if x_bdim != w_bdim:
            if x_bdim is None:
                x = VMapTrace.move_vmap_dim(x, dim_size, x_bdim, w_bdim)
                x_bdim = w_bdim
            else:
                w = VMapTrace.move_vmap_dim(w, dim_size, w_bdim, x_bdim)
        return [self(x, w, **params)], [x_bdim]

    def typecheck(self, x: SymbolicTensor, y: SymbolicTensor, **params) -> List[SymbolicTensor]:
        if not isinstance(x, (Tensor, SymbolicTensor)) or not isinstance(y, (Tensor, SymbolicTensor)):
            raise TypeError
        symx = SymbolicTensor.like(x, dtype=dtypes.bool if self.boolean_output else x.dtype)
        symy = SymbolicTensor.like(y, dtype=dtypes.bool if self.boolean_output else y.dtype)
        if x.dtype != y.dtype:
            raise TypeError(f"x.dtype ({x.dtype}) != y.dtype ({y.dtype})")
        if symx == symy:
            return [symx]
        shape_delta = len(symx.shape) - len(symy.shape)
        if shape_delta > 0:
            symy = symy.like(shape=(1,) * shape_delta + symy.shape)
        elif shape_delta < 0:
            symx = symx.like(shape=(1,) * -shape_delta + symx.shape)
        if symx == symy:
            return [symx]
        else:
            shape_ret = tuple([max(x, w) for x, w in zip(symx.shape, symy.shape)])
            if symx.shape != shape_ret:
                symx = symx.like(shape=shape_ret)
            if symy.shape != shape_ret:
                symy = symx.like(shape=shape_ret)
            if symx != symy:
                raise TypeError(f"symx ({symx}) != symy ({symy})")
            return [symx]

    def jvp(self, primals, tangents, **params):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [self(x, w, **params)], [self(x_dot, w_dot, **params)]

    def T(self, cotangents, x, w):
        (gL_y,) = cotangents
        if self.boolean_output:
            gL_y = gL_y.cast(x.dtype)
        if isinstance(x, UndefinedPrimal):
            return [gL_y, NullCotangent]
        elif isinstance(w, UndefinedPrimal):
            return [NullCotangent, gL_y]
        else:
            raise ValueError


class ReduceOperator(Operator):
    def args_fixer(self, x, *, dim=None, keepdim=False):
        if dim is None:
            dim = tuple(range(x.ndim))
        elif isinstance(dim, int):
            dim = (dim,)
        dim = tuple(a if a >= 0 else a + len(x.shape) for a in dim)
        return (x,), dict(dim=dim, keepdim=keepdim)

    def vmap(self, dim_size, vals_in, dims_in, *, dim, keepdim):
        (x,), (x_bdim,) = vals_in, dims_in
        dim = tuple(a + (x_bdim <= a) for a in dim)
        out_bdim = x_bdim - sum(a < x_bdim for a in dim)
        return [self(x, dim=dim, keepdim=keepdim)], [out_bdim]

    def typecheck(self, x: SymbolicTensor, *, dim=None, keepdim=False) -> List[SymbolicTensor]:
        dim = list(set([a + len(x.shape) if a < 0 else a for a in dim]))
        if keepdim:
            new_shape = [d if i not in dim else 1 for i, d in enumerate(x.shape)]
        else:
            new_shape = [d for i, d in enumerate(x.shape) if i not in dim]
        return [SymbolicTensor.like(x, shape=tuple(new_shape))]


class InitOperator(Operator):
    def vmap(self, dim_size, vals_in, dims_in, **params):
        (x_bdim,) = dims_in
        y = self(**params)
        y = y.unsqueeze(x_bdim)
        return [y], [x_bdim]

    def jvp(self, primals, tangents, **params):
        y = self(**params)
        y_dot = NullCotangent(y.symval)
        return [y], [y_dot]

    def T(self, cotangents, **params):
        return [NullCotangent(cotangents[0])]


class ShapeOperator(Operator):
    pass


class GeneralReduceOperator(Operator):
    pass


class OperatorSet:
    def __init__(self):
        self.register("jit_op")(JitOp)

    def register(self, name, variadic_inputs=False, nary_outputs=False, aliases=()):
        def wrap(op_cls):
            assert name not in vars(self)
            op = op_cls(name, variadic_inputs, nary_outputs)
            setattr(self, name, op)
            for a in aliases:
                setattr(self, a, op)
            return op_cls

        return wrap


class ProcedureSet:
    def register(self, aliases=()):
        def wrap(f):
            assert f.__name__ not in vars(self)
            setattr(self, f.__name__, f)
            for a in aliases:
                setattr(self, a, f)
            return f

        return wrap


class CodegenOutput(NamedTuple):
    code_lines: List[str]
    fn_defs: Dict[str, List[str]]
    in_binders: List["ProgramEnvVar"]
    outs: List["ProgramEnvVar"]


class Backend:
    LOG_LRU = int(os.environ.get("LOG_LRU", 0))
    LOG_JIT = int(os.environ.get("LOG_JIT", 0))
    LOG_TREE = int(os.environ.get("LOG_TREE", 0))
    LOG_BACKEND = int(os.environ.get("LOG_BACKEND", 0))
    LOG_PROGRAM = int(os.environ.get("LOG_PROGRAM", 0))
    LOG_INIT = int(os.environ.get("LOG_INIT", 1))
    device_var = os.environ.get("DEFAULT_DEVICE", "cpu:0")
    if device_var[-2] != ":":
        device_var += ":0"
    DEFAULT_DEVICE = devices.name_idx_device_map[device_var]
    DEFAULT_DTYPE = dtypes.name_dtype_map[os.environ.get("DEFAULT_DTYPE", "float32")]
    dtype_for_indices: DType = None  # need to override

    def __init__(
        self,
        operator_set: OperatorSet,
        procedure_set: ProcedureSet,
    ):
        self.operator_set = operator_set
        self.procedure_set = procedure_set
        self.node_types = dict()
        self.impls = dict()
        self.register_node(tuple, lambda t: (None, t), lambda _, xs: tuple(xs), "tuple")
        self.register_node(list, lambda l: (None, l), lambda _, xs: list(xs), "list")
        self.register_node(
            dict,
            lambda d: list_map(tuple, unzip2(sorted(d.items()))),
            lambda keys, vals: dict(list_zip(keys, vals)),
            "dict",
        )
        self.register_node(
            UndefinedPrimal,
            lambda u: (u.symval, ()),
            lambda symval, _: UndefinedPrimal(symval),
            "UndefinedPrimal",
        )

    def set_impl(self, op: Union[types.LambdaType, types.FunctionType]):
        def set_impl_(fn):
            self.impls[op] = types.MethodType(fn, self)

        return set_impl_

    def register_node(self, ty: Type, to_iter: Callable, from_iter: Callable, name=None) -> None:
        if name is None:
            name = str(ty)
        self.node_types[ty] = NodeType(name, to_iter, from_iter)

    def __getattr__(self, attr):
        try:
            dblog(
                f"Looking {self}.{attr} in operator_set",
                enable=backend.LOG_BACKEND,
            )
            return getattr(self.operator_set, attr)
        except:
            pass
        try:
            dblog(
                f"Looking {self}.{attr} in procedure_set",
                enable=backend.LOG_BACKEND,
            )
            return getattr(self.procedure_set, attr)
        except:
            pass
        dblog(
            f"Fallback to default {self} getattribute",
            enable=backend.LOG_BACKEND,
        )
        super().__getattribute__(attr)

    def tensor(
        self,
        val: Union[list, tuple, np.ndarray, "TensorBuffer"] = None,
        dtype: Optional[Any] = None,
        device=None,
    ):
        if isinstance(val, TensorBuffer):
            return Tensor(val)
        elif isinstance(val, Tensor):
            return val
        if type(val) is bytes:
            val = np.frombuffer(val, dtype=dtype)
        return self.from_numpy(val, dtype, device)

    def symbolic_tensor(
        self,
        shape: Union[list, tuple, np.ndarray, "TensorBuffer"] = None,
        dtype: Optional[Any] = None,
        device=None,
    ):
        dtype = dtype or self.DEFAULT_DTYPE
        device = device or self.DEFAULT_DEVICE
        return SymbolicTensor(shape, dtype, device)

    def seed(self, seed):
        raise NotImplementedError

    @property
    def default_dtype_value(self):
        return self.dtype_map[backend.DEFAULT_DTYPE]

    def set_method(self, method):
        setattr(self, method.__name__, types.MethodType(method, self))

    def from_numpy(self, val, device):
        raise NotImplementedError

    def numpy_of(self, tensor):
        raise NotImplementedError

    def device_of(self, tensor):
        raise NotImplementedError

    def shape_of(self, tensor):
        raise NotImplementedError

    def dtype_of(self, tensor):
        raise NotImplementedError

    @lru_cache_verbose()
    def jit_program(
        self,
        hashed_program: Hashed,
        hashed_consts: Tuple[Hashed, ...],
    ):
        program: Program = hashed_program.val
        typecheck_program(program)
        consts = [x.val for x in hashed_consts]
        in_symvals = [v.symval for v in program.in_binders[len(consts) :]]
        codegen_output: CodegenOutput = self.codegen(program, consts + in_symvals, fn_name="main")
        fn, code = self.compile(codegen_output)
        jit_output = JitOutput(program, codegen_output, fn, code, consts)
        return jit_output

    def codegen(self, program: "Program", args: Tuple, in_symvals: Tuple, name: str):
        "Returns compiler IR from the Program"
        raise NotImplementedError

    def compile(self, program: "Program", args: Tuple, in_symvals: Tuple, name: str):
        "Compiles compiler IR to a Python callable function"
        raise NotImplementedError

    def export(self, jit_output, *args, **params):
        raise NotImplementedError

    def load(self, path, single_key="_tensor"):
        with open(path, mode="rb") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as m:
                json_len = np.int64(m[0])
                start = 8 + json_len
                metadata = json.loads(m[8:start])
                ret = {}
                for k, v in metadata.items():
                    if k != "__metadata__":
                        dtype = Tensor.mlir_dtype_map[(v["dtype"])]
                        data_start = start + v["data_offsets"][0]
                        data_end = start + v["data_offsets"][1]
                        t_np = np.frombuffer(m[data_start:data_end], dtype=dtype.numpy())
                        t = backend.tensor(t_np, dtype=dtype)
                        t = t.reshape(tuple(v["shape"]))
                        ret[k] = t
                if len(ret) == 1 and single_key in ret.keys():
                    return ret[single_key]
                return ret

    def save(self, tensors: Dict[str, Tensor], path: str, single_key="_tensor"):
        if isinstance(tensors, Tensor):
            tensors = {single_key: tensors}
        else:
            assert all((isinstance(k, str) and isinstance(v, Tensor)) for k, v in tensors.items())

        metadata, offset = {}, 0
        for k, v in tensors.items():
            metadata[k] = {
                "dtype": v.dtype.mlir,
                "shape": list(v.shape),
                "data_offsets": [offset, offset + v.nbytes()],
            }
            offset += v.nbytes()
        j = json.dumps(metadata, separators=(",", ":"))
        Path(path).unlink(missing_ok=True)
        jbytes = j.encode("utf-8")
        start = 8 + len(jbytes)
        with open(path, mode="wb") as f:  # make empty file, fill with enough space
            f.write(b"\x00" * (start + offset))
        with open(path, mode="r+b") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_WRITE) as m:
                m[0:8] = np.int64(len(j)).tobytes()
                m[8:start] = jbytes
                for t, tm in zip(tensors.values(), metadata.values()):
                    data_start, data_end = tm["data_offsets"]
                    m[start + data_start : start + data_end] = t.numpy().tobytes()


# =================================
#   Program
# =================================


class Var:
    def __init__(self, symval):
        self.symval = symval
        self.val = None


class Lit:
    def __init__(self, val):
        self.symval = SymbolicTensor.like(get_symval(val))
        self.val = val


Atom = Union[Var, Lit]


class Instruction(NamedTuple):
    op: Operator
    inputs: List[Atom]
    params: Dict[str, Any]
    out_binders: List[Atom]


class ProgramEnvVar(NamedTuple):
    name: str
    symval: SymbolicTensor
    is_const: bool = False

    @property
    def shape(self):
        return self.symval.shape

    @property
    def dtype(self):
        return self.symval.dtype

    @property
    def device(self):
        return self.symval.device

    @property
    def ndim(self):
        return self.symval.ndim

    def numpy(self):
        return self.symval.numpy()

    def __repr__(self):
        return f"<ProgramEnvVar: name={self.name}, symval={self.symval}>"

    str_short = __repr__


class Program:
    def __init__(
        self,
        in_binders: Any,
        instructions: Tuple[Instruction],
        outs: Any,
        num_consts: int = 0,
        static_args: Any = (),
        name: str = "my_program",
        indent_amount=4,
    ):
        self.in_binders: Any = in_binders
        self.outs: Any = outs
        self.instructions = self.prune_instructions(instructions, outs)
        # self.instructions = instructions
        self.num_consts: int = num_consts
        self.static_args = static_args
        self.name: str = name
        self.indent_amount: int = indent_amount

        self.env: Dict[ProgramEnvVar, Any] = dict()
        for inb in self.in_binders:
            prefix = "x" if type(inb.symval) is SymbolicTensor else "c"
            idx = sum([1 if v.name[0] == prefix else 0 for v in self.env.values()])
            self.env[inb] = ProgramEnvVar(f"{prefix}{idx}", inb.symval, True if prefix == "c" else False)
        for instruction in self.instructions:
            if len(instruction.out_binders) == 0:
                continue
            for outb in instruction.out_binders:
                prefix = "y" if outb in self.outs else "z"
                idx = sum([1 if v.name[0] == prefix else 0 for v in self.env.values()])
                self.env[outb] = ProgramEnvVar(f"{prefix}{idx}", outb.symval)
        self.curr_repr = repr(self)

    def pprint_shape(self, symval, scalar_as_empty_array=False):
        xdtype = symval.dtype.mlir
        if len(symval.shape) > 0:
            xshape = f"{', '.join((repr(i) for i in symval.shape))}"
            return f"[{xshape}, {xdtype}]"
        else:
            return f"[{xdtype}]"

    def pprint_sig(self, in_symvals, out_symvals, unpack_unary_output=False):
        in_code = ", ".join(self.pprint_shape(t) for t in in_symvals)
        in_code = f"({in_code})" if len(in_symvals) > 1 else in_code
        out_code = ", ".join(self.pprint_shape(t) for t in out_symvals)
        out_code = f"({out_code})" if len(out_symvals) > 1 or unpack_unary_output else out_code
        typing_code = f"{in_code} -> {out_code}"
        return typing_code

    def __repr__(self):
        fn_defs = self.instructions_as_code(self, dict())
        return "\n".join(line for code_lines in fn_defs.values() for line in code_lines)

    def save(self, *args, dir_path="/tmp/slope_program", dry_run=False):
        os.makedirs(dir_path, exist_ok=True)
        head_code_lines = [f"import slope # backend={backend.__class__.__name__}"]
        fn_defs = self.instructions_as_code(self, dict())
        in_binders_vars = [self.env[i] for i in self.in_binders]
        for i in range(len(self.in_binders)):
            ibv = in_binders_vars[i]
            if ibv.is_const:
                const_filename = f"{ibv.name}.safetensors"
                const_path = os.path.join(dir_path, f"{const_filename}")
                if not dry_run:
                    backend.save(args[i], const_path)
                dblog(
                    f"Saved {ibv.name} at {const_path}",
                    enable=backend.LOG_BACKEND,
                )
                head_code_lines += [f"""{ibv.name} = slope.load("./{const_filename}")"""]
        head_code_lines += [""]
        code = "\n".join(head_code_lines + [line for code_lines in fn_defs.values() for line in code_lines])
        dblog(
            f"Contents of {self.name}:\n```\n{code}\n```",
            enable=backend.LOG_BACKEND,
        )
        program_path = os.path.join(dir_path, "main.py")
        if not dry_run:
            with open(program_path, "w") as f:
                f.write(code)
        dblog(
            f"Saved program {self.name} at {program_path}",
            enable=backend.LOG_BACKEND,
        )
        ls_contents = "\n\t".join(os.listdir(dir_path))
        dblog(
            f"Contents of {dir_path}:\n\t{ls_contents}",
            enable=backend.LOG_BACKEND,
        )

    def __hash__(self):
        return hash(self.curr_repr)

    def __eq__(self, other):
        return self is other

    @classmethod
    def instructions_as_code(cls, program, fn_defs):
        def indent(code, indent_amount):
            spaces = " " * (len(code) - len(code.lstrip()))
            spaces += " " * indent_amount
            return "\n".join([spaces + line for line in code.strip().split("\n")])

        in_binders_vars = [program.env[i] for i in program.in_binders]
        body_code_lines = []
        for instruction in program.instructions:
            if len(instruction.out_binders) == 0:
                continue
            params = instruction.params.copy()
            for param_name, param in params.items():
                if isinstance(param, Program):
                    sub_program = param
                    fn_defs = cls.instructions_as_code(sub_program, fn_defs)
                    program_in_vals = ", ".join(f"{program.env[x].name}" for x in instruction.inputs)
                    params[param_name] = f"slope.make_program({sub_program.name}, {program_in_vals})[0]"
                if isinstance(param, DType):
                    params[param_name] = f"slope.{param.name}"
            param_vals = ", ".join(f"{param_name}={param}" for param_name, param in params.items())
            in_vals = ", ".join(f"{program.env[x].name}" for x in instruction.inputs)
            out_vals = ", ".join(f"{program.env[z].name}" for z in instruction.out_binders)
            sig = program.pprint_sig(
                [program.env[x].symval for x in instruction.inputs],
                [program.env[y].symval for y in instruction.out_binders],
            )
            line = f"""{out_vals} = slope.{instruction.op.name}({in_vals}{", " if (param_vals and in_vals) else ""}{param_vals}) # {sig}"""
            body_code_lines += [indent(line, program.indent_amount)]

        fn_args_str = ", ".join([f"{i.name}" for i in in_binders_vars])
        # fn_static_args_str = ", ".join([f"{a}={a_val}" for a, a_val in program.static_args])
        out_vars = [program.env[o] for o in program.outs]
        fn_sig = program.pprint_sig(
            [i.symval for i in in_binders_vars],
            [o.symval for o in out_vars],
        )
        head_code_line = [f"def {program.name}({fn_args_str}): # {fn_sig}"]
        out_str = ", ".join([f"{o.name}" for o in out_vars])
        tail_code_line = [indent(f"return {out_str}", program.indent_amount)]
        code_lines = head_code_line + body_code_lines + tail_code_line + ["\n"]

        fn_defs[program.name] = code_lines
        return fn_defs

    @staticmethod
    def prune_instructions(instructions, outs):
        graph = dict()
        for instruction in instructions:
            parent_nodes, child_nodes = instruction.out_binders, instruction.inputs
            for parent in parent_nodes:
                if parent not in graph:
                    graph[parent] = set()
                for child in child_nodes:
                    graph[parent].add(child)
        visited_from_terminal = set()

        def dfs(node, visited):
            visited.add(node)
            if node in graph:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor, visited)

        for terminal_node in outs:
            dfs(terminal_node, visited_from_terminal)
        unreachable_nodes = set(graph.keys()) - visited_from_terminal

        instructions_to_prune = []
        for instruction in instructions:
            parent_nodes, child_nodes = instruction.out_binders, instruction.inputs
            if any(node in unreachable_nodes for node in parent_nodes) or any(node in unreachable_nodes for node in child_nodes):
                instructions_to_prune += [instruction]
        new_instructions = [inst for inst in instructions if inst not in instructions_to_prune]
        if backend.LOG_PROGRAM:
            LI = len(instructions)
            LNI = len(new_instructions)
            DIFF = LI - LNI
            UN = len(unreachable_nodes)
            dblog(f"Before: {LI}\tAfter: {LNI}\tDiff vs Unreachables: {DIFF} == {UN} = {DIFF==UN}")
        return new_instructions


class ProgramType(NamedTuple):
    in_types: Tuple[SymbolicTensor]
    out_types: Tuple[SymbolicTensor]

    def __repr__(self):
        in_types = ", ".join(symval.str_short() for symval in self.in_types)
        out_types = ", ".join(symval.str_short() for symval in self.out_types)
        return f"({in_types}) -> ({out_types})"


# =================================
#   Tracer and Trace
# =================================


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
    flatten: Callable
    unflatten: Callable


class TreeDef(NamedTuple):
    node_type: NodeType
    node_metadata: Hashable
    child_treedefs: Tuple["TreeDef", ...]

    def __repr__(self):
        ret = self.tree_repr(self)
        return ret

    def tree_repr(self, tree, indent="  ", prefix="", last=True):
        ret = ""

        def _tree_repr(tree, indent, prefix, last):
            nonlocal ret
            if isinstance(tree, TreeDef):
                ret += f'{prefix} {("└─" if last else "├─")} {tree.node_type.name}\n'
                for i, item in enumerate(tree.child_treedefs):
                    new_prefix = prefix + (indent if not last else "   ")
                    new_last = i == len(tree.child_treedefs) - 1
                    _tree_repr(item, indent, new_prefix, new_last)
            else:
                ret += f'{prefix} {("└─" if last else "├─")} {tree}\n'

        _tree_repr(tree, indent="  ", prefix="", last=True)
        return ret

    @property
    def num_leaves(self):
        def get_num_leaves(x):
            if isinstance(x, Leaf):
                return 1
            else:
                return sum(get_num_leaves(sub_x) for sub_x in x.child_treedefs)

        return sum(get_num_leaves(x) for x in self.child_treedefs)


class Leaf:
    def __init__(self, val):
        if hasattr(val, "shape"):
            val = SymbolicTensor.like(val)
        self.val = val

    def __repr__(self):
        ret = self.val.str_short() if isinstance(self.val, SymbolicTensor) else repr(self.val)
        return f"<Leaf: {ret}>"

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        return True  # make TreeDef __eq__ don't care Leaf
        # if isinstance(other, Leaf): # TODO: test above assumption
        #     return self.val == other.val


# =================================
#   jit operator
# =================================


class JitOutput:
    def __init__(self, program: Program, codegen_output: CodegenOutput, fn, code: str, consts: List[Any]):
        super().__init__()
        self.program = program
        self.code = code
        self.codegen_output = codegen_output
        self.fn: Callable = fn
        self.consts = consts

    def __call__(self, *args, **params):
        args, in_tree = tree_flatten(args)
        args = tree_map(lambda a: a.val if isinstance(a, Tensor) else a, args)
        try:
            outs = self.fn(*args, **params)
            if not isinstance(outs, tuple):  # TODO: IREE FunctionInvoker destructure 1-tuple, need to undo
                outs = (outs,)
        except Exception as e:
            dblog(self.code, enable=backend.LOG_JIT)
            raise
        return [backend.tensor(TensorBuffer(o)) for o in outs]


class JitOp(MetaOperator):
    def meta_impl(self, *args, program: Program, **_):
        hashed_program = Hashed(program)
        num_consts = program.num_consts
        consts, args = args[:num_consts], args[num_consts:]
        hashed_consts = tuple(map(Hashed, consts))
        jit_output = backend.jit_program(hashed_program, hashed_consts)
        ret = jit_output(*consts, *args)
        return ret

    def reorg_args(self, args, params):
        return args, params

    def typecheck(self, *in_types, program: Program):
        program_type = typecheck_program(program)
        if not all(t1 == t2 for t1, t2 in zip(program_type.in_types, in_types)):
            ret = "Type mismatch program.in_types vs in_types:\n"
            for i, j in zip(program_type.in_types, in_types):
                ret += f"{i}, {j}, {i == j}"
            raise TypeError(ret)
        return program_type.out_types

    def vmap(self, dim_size, vals_in, dims_in, program: Program):
        program, consts = vmap_program(program, dim_size, tuple(dims_in))
        outs = self(*consts, *vals_in, program=program)
        if not isinstance(outs, tuple):
            outs = (outs,)
        return outs, [0] * len(outs)

    def jvp(self, primals, tangents, *, program):
        new_program, new_consts = jvp_program(program)
        outs = bind(
            self,
            *new_consts,
            *primals,
            *tangents,
            program=new_program,
        )
        n = len(outs) // 2
        primals_out, tangents_out = outs[:n], outs[n:]
        return primals_out, tangents_out

    def T(self, cotangents, *invals, program):
        undef_primals = [isinstance(x, UndefinedPrimal) for x in invals]
        transposed_program, new_consts = transpose_program(program, tuple(undef_primals))

        residuals, _ = partition_list(undef_primals, invals)
        outs = bind(
            self,
            *new_consts,
            *residuals,
            *cotangents,
            program=transposed_program,
        )
        outs = iter(outs)

        return [next(outs) if undef else None for undef in undef_primals]

    def partial_run(self, trace, tracers, *, program):
        in_unknowns = [not t.pval.is_known for t in tracers]
        program1, program2, out_unknowns, num_res = partial_run_program(program, in_unknowns)
        known_tracers, unknown_tracers = partition_list(in_unknowns, tracers)
        known_vals = [t.pval.const for t in known_tracers]
        outs1_res = bind(backend.jit_op, *known_vals, program=program1)
        outs1, res = split_list(outs1_res, len(program1.outs) - num_res)
        res_tracers = [trace.instantiate_const(full_raise(trace, x)) for x in res]
        outs2 = [PartialRunTraceTensor(trace, make_unknown_pval(v.symval), None) for v in program2.outs]
        instruction = InstructionDraft(
            self,
            res_tracers + unknown_tracers,
            dict(program=program2),
            [v.symval for v in program2.outs],
            list_map(weakref.ref, outs2),
        )
        for t in outs2:
            t.draft = instruction

        return merge_lists(out_unknowns, outs1, outs2)

    def partial_run_instruction(self, unks_in, instruction) -> Tuple[Instruction, Instruction, List[bool], List[Var]]:
        program = instruction.params["program"]
        program1, program2, out_unknowns, num_res = partial_run_program(program, unks_in)
        ins1, ins2 = partition_list(unks_in, instruction.inputs)
        out_binders1, out_binders2 = partition_list(out_unknowns, instruction.out_binders)
        res = [Var(v.symval) for v in program2.in_binders[:num_res]]
        instruction1 = Instruction(self, ins1, dict(program=program1), out_binders1 + res)
        instruction2 = Instruction(self, res + ins2, dict(program=program2), out_binders2)
        return instruction1, instruction2, out_unknowns, res


# =================================
#   Compiler
# =================================


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

    def run_op(self, op, tracers, params):
        raise NotImplementedError


class RunTrace(Trace):
    pure = lambda self, x: x

    def run_op(self, op: Operator, args, params):
        if isinstance(op, MetaOperator):
            args, params = op.reorg_args(args, params)
            args, params = op.args_fixer(*args, **params)
            ret = op.meta_impl(*args, **params)
        else:
            fn = self.get_fn(op, *tuple(SymbolicTensor.like(a) for a in args), **params)
            # with Timing(f"RUN {op}"):ret = jit(
            ret = jit(
                fn,
                static_argnames=("params",),
                name=jit.get_jit_name(args, params, op.name),
            )(*args, **params)

        return ret

    @staticmethod
    @lru_cache_verbose()
    def get_fn(op, *symval_args, **params):
        def fn(*args, **params):
            return [op(*args, **params)]

        return fn


class SymbolicRunTrace(Trace):
    # pure = lambda self, x: x
    def pure(self, val: Any) -> SymbolicTensor:
        return val.symval

    def run_op(self, op, tracers, params):
        symvals_in = tree_map(lambda x: x.symval, tracers)
        symvals_out = op.typecheck(*symvals_in, **params)
        return symvals_out


class TraceTensor(Tensor):
    PYTHON_TYPES = {
        bool,
        int,
        float,
    }
    _trace: "Trace"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    symval = property(lambda self: get_symval(self.val))
    dtype = property(lambda self: self.symval.dtype)
    shape = property(lambda self: self.symval.shape)
    device = property(lambda self: self.symval.device)

    @property
    def val(self):
        raise NotImplementedError

    def __str__(self):
        return repr(self)

    def full_lower(self):
        return self

    @property
    def ndim(self):
        return len(self.shape)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.symval)})"


class VMapTraceTensor(TraceTensor):
    def __init__(self, trace, val, vmap_dim):
        self._trace = trace
        self._val = val
        self.vmap_dim = vmap_dim

    @property
    def val(self):
        return self._val

    @property
    def symval(self):
        symval = get_symval(self.val)
        if self.vmap_dim is None:
            return symval
        else:
            shape = list(symval.shape)
            del shape[self.vmap_dim]
            return symval.like(shape=tuple(shape))

    def full_lower(self):
        if self.vmap_dim is None:
            return full_lower(self.val)
        else:
            return self


class VMapTrace(Trace):
    pure = lambda self, val: VMapTraceTensor(self, val, None)

    @property
    def dim_size(self):
        return self.main.global_data

    def run_op(self, op, tracers, params):
        vals_in, bdims_in = unzip2((t.val, t.vmap_dim) for t in tracers)
        val_outs, bdim_outs = op.vmap(self.dim_size, vals_in, bdims_in, **params)
        return [VMapTraceTensor(self, x, bd) for x, bd in list_zip(val_outs, bdim_outs)]

    @staticmethod
    def move_vmap_dim(x, dim_size, src: int, dst: int):
        if src is None:  # unsqueeze and expand
            target_shape = list(x.shape)
            target_shape.insert(dst, dim_size)
            unsqueeze_shape = [1 if d == dst else target_shape[d] for d in range(len(target_shape))]
            x = x.reshape(tuple(unsqueeze_shape))
            x = x.expand(tuple(target_shape))
            return x
        elif src == dst:
            return x
        else:
            perm = [i for i in range(len(x.shape)) if i != src]
            perm.insert(dst, src)
            return x.permute(tuple(perm))


class JVPTraceTensor(TraceTensor):
    def __init__(self, trace, primal, tangent):
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def symval(self):
        return get_symval(self.primal)

    @property
    def val(self):
        return self.primal

    @property
    def dtype(self):
        return self.primal.dtype


class JVPTrace(Trace):
    def pure(self, val):
        if isinstance(val, PartialRunTrace):
            val = val.pval.const
        return JVPTraceTensor(self, val, backend.zeros_like(val))

    def run_op(self, op, tracers, params):
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        primals_out, tangents_out = op.jvp(primals_in, tangents_in, **params)
        return [JVPTraceTensor(self, x, t) for x, t in list_zip(primals_out, tangents_out)]


class ProgramTraceTensor(TraceTensor):
    __slots__ = ["symval"]
    symval: SymbolicTensor

    def __init__(self, trace, symval):
        self._trace = trace
        self.symval = symval


class ProgramTrace(Trace):
    @property
    def builder(self):
        return self.main.global_data

    def new_arg(self, symval) -> ProgramTraceTensor:
        symval = SymbolicTensor.like(symval)
        tracer = self.builder.new_tracer(self, symval)
        self.builder.tracer_to_var[id(tracer)] = Var(symval)

        return tracer

    def pure(self, val: Any) -> ProgramTraceTensor:
        # get_or_make_const_tracer
        tracer = self.builder.const_tracers.get(id(val))
        if tracer is None:
            tracer = self.builder.new_tracer(self, get_symval(val))
            self.builder.add_const(tracer, val)
        # print(self.builder.const_tracers)
        return tracer

    def run_op(self, op, tracers, params):
        symvals_in = tree_map(lambda x: x.symval, tracers)
        symvals_out = op.typecheck(*symvals_in, **params)

        out_tracers = [self.builder.new_tracer(self, a) for a in symvals_out]
        inputs = [self.builder.getvar(t) for t in tracers]
        outvars = [self.builder.add_var(t) for t in out_tracers]

        self.builder.add_instruction(Instruction(op, inputs, params, outvars))
        return out_tracers


class ProgramBuilder:
    instructions: List[Instruction]
    tracer_to_var: Dict[int, Var]
    const_tracers: Dict[int, TraceTensor]
    constvals: Dict[Var, Any]
    tracers: List[ProgramTraceTensor]

    def __init__(self):
        self.instructions = []
        self.tracer_to_var = {}
        self.const_tracers = {}
        self.constvals = {}
        self.tracers = []

    def new_tracer(self, trace: ProgramTrace, symval: SymbolicTensor) -> ProgramTraceTensor:
        tracer = ProgramTraceTensor(trace, symval)
        self.tracers += [tracer]
        return tracer

    def add_instruction(self, instruction: Instruction) -> None:
        self.instructions += [instruction]

    def add_var(self, tracer: ProgramTraceTensor) -> Var:
        assert id(tracer) not in self.tracer_to_var
        var = self.tracer_to_var[id(tracer)] = Var(tracer.symval)
        return var

    def getvar(self, tracer: ProgramTraceTensor) -> Var:
        var = self.tracer_to_var.get(id(tracer))
        assert var is not None
        return var

    def add_const(self, tracer: ProgramTraceTensor, val: Any) -> Var:
        var = self.add_var(tracer)
        self.const_tracers[id(val)] = tracer
        self.constvals[var] = val
        return var

    def build(self, in_tracers: Any, out_tracers: Any, static_args, name) -> Tuple[Program, List[Any]]:
        constvars, constvals = unzip2(self.constvals.items())
        t2v = lambda t: self.tracer_to_var[id(t)]
        in_binders = constvars + [t2v(t) for t in in_tracers]
        out_vars = [t2v(t) for t in out_tracers]
        program = Program(
            in_binders,
            self.instructions,
            out_vars,
            len(constvals),
            static_args,
            name,
        )
        typecheck_program(program)
        program, constvals = self._inline_literals(program, constvals)
        typecheck_program(program)
        # dblog(program, enable=backend.LOG_PROGRAM)
        return program, constvals

    def _inline_literals(self, program: Program, consts: List[Any]) -> Tuple[Program, List[Any]]:
        const_binders, other_binders = split_list(program.in_binders, len(consts))
        scalars = [type(x) in TraceTensor.PYTHON_TYPES and not get_symval(x).shape for x in consts]
        new_const_binders, lit_binders = partition_list(scalars, const_binders)
        new_consts, lit_vals = partition_list(scalars, consts)
        literals = dict(list_zip(lit_binders, list_map(Lit, lit_vals)))
        new_outs = [literals.get(x, x) for x in program.outs]
        new_instructions = [
            Instruction(
                instruction.op,
                [literals.get(x, x) for x in instruction.inputs],
                instruction.params,
                instruction.out_binders,
            )
            for instruction in program.instructions
        ]
        new_program = Program(
            new_const_binders + other_binders,
            new_instructions,
            new_outs,
            len(new_consts),
            program.static_args,
            program.name,
        )
        return new_program, tuple(new_consts)

    def get_current_scope_info(self):
        current_frame = inspect.currentframe()
        current_function_name = current_frame.f_code.co_name
        current_module_name = inspect.getmodulename(current_frame.f_code.co_filename)
        current_class_name = None
        for frame_info in inspect.getouterframes(current_frame):
            print(frame_info)
            frame_locals = frame_info.frame.f_locals
            print(frame_locals)
            if "self" in frame_locals:
                current_class_name = frame_locals["self"].__class__.__name__
                break
        return {
            "Function": current_function_name,
            "Module": current_module_name,
            "Class": current_class_name,
        }


class UndefinedPrimal(NamedTuple):
    symval: SymbolicTensor

    @property
    def shape(self):
        return self.symval.shape

    @property
    def dtype(self):
        return self.symval.dtype

    @property
    def device(self):
        return self.symval.device

    @property
    def ndim(self):
        return self.symval.ndim

    def __repr__(self):
        return f"<UndefinedPrimal: symval={self.symval}>"

    str_short = __repr__


class PartialValue(NamedTuple):
    symval: SymbolicTensor
    const: Optional[Any]

    is_known = property(lambda self: self.const is not None)
    is_unknown = property(lambda self: self.const is None)


class LambdaBindingDraft(NamedTuple):
    pass


class ConstDraft(NamedTuple):
    val: Any


class InstructionDraft(NamedTuple):
    prim: Operator
    tracers_in: List["PartialRunTraceTensor"]
    params: Dict[str, Any]
    symvals_out: List[SymbolicTensor]
    tracer_refs_out: List[weakref.ReferenceType["PartialRunTraceTensor"]]


ProgramDraft = Union[LambdaBindingDraft, ConstDraft, InstructionDraft]


class PartialRunTraceTensor(TraceTensor):
    def __init__(self, trace, pval, draft):
        self._trace = trace
        self.pval = pval
        self.draft = draft

    symval = property(lambda self: self.pval.symval)
    val = property(lambda self: self.pval.const)

    def full_lower(self):
        if self.pval.is_known:
            return full_lower(self.pval.const)
        return self


class PartialRunTrace(Trace):
    def new_arg(self, pval: PartialValue) -> Any:
        return PartialRunTraceTensor(self, pval, LambdaBindingDraft())

    def pure(self, val: Any) -> PartialRunTraceTensor:
        return PartialRunTraceTensor(self, make_known_pval(val), None)

    def instantiate_const(self, tracer: PartialRunTraceTensor) -> PartialRunTraceTensor:
        if tracer.pval.is_unknown:
            return tracer
        else:
            pval = make_unknown_pval(SymbolicTensor.like(tracer.symval))
            return PartialRunTraceTensor(self, pval, ConstDraft(tracer.pval.const))

    def run_op(self, op, tracers, params):
        is_knowns = tuple(t.pval.is_known for t in tracers)

        if all(is_knowns):
            return bind(op, *list_map(full_lower, tracers), **params)
        return op.partial_run(self, tracers, **params)


trace_stack: List[MainTrace] = []
stashed_trace: Optional[MainTrace] = None
trace_stack += [MainTrace(0, RunTrace, None)]


class UndefBackend:
    def __getattr__(self, attr):
        raise NotImplementedError("Backend not init yet with slope.core.set_backend(backend)")


backend = UndefBackend()


def set_backend(name, where="slope.backends"):
    global backend
    backend = importlib.import_module(f"{where}.{name}").backend
    import slope.nn as nn

    # backend.register_node(nn.Module, nn.Module.flatten, nn.Module.unflatten, "Module")

    dblog(f"slope backend is {backend}", enable=backend.LOG_INIT)


def stack_str():
    ret = ""
    for trace in trace_stack:
        ret += f"{trace.level}: {trace.trace_type.__name__}\t{trace.global_data=}\n"
    return ret


def make_known_pval(val: Any):
    return PartialValue(get_symval(val), val)


def make_unknown_pval(symval: SymbolicTensor):
    return PartialValue(symval, None)


def get_symval(x):
    if isinstance(x, TraceTensor):
        return x.symval
    elif type(x) in TraceTensor.PYTHON_TYPES:
        return backend.tensor(x)
    elif isinstance(x, Tensor):
        return x
    elif isinstance(x, SymbolicTensor):
        return x
    else:
        raise TypeError(type(x))


def tree_flatten(x: Any) -> Any:
    def _tree_flatten(x_: Any) -> Tuple[Iterable, Union[TreeDef, Leaf]]:
        node_type = None
        for k in backend.node_types.keys():
            if isinstance(x_, k):
                node_type = backend.node_types[k]

        if node_type is not None:
            node_metadata, children = node_type.flatten(x_)
            children_flat, child_trees = unzip2(list_map(_tree_flatten, children))
            children_iter = itertools.chain.from_iterable(children_flat)
            treedef = TreeDef(node_type, node_metadata, tuple(child_trees))
            return children_iter, treedef
        else:
            return (x_,), Leaf(x_)

    children_iter, treedef = _tree_flatten(x)
    return tuple(children_iter), treedef


def tree_unflatten(treedef: TreeDef, xs: Tuple[Any]) -> Any:
    def _tree_unflatten(treedef_: TreeDef, xs_: Iterator) -> Any:
        if isinstance(treedef_, Leaf):
            dblog(f"    tree leaf found: {xs_}\n", enable=backend.LOG_TREE)
            return next(xs_)
        else:
            dblog(f"    now\n  {treedef_}", enable=backend.LOG_TREE)
            children = (_tree_unflatten(t, xs_) for t in treedef_.child_treedefs)
            dblog(f"{children=}\n", enable=backend.LOG_TREE)
            return treedef_.node_type.unflatten(treedef_.node_metadata, children)

    dblog(f"unflattening {treedef}", enable=backend.LOG_TREE)
    return _tree_unflatten(treedef, iter(xs))
    # with Timing(f"\nTREE:\n{treedef}"):
    #     ret = _tree_unflatten(treedef, iter(xs))
    # return ret


def tree_transpose(
    outer_treedef: TreeDef,
    inner_treedef: TreeDef,
    tree_to_transpose: Any,
) -> Any:
    flat, treedef = tree_flatten(tree_to_transpose)
    inner_size = inner_treedef.num_leaves
    outer_size = outer_treedef.num_leaves
    if treedef.num_leaves != (inner_size * outer_size):
        raise TypeError
    iter_flat = iter(flat)
    lol = [[next(iter_flat) for _ in range(inner_size)] for __ in range(outer_size)]
    permuted_lol = zip(*lol)
    subtrees = map(partial(tree_unflatten, outer_treedef), permuted_lol)
    return tree_unflatten(inner_treedef, subtrees)


def flatten_fn(f, in_tree, *, has_aux=False):
    store = Store()

    def flat_fn(*args_flat, **params):
        tree_args = tree_unflatten(in_tree, args_flat)
        out = f(*tree_args, **params)
        if has_aux:
            out, aux = out
        out_flat, out_tree = tree_flatten(out)
        store.set_value(out_tree)
        return (out_flat, aux) if has_aux else out_flat

    return flat_fn, store


def tree_map(f: Callable[..., Any], tree, *rest, out_leaf=False) -> Any:
    leaves, treedef = tree_flatten(tree)
    if len(rest) == 0:
        out_tree_flat = tuple(f(leaf) for leaf in leaves)
        out_tree = tree_unflatten(treedef, out_tree_flat)
    else:
        all_leaves = [leaves]
        for t in rest:
            t_leaves, t_treedef = tree_flatten(t)
            assert t_treedef == treedef
            all_leaves += [t_leaves]

        out_tree_flat = tuple(f(*xs) for xs in zip(*all_leaves))
        out_tree = tree_unflatten(treedef, out_tree_flat)
    ret = out_tree
    if out_leaf:
        ret = (ret, tree_flatten(out_tree_flat[0]))
    return ret


@contextmanager
def new_main_trace(trace_type: Type["Trace"], global_data=None):
    global trace_stack
    level = len(trace_stack)
    main = MainTrace(level, trace_type, global_data)
    trace_stack += [main]

    try:
        yield main
    finally:
        trace_stack.pop()


def bind(op, *args, **params):
    top_trace = find_top_trace(args)
    tracers = tuple([full_raise(top_trace, arg) for arg in args])
    outs = top_trace.run_op(op, tracers, params)
    lowered = tuple([full_lower(out) for out in outs])
    return lowered


def find_top_trace(xs) -> Trace:
    arrs = []

    def get_arr_from_seq(seq):
        nonlocal arrs
        for x in seq:
            if type(x) in (tuple, list):
                get_arr_from_seq(x)
            elif isinstance(x, TraceTensor):
                arrs += [x]

    get_arr_from_seq(xs)
    arrs = tuple(arrs)
    top_main = max(
        (x._trace.main for x in arrs),
        default=trace_stack[0],
        key=operator_py.attrgetter("level"),
    )
    if stashed_trace and stashed_trace.level > top_main.level:
        top_main = stashed_trace
    return top_main.trace_type(top_main)


def full_raise(trace: Trace, val: Any) -> TraceTensor:
    if not isinstance(val, TraceTensor):
        return trace.pure(val)
    level = trace.main.level
    if val._trace.main is trace.main:
        return val
    elif val._trace.main.level < level:
        return trace.pure(val)
    elif val._trace.main.level > level:
        raise Exception(f"Can't lift level {val._trace.main.level} to {level}.")
    else:
        raise Exception(f"Different traces at same level: {val._trace}, {trace}.")


def full_lower(val: Any):
    if isinstance(val, TraceTensor):
        return val.full_lower()
    elif type(val) in (list, tuple):
        return tuple(full_lower(v) for v in val)
    else:
        return val


def typecheck_program(program: Program) -> ProgramType:
    env: Set[Var] = set()

    for v in program.in_binders:
        if v in env:
            raise TypeError
        env.add(v)

    for instruction in program.instructions:
        in_types = [typecheck_atom(env, x) for x in instruction.inputs]
        out_types = instruction.op.typecheck(*in_types, **instruction.params)
        for out_binder, out_type in list_zip(instruction.out_binders, out_types):
            if not out_type == out_binder.symval:
                raise TypeError
        for out_binder in instruction.out_binders:
            if out_binder in env:
                raise TypeError
            env.add(out_binder)

    in_types = [v.symval for v in program.in_binders]
    out_types = [typecheck_atom(env, x) for x in program.outs]
    return ProgramType(tuple(in_types), tuple(out_types))


def typecheck_atom(env: Set[Var], x: Atom) -> SymbolicTensor:
    if isinstance(x, Var):
        if x not in env:
            raise TypeError("unbound variable")
        return x.symval
    elif isinstance(x, Lit):
        return get_symval(x.val)
    else:
        assert False


def run_program(program: Program, args: List[Any]) -> List[Any]:
    env: Dict[Var, Any] = {}

    def read(x: Atom) -> Any:
        return env[x] if type(x) is Var else x.val

    def write(v: Var, val: Any) -> None:
        assert v not in env  # single-assignment
        env[v] = val

    list_map(write, program.in_binders, args)
    for instruction in program.instructions:
        in_vals = list_map(read, instruction.inputs)
        outs = bind(instruction.op, *in_vals, **instruction.params)
        list_map(write, instruction.out_binders, outs)
    return list_map(read, program.outs)


def program_as_fun(program: Program):
    return lambda *args: run_program(program, args)


def vmap_flat(f, in_dim, out_dim, dim_size, *args):
    if dim_size is None:
        dims = set([x.shape[d] for x, d in list_zip(args, in_dim) if d is not None])
        assert len(dims) == 1
        (dim_size,) = dims
    with new_main_trace(VMapTrace, dim_size) as main:
        trace = VMapTrace(main)
        tracers_in = [VMapTraceTensor(trace, x, dim) if dim is not None else x for x, dim in list_zip(args, in_dim)]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
        vals_out, y_vmap_dims = unzip2((t.val, t.vmap_dim) for t in tracers_out)
    ret = [VMapTrace.move_vmap_dim(val_out, dim_size, bdim, out_dim) for val_out, bdim, out_dim in zip(vals_out, y_vmap_dims, out_dim)]
    return ret


def vmap(f, in_dim=0, out_dim=0, dim_size=None):
    def batched_f(*args):
        nonlocal in_dim, out_dim, dim_size
        args_flat, in_tree = tree_flatten(args)
        in_dim = (in_dim,) * len(args) if isinstance(in_dim, int) else in_dim
        out_dim = (out_dim,) * len(args) if isinstance(out_dim, int) else out_dim
        in_dim_flat, in_dim_tree = tree_flatten(in_dim)
        out_dim_flat, out_dim_tree = tree_flatten(out_dim)
        if not (in_tree == in_dim_tree == out_dim_tree):
            raise TypeError(f"\n{in_tree}\n!=\n{in_dim_tree}!=\n{out_dim_tree}")
        f_flat, out_tree_store = flatten_fn(f, in_tree)
        # if len(args_flat) > len(in_dim_flat):
        #     in_dim_flat = (in_dim[0],) * len(args_flat)
        outs_flat = vmap_flat(f_flat, in_dim_flat, out_dim_flat, dim_size, *args_flat)
        return tree_unflatten(out_tree_store(), outs_flat)

    return batched_f


def jvp_flat(f, primals, tangents, *, has_aux, global_data, **static_args):
    with new_main_trace(JVPTrace, global_data) as main:
        trace = JVPTrace(main)
        tracers_in = [JVPTraceTensor(trace, x, t) for x, t in list_zip(primals, tangents)]
        jvp_flat_ret = f(*tracers_in, **static_args)
        if has_aux:
            (outs, aux) = jvp_flat_ret
            # aux_ = aux
            aux = tree_map(lambda x: x.primal, aux)
            # aux = tree_map(lambda x: x.full_lower(), aux)
            #
        else:
            outs = jvp_flat_ret
        tracers_out = [full_raise(trace, out) for out in outs]
        primals_out, tangents_out = unzip2((t.primal, t.tangent) for t in tracers_out)
    return ((primals_out, tangents_out), aux) if has_aux else (primals_out, tangents_out)


def jvp(f, primals, tangents, *, has_aux=False, global_data=None, **static_args):
    primals_flat, in_tree = tree_flatten(primals)
    tangents_flat, in_tree2 = tree_flatten(tangents)
    for p, t in zip(primals_flat, tangents_flat):
        assert p.shape == t.shape, f"{p.shape=} != {t.shape=}"
        assert p.dtype == t.dtype, f"{p.dtype=} != {t.dtype=}"
        assert p.device == t.device, f"{p.device=} != {t.device=}"
    if in_tree != in_tree2:
        raise TypeError
    f, out_tree_store = flatten_fn(f, in_tree, has_aux=has_aux)
    jvp_ret = jvp_flat(
        f,
        primals_flat,
        tangents_flat,
        has_aux=has_aux,
        global_data=global_data,
        **static_args,
    )
    if has_aux:
        (primals_out_flat, tangents_out_flat), aux = jvp_ret
    else:
        (primals_out_flat, tangents_out_flat) = jvp_ret
    primals_out = tree_unflatten(out_tree_store(), primals_out_flat)
    tangents_out = tree_unflatten(out_tree_store(), tangents_out_flat)
    return ((primals_out, tangents_out), aux) if has_aux else (primals_out, tangents_out)


def jacfwd(f, x):
    pushfwd = lambda v: jvp(f, (x,), (v,))[1]
    vecs_in = backend.eye(math.prod(x.shape)).reshape(x.shape * 2)
    return vmap(pushfwd, (0,))(vecs_in)


@contextmanager
def stash_trace(main: MainTrace):
    global stashed_trace
    prev_stashed_trace, stashed_trace = stashed_trace, main
    try:
        yield
    finally:
        stashed_trace = prev_stashed_trace


@contextmanager
def symbolic_run():
    global trace_stack
    level = len(trace_stack)
    main = MainTrace(level, SymbolicRunTrace, global_data=None)
    trace_stack += [main]
    global stashed_trace
    prev_stashed_trace, stashed_trace = stashed_trace, main
    try:
        yield
    finally:
        stashed_trace = prev_stashed_trace
        trace_stack.pop()


@lru_cache_verbose()
def make_program(f: Callable, *symvals_in: SymbolicTensor, static_args, name) -> Tuple[Program, List[Any], TreeDef]:
    symvals_in, in_tree = tree_flatten(symvals_in)
    f, out_tree_store = flatten_fn(f, in_tree)
    builder = ProgramBuilder()
    with new_main_trace(ProgramTrace, builder) as main:
        with stash_trace(main):
            trace = ProgramTrace(main)
            tracers_in = [trace.new_arg(symval) for symval in symvals_in]
            outs = f(*tracers_in, **{k: v for k, v in static_args})
            tracers_out = [full_raise(trace, out) if isinstance(out, ProgramTraceTensor) else out.val for out in outs]
            program, consts = builder.build(tracers_in, tracers_out, static_args, name)

    return program, consts, out_tree_store()


@lru_cache_verbose()
def vmap_program(program: Program, dim_size, dims_in) -> tuple[Program, list[Any]]:
    def unmapped_symval(axis_size: int, batch_dim, symval: SymbolicTensor) -> SymbolicTensor:
        if batch_dim is None:
            return symval
        else:
            shape = list(symval.shape)
            shape.insert(batch_dim, axis_size)
            return symval.like(shape=tuple(shape))

    vmap_traceable = vmap(program_as_fun(program), tuple(dims_in))
    in_symvals = [unmapped_symval(dim_size, d, v.symval) for v, d in zip(program.in_binders, dims_in)]
    program, consts, _ = make_program(
        vmap_traceable,
        *in_symvals,
        static_args=program.static_args,
        name=f"vmap_{program.name}",
    )
    return program, consts


@lru_cache_verbose()
def jvp_program(program: Program) -> Tuple[Program, List[Any]]:
    def jvp_traceable(*primals_and_tangents):
        n = len(primals_and_tangents) // 2
        primals, tangents = primals_and_tangents[:n], primals_and_tangents[n:]
        return jvp(program_as_fun(program), primals, tangents)

    in_symvals = tree_map(lambda v: v.symval, program.in_binders)
    new_program, new_consts, _ = make_program(
        jvp_traceable,
        *in_symvals,
        *in_symvals,
        static_args=program.static_args,
        name=f"{program.name}_jvp",
    )
    return new_program, new_consts


def partial_run_flat(
    f: Callable, pvals_in: List["PartialValue"], has_aux, global_data=None
) -> Tuple[Program, List["PartialValue"], List[Any]]:
    with new_main_trace(PartialRunTrace, global_data) as main:
        trace = PartialRunTrace(main)
        tracers_in = [trace.new_arg(pval) for pval in pvals_in]
        outs = f(*tracers_in)
        if has_aux:
            outs, aux = outs
        tracers_out = [full_raise(trace, out) for out in outs]
        pvals_out = [t.pval for t in tracers_out]
        unk_tracers_in = [t for t in tracers_in if t.pval.is_unknown]
        unk_tracers_out = [t for t in tracers_out if t.pval.is_unknown]
        program, consts = tracers_to_program(unk_tracers_in, unk_tracers_out)

    return (program, pvals_out, consts, aux) if has_aux else (program, pvals_out, consts)


def partial_run_program(
    program: Program,
    in_unknowns: List[bool],
    instantiate: Optional[List[bool]] = None,
) -> Tuple[Program, Program, List[bool], int]:
    env: Dict[Var, bool] = {}
    residuals: Set[Var] = set()

    def read(x: Atom) -> bool:
        return type(x) is Var and env[x]

    def write(unk: bool, v: Var) -> None:
        env[v] = unk

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
            instructions1 += [instruction1]
        if instruction2 is not None:
            instructions2 += [instruction2]
        if res is not None:
            residuals.update(res)
        list_map(write, unks_out, instruction.out_binders)

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

    program1 = Program(
        ins1,
        instructions1,
        outs1 + residuals,
        0,
        program.static_args,
        f"{program.name}_partial1",
    )
    program2 = Program(
        residuals + ins2,
        instructions2,
        outs2,
        0,
        program.static_args,
        f"{program.name}_partial2",
    )
    typecheck_partial_run_program(program, in_unknowns, out_unknowns, program1, program2)

    return program1, program2, out_unknowns, num_res


def typecheck_partial_run_program(program, in_unknowns, out_unknowns, program1, program2):
    programty = typecheck_program(program)  # (a1,  a2) -> (b1, b2 )
    program1ty = typecheck_program(program1)  #  a1       -> (b1, res)
    program2ty = typecheck_program(program2)  # (res, a2) -> b2

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


def linearize_flat(f, *primals_in, has_aux):
    pvals_in = [make_known_pval(x) for x in primals_in] + [make_unknown_pval(SymbolicTensor.like(get_symval(x))) for x in primals_in]

    def f_jvp(*primals_tangents_in):
        jvp_ret = jvp(f, *split_half(primals_tangents_in), has_aux=has_aux)
        if has_aux:
            (primals_out, tangents_out), aux = jvp_ret
            return ((*primals_out, *tangents_out), aux)
        else:
            primals_out, tangents_out = jvp_ret
            return (*primals_out, *tangents_out)

    partial_run_flat_ret = partial_run_flat(f_jvp, pvals_in, has_aux)
    if has_aux:
        program, pvals_out, consts, aux = partial_run_flat_ret
    else:
        program, pvals_out, consts = partial_run_flat_ret
    primal_pvals, _ = split_half(pvals_out)
    assert all(pval.is_known for pval in primal_pvals)
    primals_out = [pval.const for pval in primal_pvals]
    f_lin = lambda *tangents: run_program(program, [*consts, *tangents])
    return (primals_out, f_lin, aux) if has_aux else (primals_out, f_lin)


def linearize(f, *primals_in, has_aux=False):
    primals_in_flat, in_tree = tree_flatten(primals_in)
    f, out_tree_store = flatten_fn(f, in_tree, has_aux=has_aux)
    linearize_flat_ret = linearize_flat(f, *primals_in_flat, has_aux=has_aux)
    if has_aux:
        primals_out_flat, f_lin_flat, aux = linearize_flat_ret
    else:
        primals_out_flat, f_lin_flat = linearize_flat_ret

    primals_out = tree_unflatten(out_tree_store(), primals_out_flat)

    def f_lin(*tangents_in):
        tangents_in_flat, in_tree2 = tree_flatten(tangents_in)
        if in_tree != in_tree2:
            raise TypeError
        tangents_out_flat = f_lin_flat(*tangents_in_flat)
        return tree_unflatten(out_tree_store(), tangents_out_flat)

    return (primals_out, f_lin, aux) if has_aux else (primals_out, f_lin)


def tracers_to_program(
    tracers_in: List["PartialRunTraceTensor"],
    tracers_out: List["PartialRunTraceTensor"],
):
    def tracer_parents(t: PartialRunTraceTensor) -> List[PartialRunTraceTensor]:
        return t.draft.tracers_in if isinstance(t.draft, InstructionDraft) else []

    def draft_to_instruction(tracer_to_var: Dict[int, Var], draft: InstructionDraft) -> Instruction:
        inputs = [tracer_to_var[id(t)] for t in draft.tracers_in]
        out_binders = [Var(symval) for symval in draft.symvals_out]
        for t_ref, var in list_zip(draft.tracer_refs_out, out_binders):
            if t_ref() is not None:
                tracer_to_var[id(t_ref())] = var
        return Instruction(draft.prim, inputs, draft.params, out_binders)

    tracer_to_var: Dict[int, Var] = {id(t): Var(SymbolicTensor.like(t.symval)) for t in tracers_in}
    constvar_to_val: Dict[int, Any] = {}
    constid_to_var: Dict[int, Var] = {}
    processed_instructions: Set[int] = set()
    instructions: List[Instruction] = []
    for t in toposort(tracers_out, tracer_parents):
        if isinstance(t.draft, LambdaBindingDraft):
            assert id(t) in set(list_map(id, tracers_in))
        elif isinstance(t.draft, ConstDraft):
            val = t.draft.val
            var = constid_to_var.get(id(val))
            if var is None:
                symval = SymbolicTensor.like(get_symval(val))
                var = constid_to_var[id(val)] = Var(symval)
                constvar_to_val[var] = val
            tracer_to_var[id(t)] = var
        elif isinstance(t.draft, InstructionDraft):
            if id(t.draft) not in processed_instructions:
                instructions += [draft_to_instruction(tracer_to_var, t.draft)]
                processed_instructions.add(id(t.draft))
        else:
            raise TypeError(t.draft)

    constvars, constvals = unzip2(constvar_to_val.items())
    in_binders = constvars + [tracer_to_var[id(t)] for t in tracers_in]
    # out_vars = [tracer_to_var[id(t)] for t in tracers_out if id(t) in tracer_to_var]
    out_vars = [tracer_to_var[id(t)] for t in tracers_out]
    program = Program(tuple(in_binders), tuple(instructions), tuple(out_vars))
    typecheck_program(program)
    return program, constvals


def toposort(out_nodes: List[Any], parents: Callable[[Any], List[Any]]):
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
        sorted_nodes += [node]
        for parent in parents(node):
            if child_counts[id(parent)] == 1:
                childless_nodes += [parent]
            else:
                child_counts[id(parent)] -= 1

    sorted_nodes = sorted_nodes[::-1]
    check_toposort(sorted_nodes, parents)
    return sorted_nodes


def vjp_flat(f, *primals_in, has_aux=False, **static_args):
    pvals_in = [make_known_pval(x) for x in primals_in] + [make_unknown_pval(SymbolicTensor.like(get_symval(x))) for x in primals_in]
    primal_pvals_in, tangent_pvals_in = split_half(pvals_in)
    del primal_pvals_in

    def f_jvp(*primals_tangents_in):
        jvp_ret = jvp(
            f,
            *split_half(primals_tangents_in),
            has_aux=has_aux,
            global_data="vjp",
            **static_args,
        )
        if has_aux:
            ((primals_out, tangents_out), aux) = jvp_ret
        else:
            (primals_out, tangents_out) = jvp_ret
        return ([*primals_out, *tangents_out], aux) if has_aux else [*primals_out, *tangents_out]

    partial_run_flat_ret = partial_run_flat(f_jvp, pvals_in, has_aux, "vjp")
    if has_aux:
        program, pvals_out, consts, aux = partial_run_flat_ret
    else:
        program, pvals_out, consts = partial_run_flat_ret

    primal_pvals, tangent_pvals = split_half(pvals_out)
    del tangent_pvals
    assert all(pval.is_known for pval in primal_pvals)
    primals_out_flat = [pval.const for pval in primal_pvals]
    transpose_inputs = consts + [UndefinedPrimal(t.symval) for t in tangent_pvals_in]

    def f_vjp_flat(*cotangents):
        # return backward_pass(program, transpose_inputs, cotangents)
        undef_primals = tuple(isinstance(x, UndefinedPrimal) for x in transpose_inputs)
        transposed_program, new_consts = transpose_program(program, undef_primals)
        residuals, _ = partition_list(undef_primals, transpose_inputs)
        outs = run_program(transposed_program, (*new_consts, *residuals, *cotangents))
        return outs

    return (primals_out_flat, f_vjp_flat, aux) if has_aux else (primals_out_flat, f_vjp_flat)


def vjp(f, *primals_in, has_aux=False, **static_args):
    primals_in_flat, in_tree = tree_flatten(primals_in)
    f, out_tree_store = flatten_fn(f, in_tree, has_aux=has_aux)
    vjp_ret = vjp_flat(f, *primals_in_flat, has_aux=has_aux, **static_args)
    if has_aux:
        primals_out_flat, f_vjp_flat, aux = vjp_ret
    else:
        primals_out_flat, f_vjp_flat = vjp_ret
    primals_out = tree_unflatten(out_tree_store(), primals_out_flat)

    def f_vjp(*cotangents_out):
        cotangents_out_flat, _ = tree_flatten(cotangents_out)
        cotangents_in_flat = f_vjp_flat(*cotangents_out_flat)

        return tree_unflatten(in_tree, cotangents_in_flat)

    return (primals_out, f_vjp, aux) if has_aux else (primals_out, f_vjp)


NullCotangent = None


def backward_pass(program: Program, args: List[Any], cotangents: List[Any]) -> List[Any]:
    primal_env: Dict[Var, Any] = {}
    ct_env: Dict[Var, Any] = {}

    def read_primal(x: Atom) -> Any:
        return primal_env.get(x, UndefinedPrimal(x.symval)) if type(x) is Var else x.val

    def write_primal(v: Var, val: Any) -> None:
        if type(val) is not UndefinedPrimal:
            primal_env[v] = val

    def read_cotangent(v: Var) -> Any:
        return ct_env.pop(v, backend.zeros(v.symval.shape, v.symval.dtype))

    def write_cotangent(x: Atom, ct: Any):
        if type(x) is Var and ct is not NullCotangent:
            ct_env[x] = (ct_env[x] + ct) if x in ct_env else ct

    list_map(write_primal, program.in_binders, args)
    list_map(write_cotangent, program.outs, cotangents)
    for instruction in program.instructions[::-1]:
        primals_in = list_map(read_primal, instruction.inputs)
        cotangents_in = list_map(read_cotangent, instruction.out_binders)
        inp, params = primals_in, instruction.params
        cotangents_out = instruction.op.T(cotangents_in, *inp, **params)
        list_map(write_cotangent, instruction.inputs, cotangents_out)

    ret = [read_cotangent(v) for v, x in list_zip(program.in_binders, args) if isinstance(x, UndefinedPrimal)]
    return ret


@lru_cache_verbose()
def transpose_program(program: Program, undef_primals: tuple[bool, ...]) -> tuple[Program, list[Any]]:
    symvals_in, symvals_out = typecheck_program(program)
    traceable = partial(backward_pass, program)
    ()
    args = [UndefinedPrimal(a) if u else a for a, u in zip(symvals_in, undef_primals)]
    trans_program, consts, _ = make_program(
        traceable,
        tuple(args),
        tuple(symvals_out),
        static_args=program.static_args,
        name=f"{program.name}_T",
    )
    typecheck_program(trans_program)

    return trans_program, consts


def grad(f, argnums=(0,), argnames="", has_aux=False, return_value=False):
    f, rejit = (f, False) if not isinstance(f, jit) else (f.f, True)
    if isinstance(argnums, int):
        argnums = (argnums,)

    def gfn(x, *xs, **static_args):
        vjp_ret = vjp(f, x, *xs, has_aux=has_aux, **static_args)
        if has_aux:
            y, f_vjp, aux = vjp_ret
        else:
            y, f_vjp = vjp_ret
        if np.shape(y) != ():
            raise TypeError("grad output must be 0-dim scalar with shape ()")
        gL_xs = f_vjp(backend.ones(()))
        gL_xs = tuple(gL_xs[i] for i in argnums) if len(argnums) > 1 else gL_xs[argnums[0]]
        if return_value:
            return ((y, aux), gL_xs) if has_aux else (y, gL_xs)
        else:
            return (gL_xs, aux) if has_aux else gL_xs

    return jit(gfn) if rejit else gfn


def value_and_grad(f, argnums=(0,), argnames="", has_aux=False):
    return grad(
        f,
        argnums=argnums,
        argnames=argnames,
        has_aux=has_aux,
        return_value=True,
    )


def jit_partial_run(trace, tracers, *, program):
    in_unknowns = [not t.pval.is_known for t in tracers]
    program1, program2, out_unknowns, num_res = partial_run_program(program, in_unknowns)
    known_tracers, unknown_tracers = partition_list(in_unknowns, tracers)
    known_vals = [t.pval.const for t in known_tracers]
    outs1_res = backend.jit_op(*known_vals, program=program)
    outs1, res = split_list(outs1_res, len(program1.outs) - num_res)
    res_tracers = [trace.instantiate_const(full_raise(trace, x)) for x in res]
    outs2 = [PartialRunTraceTensor(trace, PartialValue.unknown(v.symval), None) for v in program2.outs]
    draft = InstructionDraft(
        backend.jit_op,
        res_tracers + unknown_tracers,
        dict(program=program2),
        [v.symval for v in program2.outs],
        map(weakref.ref, outs2),
    )
    for t in outs2:
        t.draft = draft
    return merge_lists(out_unknowns, outs1, outs2)


class jit:
    def __init__(self, f, static_argnames=(), name=None, dynamic_axes=None):
        if isinstance(static_argnames, str):
            static_argnames = tuple(static_argnames.split(" "))
        assert type(static_argnames) is tuple and all(type(s) is str for s in static_argnames)
        self.f = f
        self.name = name if name is not None else self.f.__name__
        self.static_argnames = static_argnames
        self.dynamic_axes = dynamic_axes

    @classmethod
    def with_options(cls, **kwargs):
        return partial(cls, **kwargs)

    @classmethod
    def get_jit_name(cls, args, static_args, prefix="jit", short=False):
        name = f"{prefix}_"
        if short:
            static_args_tup = tuple(static_args.items())
            ids = repr(hash((prefix, args, static_args_tup)))[-4:]
            name = f"{prefix}_{ids}"
        else:
            for a in args:
                name += f"shape_{a.shape}_dtype_{a.dtype.name}_"
            for k, v in static_args.items():
                name += f"{k}_{v}_"
            name = name.replace("(", "L")
            name = name.replace(")", "R")
            name = name.replace(",", "C")
            name = name.replace(" ", "")
            name = name.replace(".", "D")

        return name

    def get_program(self, *args, **static_args):
        sig = inspect.signature(self.f)
        if all("*" not in repr(v) for v in sig.parameters.values()):
            args_strs = [k for k, v in sig.parameters.items() if k != "self" and k not in self.static_argnames]
            static_args_strs = [k for k, v in sig.parameters.items() if k != "self" and k in self.static_argnames]

            if args:
                if len(args) > len(args_strs):
                    assert static_args_strs
                    args, rest = args[: len(args_strs)], args[len(args_strs) :]
                    new_static_args = {k: rest_arg for k, rest_arg in zip(static_args_strs, rest) if k not in static_args}
                    static_args = {**new_static_args, **static_args}
            else:
                args = tuple([static_args[k] if k in static_args else arg for k, arg in zip(args_strs, args)])

        symvals_in = tree_map(lambda x: SymbolicTensor.like(get_symval(x)), args)
        static_args = tuple(static_args.items())
        if self.name is None:
            self.name = f"jit_{str(hash((self.f, symvals_in, static_args)))[-5:]}"
        program, consts, out_tree = make_program(self.f, *symvals_in, static_args=static_args, name=self.name)
        return program, consts, out_tree
        program, consts, out_tree = make_program(
            self.f,
            *symvals_in,
            static_args=tuple(static_args.items()),
            name=self.name,
        )
        return program, consts, out_tree

    def __call__(self, *args, **static_args):
        program, consts, out_tree = self.get_program(*args, **static_args)
        args, in_tree = tree_flatten(args)
        outs = bind(backend.jit_op, *consts, *args, program=program)
        return tree_unflatten(out_tree, outs)

    def lower(self, *args, **static_args):
        program, consts, out_tree = self.get_program(*args, **static_args)
        args, in_tree = tree_flatten(args)
        hashed_program = Hashed(program)
        num_consts = program.num_consts
        consts, args = args[:num_consts], args[num_consts:]
        hashed_consts = tuple(map(Hashed, consts))
        jit_output = backend.jit_program(hashed_program, hashed_consts)
        return jit_output

    def export(self, output_path, args, export_params=True, input_names=None, output_names=None, **kwargs):
        if isinstance(args, Tensor):
            args, static_args = (args,), dict()
        elif not isinstance(args[-1], dict):
            assert all(isinstance(a, Tensor) for a in args)
            static_args = dict()
        else:
            args, static_args = args[:-1], args[-1]
        assert isinstance(args, (tuple, list)) and isinstance(static_args, dict)
        jit_output = self.lower(*args, **static_args)
        backend.export(jit_output, output_path, export_params, input_names, output_names, **kwargs)
