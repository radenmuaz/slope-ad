from dataclasses import dataclass
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
from functools import partial, lru_cache
import slope
import mmap
import traceback
import importlib


# ================
#   Utils
# ================


def dblog(*msg, enable=True):
    if enable:
        print(*msg)


def unzip2(pairs) -> Tuple[List[Any], List[Any]]:
    lst1, lst2 = [], []
    for i1, i2 in pairs:
        lst1.append(i1)
        lst2.append(i2)
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


def partition_list(bs: List[bool], l: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert len(bs) == len(l)
    lst1: List[Any] = []
    lst2: List[Any] = []
    lists = lst1, lst2
    for b, x in list_zip(bs, l):
        lists[b].append(x)
    return lst1, lst2


def lru_cache_verbose(maxsize: int = 100, typed: bool = False, tb_start: int = -12, tb_end: int = -7):
    def decorator(fn: Callable):
        @lru_cache(maxsize=maxsize, typed=typed)
        def wrapper(*args, **kwargs) -> Callable:
            return fn(*args, **kwargs)

        def decorated_function(*args, **kwargs) -> Any:
            result = wrapper(*args, **kwargs)
            cache_info = wrapper.cache_info()

            dblog(f"{fn.__name__}.{cache_info} {args.__hash__()}", enable=backend.LOG_LRU)
            tb = (
                "".join(traceback.format_list(traceback.extract_stack())[tb_start:tb_end]).replace("\n    ", ":\t")
                + "-" * 20
                + "\n"
            )
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
        return PPrint(self.lines[:-1] + [(indent, common_line)] + indented_block.lines[1:])

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
            if isinstance(self.val, Tensor) and isinstance(other.val, Tensor):
                # because Tensor.__eq__ already for Tensor.equal
                return id(self.val) == id(other.val)
            return self.val == other.val
        return False

    def __repr__(self):
        return f"Hashed: {repr(self.val)}"


# ================
#   Tensors
# ================


class DType(NamedTuple):
    priority: int
    itemsize: int
    name: str
    short_name: str
    numpy: type

    def __repr__(self):
        return f"{self.name}"


class TensorBuffer:
    def __init__(self, val):
        self.val = val


class Tensor:
    float32: Final[DType] = DType(4, 4, "float32", "f32", np.float32)
    uint8: Final[DType] = DType(0, 1, "uint8", "u8", np.uint8)
    int8: Final[DType] = DType(0, 1, "int8", "i8", np.int8)
    bool: Final[DType] = DType(0, 1, "bool", "i1", bool)
    int32: Final[DType] = DType(1, 4, "int32", "i32", np.int32)
    int64: Final[DType] = DType(2, 8, "int64", "i64", np.int64)
    float16: Final[DType] = DType(0, 2, "float16", "f16", np.float16)

    dtypes = (bool, float16, float32, int8, int32, int64, uint8)
    dtype_names = {k.name: k for k in dtypes}
    dtype_names_inv = {v: k for k, v in dtype_names.items()}
    dtype_short_names = {k.short_name: k for k in dtypes}
    dtype_short_names_inv = {v: k for k, v in dtype_short_names.items()}

    def __init__(self, val: TensorBuffer):
        assert isinstance(val, TensorBuffer)
        self.buf = val

    @property
    def default_dtype(self):
        return backend.default_dtype

    def is_int(self) -> bool:
        return self.dtype in (self.int8, self.uint8, self.int32, self.int64)

    def is_float(self) -> bool:
        return self.dtype in (self.float16, self.float32)

    def is_unsigned(self) -> bool:
        return self.dtype is self.uint8

    def to_bool(self):
        return self.cast(self.bool)

    def short(self):
        return self.cast(self.int8)

    def int(self):
        return self.cast(self.int32)

    def long(self):
        return self.cast(self.int64)

    def half(self):
        return self.cast(self.float16)

    def float(self):
        return self.cast(self.float32)

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
        return f"Tensor: {self.dtype}, {self.shape}]"

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

    def numpy(self):
        return backend.numpy_of(self)

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
        return f"Tensor: {self.numpy()}, {self.shape}, {self.dtype}, {self.device}"


class VoidTensor(Tensor):
    def __init__(self, shape, dtype):
        assert isinstance(dtype, DType)
        self._shape = tuple(int(i) for i in shape)
        self._dtype = dtype

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @classmethod
    def like(cls, maybe_tensor):
        shape = maybe_tensor.shape
        dtype = maybe_tensor.dtype
        return cls(shape, dtype)

    def str_short(self):
        return f'{str(self.dtype)}[{",".join(str(d) for d in self.shape)}]'

    def __hash__(self):
        return hash((self.shape, self.dtype))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return tuple(self.shape) == tuple(other.shape) and self.dtype == other.dtype

    def __repr__(self):
        return f"VoidTensor(shape={self.shape}, dtype={self.dtype})"


# ================
#   Operator
# ================


class Operator:
    def __init__(self, name, nary_inputs=False, nary_outputs=False):
        self.name = name
        self.nary_inputs = nary_inputs
        self.nary_outputs = nary_outputs
        if self.nary_inputs:
            self.reorg_args = self.reorg_args_nary

    def get_jit_name(self, args, params):
        name = f"{self.name}_"
        for a in args:
            name += f"shape_{a.shape}_dtype_{a.dtype}_"
        for k, v in params.items():
            name += f"{k}_{v}_"
        name = name.replace("(", "_lp_")
        name = name.replace(")", "_rp_")
        name = name.replace(",", "_cm_")
        name = name.replace(" ", "")
        name = name.replace(".", "_dt_")
        return name

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
        args_strs = [
            k for k, v in sig.parameters.items() if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and k != "self"
        ]
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
        void_tensors_in = [t.void_tensor for t in tracers_in]
        void_tensors_out = self.typecheck(*void_tensors_in, **params)
        tracers_out = [
            PartialRunTracerTensor(trace, make_unknown_pval(void_tensor), None) for void_tensor in void_tensors_out
        ]
        instruction = InstructionDraft(self, tracers_in, params, void_tensors_out, list_map(weakref.ref, tracers_out))
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
        return [VoidTensor(x.shape, x.dtype)]

    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [self(x, **params)], [self(x_dot, **params)]


class BinaryOperator(Operator):
    def args_fixer(self, x, w, **params):
        if type(x) is PrimalProxy or type(w) is PrimalProxy:
            assert x.shape == w.shape
            return (x, w), params

        if type(x) in TracerTensor.PYTHON_TYPES:
            x = backend.full(shape=(), fill_value=x, dtype=w.dtype)
        elif type(w) in TracerTensor.PYTHON_TYPES:
            w = backend.full(shape=(), fill_value=w, dtype=x.dtype)

        if type(x) is Tensor and isinstance(w, TracerTensor):
            x = w._trace.pure(x)
        elif type(w) is Tensor and isinstance(x, TracerTensor):
            w = x._trace.pure(w)

        if (xshape := x.shape) == (wshape := w.shape):
            return (x, w), params
        shape_delta = len(xshape) - len(wshape)
        if shape_delta > 0:
            w = w.reshape((1,) * shape_delta + wshape)
        elif shape_delta < 0:
            x = x.reshape((1,) * -shape_delta + xshape)
        if (xshape := x.shape) == (wshape := w.shape):
            return (x, w), params

        shape_ret = tuple([max(x, w) for x, w in zip(xshape, wshape)])
        if xshape != shape_ret:
            x = x.expand(shape_ret)
        if wshape != shape_ret:
            w = w.expand(shape_ret)
        return (x, w), params

    def vmap(self, dim_size, vals_in, dims_in, **params):
        (x, w), (x_bdim, y_bdim) = vals_in, dims_in
        if x_bdim != y_bdim:
            if x_bdim is None:
                x = BatchTrace.move_batch_dim(dim_size, x_bdim, y_bdim, x)
                x_bdim = y_bdim
            else:
                y = BatchTrace.move_batch_dim(dim_size, y_bdim, x_bdim, y)
        return [self(x, w, **params)], [x_bdim]

    def typecheck(self, x: VoidTensor, y: VoidTensor, **params) -> List[VoidTensor]:
        if not type(x) in (Tensor, VoidTensor) or not type(x) in (
            Tensor,
            VoidTensor,
        ):
            raise TypeError
        void_x = VoidTensor.like(x)
        void_y = VoidTensor.like(y)
        if x.dtype != y.dtype:
            raise TypeError
        if void_x == void_y:
            return [void_x]
        shape_delta = len(void_x.shape) - len(void_y.shape)
        if shape_delta > 0:
            void_y = VoidTensor((1,) * shape_delta + void_y.shape, void_y.dtype)
        elif shape_delta < 0:
            x = x.reshape((1,) * -shape_delta + void_x.shape)
            void_x = VoidTensor((1,) * -shape_delta + void_x.shape, void_x.dtype)
        if void_x == void_y:
            return [void_x]
        else:
            shape_ret = tuple([max(x, w) for x, w in zip(void_x.shape, void_y.shape)])
            if void_x.shape != shape_ret:
                void_x = VoidTensor(shape_ret, void_x.dtype)
            if void_y.shape != shape_ret:
                void_y = VoidTensor(shape_ret, void_y.dtype)
            if void_x != void_y:
                raise TypeError
            return [void_x]

    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [self(x, **params)], [self(x_dot, **params)]


class ReduceOperator(Operator):
    def args_fixer(self, x, *, dim=None, keepdim=False):
        if dim is None:
            dim = tuple(range(x.ndim))
        elif isinstance(dim, int):
            dim = (dim,)
        dim = tuple(a if a >= 0 else a + len(x.shape) for a in dim)
        return (x,), dict(dim=dim, keepdim=keepdim)

    def vmap(self, dim_size, vals_in, dims_in, **params):
        (x,), (x_bdim,) = vals_in, dims_in
        dim = list(params["dim"])
        dim = tuple(a + (x_bdim <= a) for a in dim)
        out_bdim = x_bdim - sum(a < x_bdim for a in dim)
        params["dim"] = tuple(dim)
        return [cls.do(x, **params)], [out_bdim]

    def typecheck(self, x: VoidTensor, *, dim=None, keepdim=False) -> List[VoidTensor]:
        dim = [a + len(x.shape) if a < 0 else a for a in dim]
        dim_ = set(dim)
        if keepdim:
            new_shape = [d if i not in dim_ else 1 for i, d in enumerate(x.shape)]
        else:
            new_shape = [d for i, d in enumerate(x.shape) if i not in dim_]
        return [VoidTensor(tuple(new_shape), x.dtype)]


class InitOperator(Operator):
    def jvp(self, primals, tangents, **kwargs):
        out = self(**kwargs)
        out_jvp = backend.ones_like(out)
        return [out], [out_jvp]

    def T(self, cotangents, **kwargs):
        return [None]


class ShapeOperator(Operator):
    pass


class OperatorSet:
    def __init__(self):
        self.register("jit_op")(JitOp)

    def register(self, name, nary_inputs=False, nary_outputs=False, aliases=()):
        def wrap(op_cls):
            assert name not in vars(self)
            op = op_cls(name, nary_inputs, nary_outputs)
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


class Backend:
    LOG_LRU = int(os.environ.get("LOG_LRU", 0))
    LOG_JIT = int(os.environ.get("LOG_JIT", 0))
    LOG_PYTREE = int(os.environ.get("LOG_PYTREE", 0))
    LOG_BACKEND = int(os.environ.get("LOG_BACKEND", 0))
    LOG_INIT = int(os.environ.get("LOG_INIT", 1))
    DEFAULT_DEVICE = os.environ.get("DEFAULT_DEVICE", "cpu")
    DEFAULT_DTYPE = Tensor.dtype_names[os.environ.get("DEFAULT_DTYPE", "float32")]
    dtype_map: dict
    operator_set: OperatorSet
    procedure_set: ProcedureSet

    def __init__(self):
        self.dtype_map_inv = {v: k for k, v in self.dtype_map.items()}
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
            PrimalProxy,
            lambda u: (u.void_tensor, ()),
            lambda void_tensor, _: PrimalProxy(void_tensor),
            "PrimalProxy",
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
            dblog(f"Looking {self}.{attr} in operator_set", enable=backend.LOG_BACKEND)
            return getattr(self.operator_set, attr)
        except:
            pass
        try:
            dblog(f"Looking {self}.{attr} in procedure_set", enable=backend.LOG_BACKEND)
            return getattr(self.procedure_set, attr)
        except:
            pass
        dblog(f"Fallback to default {self} getattribute", enable=backend.LOG_BACKEND)
        super().__getattribute__(attr)

    def tensor(
        self,
        val: Union[list, tuple, np.ndarray, "TensorBuffer"] = None,
        dtype: Optional[Any] = Tensor.float32,
    ):
        if isinstance(val, TensorBuffer):
            return Tensor(val)
        elif isinstance(val, Tensor):
            return val
        if type(val) is bytes:
            val = np.frombuffer(val, dtype=dtype)
        return self.from_numpy(val, dtype)

    def seed(self, seed):
        raise NotImplementedError

    @property
    def default_dtype_value(self):
        return self.dtype_map[backend.DEFAULT_DTYPE]

    def set_method(self, method):
        setattr(self, method.__name__, types.MethodType(method, self))

    def from_numpy(self, val):
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
    def gen_jit_object(
        self,
        hashed_program: Hashed,
        hashed_consts: Tuple[Hashed, ...],
    ):
        program: Program = hashed_program.val
        typecheck_program(program)
        consts = [x.val for x in hashed_consts]
        in_void_tensors = [v.void_tensor for v in program.in_binders[len(consts) :]]
        codegen_out = self.codegen(program, consts + in_void_tensors, fn_name="main")
        fn, code = self.compile(codegen_out)
        compiled = JitObject(program, codegen_out, fn, code)
        return compiled

    def codegen(self, program: "Program", args: Tuple, in_void_tensors: Tuple, name: str):
        "Returns compiler IR from the Program"
        raise NotImplementedError

    def compile(self, program: "Program", args: Tuple, in_void_tensors: Tuple, name: str):
        "Compiles compiler IR to a Python callable function"
        raise NotImplementedError

    def export(self, jit_object, output_path, *args, **params):
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
                        dtype = Tensor.dtype_short_names[(v["dtype"])]
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
                "dtype": v.dtype.short_name,
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


# ================
#   IR, Programs, Instructions
# ================


class Var:
    val = None
    void_tensor: VoidTensor

    def __init__(self, void_tensor):
        self.void_tensor = void_tensor


class Lit:
    val: Any
    void_tensor: VoidTensor

    def __init__(self, val):
        self.void_tensor = VoidTensor.like(get_void_tensor_or_tensor(val))
        self.val = val


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
    num_consts: int = 0
    static_args: Any = ()
    name: str = "my_program"

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        return self is other

    def __repr__(self):
        namegen = (
            # "z" + repr(r) for r in itertools.count()
            "".join(s)
            for r in itertools.count(1)
            for s in itertools.permutations(string.ascii_lowercase, r)
        )
        names = defaultdict(lambda: next(namegen))
        in_binders = ", ".join(self.var_str(names, x) for x in self.in_binders)
        instructions = PPrint.vcat([self.pp_instruction(names, e) for e in self.instructions])
        outs = [names[v] if isinstance(v, Var) else str(v.val) for v in self.outs]
        outs = ", ".join(outs)
        ret = str(
            PPrint.pp(f"{{ {self.name} {in_binders} .")
            + ((PPrint.pp("let ") >> instructions) + PPrint.pp(f"in ( {outs} ) }}")).indent(2)
        )
        return ret

    def pp_instruction(self, names: DefaultDict[Var, str], instruction: Instruction) -> PPrint:
        lhs = PPrint.pp(" ".join(self.var_str(names, v) for v in instruction.out_binders))
        rhs = (
            PPrint.pp(repr(instruction.op.name))
            >> self.pp_params(instruction.params)
            >> PPrint.pp(" ".join(names[x] if isinstance(x, Var) else str(x.val) for x in instruction.inputs))
        )
        return lhs >> PPrint.pp(" = ") >> rhs

    def pp_params(self, params: Dict[str, Any]) -> PPrint:
        items = sorted(params.items())
        if items:
            return PPrint.pp(" [ ") >> PPrint.vcat([PPrint.pp(f"{k}={v}") for k, v in items]) >> PPrint.pp(" ] ")
        else:
            return PPrint.pp(" ")

    def var_str(self, names: DefaultDict[Var, str], v) -> str:
        return f"{names[v]}:{v.void_tensor.str_short()}"


class ProgramType(NamedTuple):
    in_types: Tuple[VoidTensor]
    out_types: Tuple[VoidTensor]

    def __repr__(self):
        in_types = ", ".join(void_tensor.str_short() for void_tensor in self.in_types)
        out_types = ", ".join(void_tensor.str_short() for void_tensor in self.out_types)
        return f"({in_types}) -> ({out_types})"


# ================
#   Tracer and Trace
# ================


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


class PyTreeDef(NamedTuple):
    node_type: NodeType
    node_metadata: Hashable
    child_treedefs: Tuple["PyTreeDef", ...]

    def __repr__(self):
        ret = self.tree_repr(self)
        return ret

    def tree_repr(self, pytree, indent="  ", prefix="", last=True):
        ret = ""

        def _tree_repr(pytree, indent, prefix, last):
            nonlocal ret
            if isinstance(pytree, PyTreeDef):
                ret += f'{prefix} {("└─" if last else "├─")} {pytree.node_type.name}\n'
                for i, item in enumerate(pytree.child_treedefs):
                    new_prefix = prefix + (indent if not last else "   ")
                    new_last = i == len(pytree.child_treedefs) - 1
                    _tree_repr(item, indent, new_prefix, new_last)
            else:
                ret += f'{prefix} {("└─" if last else "├─")} {pytree}\n'

        _tree_repr(pytree, indent="  ", prefix="", last=True)
        return ret

    @property
    def num_leaves(self):
        def _get_num_leaves(x_):
            if isinstance(x_, Leaf):
                return 1
            else:
                return sum(_get_num_leaves(x__) for x__ in x_.child_treedefs)

        return sum(_get_num_leaves(x) for x in self.child_treedefs)


class Leaf:
    def __init__(self, val):
        if hasattr(val, "shape"):
            val = VoidTensor.like(val)
        self.val = val

    def __repr__(self):
        if isinstance(self.val, VoidTensor):
            return self.val.str_short()
        return repr(self.val)

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        if isinstance(other, Leaf):
            return self.val == other.val


# ================
#   jit operator
# ================


class JitObject:
    def __init__(self, program, codegen_out, fn, code):
        super().__init__()
        self.program = program
        self.code = code
        self.codegen_out = codegen_out
        self.fn = fn

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

    def export(self, output_path, *args, **params):
        return backend.export(self, output_path, *args, **params)


class JitOp(MetaOperator):
    def meta_impl(self, *args, program):
        hashed_program = Hashed(program)
        num_consts = program.num_consts
        consts, args = args[:num_consts], args[num_consts:]
        hashed_consts = tuple(map(Hashed, consts))
        jit_object = backend.gen_jit_object(hashed_program, hashed_consts)
        ret = jit_object(*consts, *args)
        return ret

    def reorg_args(self, args, params):
        return args, params

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

    def typecheck(self, *in_types, program):
        program_type = typecheck_program(program)
        if not all(t1 == t2 for t1, t2 in zip(program_type.in_types, in_types)):
            for i, j in zip(program_type.in_types, in_types):
                print(i, j, i == j)
            breakpoint()
            raise TypeError
        return program_type.out_types

    def T(self, cotangents, *invals, program):
        undef_primals = [type(x) is PrimalProxy for x in invals]
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
        breakpoint()
        return [next(outs) if undef else None for undef in undef_primals]

    def partial_run(self, trace, tracers, *, program):
        in_unknowns = [not t.pval.is_known for t in tracers]
        program1, program2, out_unknowns, num_res = partial_run_program(program, in_unknowns)
        known_tracers, unknown_tracers = partition_list(in_unknowns, tracers)
        known_vals = [t.pval.const for t in known_tracers]
        outs1_res = bind(backend.jit_op, *known_vals, program=program1)
        outs1, res = split_list(outs1_res, len(program1.outs) - num_res)
        res_tracers = [trace.instantiate_const(full_raise(trace, x)) for x in res]
        outs2 = [PartialRunTracerTensor(trace, make_unknown_pval(v.void_tensor), None) for v in program2.outs]
        instruction = InstructionDraft(
            self,
            res_tracers + unknown_tracers,
            dict(program=program2),
            [v.void_tensor for v in program2.outs],
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
        res = [Var(v.void_tensor) for v in program2.in_binders[:num_res]]
        instruction1 = Instruction(self, ins1, dict(program=program1), out_binders1 + res)
        instruction2 = Instruction(self, res + ins2, dict(program=program2), out_binders2)
        return instruction1, instruction2, out_unknowns, res


# ================
#   Compiler
# ================


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
            fn = self.get_fn(op, *tuple(VoidTensor.like(a) for a in args), **params)
            ret = jit(fn, static_argnames=("params",), name=op.get_jit_name(args, params))(*args, **params)

        return ret

    @staticmethod
    @lru_cache_verbose()
    def get_fn(op, *void_tensor_args, **params):
        def fn(*args, **params):
            return [op(*args, **params)]

        return fn


class TracerTensor(Tensor):
    PYTHON_TYPES = {
        bool,
        int,
        float,
    }
    _trace: "Trace"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    void_tensor = property(lambda self: get_void_tensor_or_tensor(self.val))
    dtype = property(lambda self: self.void_tensor.dtype)
    shape = property(lambda self: self.void_tensor.shape)

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
        return f"{self.__class__.__name__}({repr(self.void_tensor)})"


Batchdim = Union[None, int]


class BatchTracerTensor(TracerTensor):
    def __init__(self, trace, val, batch_dim: Batchdim):
        self._trace = trace
        self.val = val
        self.batch_dim = batch_dim

    @property
    def void_tensor(self):
        void_tensor = get_void_tensor_or_tensor(self.val)
        if self.batch_dim is None:
            return void_tensor
        else:
            shape = list(void_tensor.shape)
            del shape[self.batch_dim]
            return VoidTensor(tuple(shape), void_tensor.dtype)

    def full_lower(self):
        if self.batch_dim is None:
            return full_lower(self.val)
        else:
            return self


class BatchTrace(Trace):
    pure = lambda self, val: BatchTracerTensor(self, val, None)

    def run_op(self, op, tracers, params):
        vals_in, bdims_in = unzip2((t.val, t.batch_dim) for t in tracers)
        val_outs, bdim_outs = op.vmap(self.dim_size, vals_in, bdims_in, **params)
        return [BatchTracerTensor(self, x, bd) for x, bd in list_zip(val_outs, bdim_outs)]

    @property
    def dim_size(self):
        return self.main.global_data

    @staticmethod
    def move_batch_dim(dim_size, src, dst, x):
        if src is None:
            target_shape = list(x.shape)
            target_shape.insert(dst, dim_size)
            out_ndim = len(target_shape)
            if type(dst) in (tuple, list):
                out_ndim += 1
            reshape_shape = [1 if ax == dst else target_shape for ax in range(out_ndim)]
            x = x.reshape(reshape_shape)
            x = x.expand(target_shape)
            return x
        elif src == dst:
            return x
        else:
            perm = [i for i in range(len(x.shape)) if i != src]
            perm.insert(dst, src)
            return x.permute(perm)


class JVPTracerTensor(TracerTensor):
    def __init__(self, trace, primal, tangent):
        self._trace = trace
        self.primal = primal
        self.tangent = tangent

    @property
    def void_tensor(self):
        return get_void_tensor_or_tensor(self.primal)

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
        return JVPTracerTensor(self, val, backend.zeros_like(val))

    def run_op(self, op, tracers, params):
        primals_in, tangents_in = unzip2((t.primal, t.tangent) for t in tracers)
        primals_out, tangents_out = op.jvp(primals_in, tangents_in, **params)
        return [JVPTracerTensor(self, x, t) for x, t in list_zip(primals_out, tangents_out)]


class ProgramTracerTensor(TracerTensor):
    __slots__ = ["void_tensor"]
    void_tensor: VoidTensor

    def __init__(self, trace, void_tensor):
        self._trace = trace
        self.void_tensor = void_tensor


class ProgramTrace(Trace):
    def new_arg(self, void_tensor) -> ProgramTracerTensor:
        void_tensor = VoidTensor.like(void_tensor)
        tracer = self.builder.new_tracer(self, void_tensor)
        self.builder.tracer_to_var[id(tracer)] = Var(void_tensor)

        return tracer

    def pure(self, val: Any) -> ProgramTracerTensor:
        # get_or_make_const_tracer
        tracer = self.builder.const_tracers.get(id(val))
        if tracer is None:
            tracer = self.builder.new_tracer(self, get_void_tensor_or_tensor(val))
            self.builder.add_const(tracer, val)
        return tracer

    def run_op(self, op, tracers, params):
        void_tensors_in = [t.void_tensor for t in tracers]
        void_tensors_in = tree_map(lambda x: x.void_tensor, tracers)
        void_tensors_out = op.typecheck(*void_tensors_in, **params)

        out_tracers = [self.builder.new_tracer(self, a) for a in void_tensors_out]
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
    const_tracers: Dict[int, TracerTensor]
    constvals: Dict[Var, Any]
    tracers: List[ProgramTracerTensor]

    def __init__(self):
        self.instructions = []
        self.tracer_to_var = {}
        self.const_tracers = {}
        self.constvals = {}
        self.tracers = []

    def new_tracer(self, trace: ProgramTrace, void_tensor: VoidTensor) -> ProgramTracerTensor:
        tracer = ProgramTracerTensor(trace, void_tensor)
        self.tracers.append(tracer)
        return tracer

    def add_instruction(self, instruction: Instruction) -> None:
        self.instructions.append(instruction)

    def add_var(self, tracer: ProgramTracerTensor) -> Var:
        assert id(tracer) not in self.tracer_to_var
        var = self.tracer_to_var[id(tracer)] = Var(tracer.void_tensor)
        return var

    def getvar(self, tracer: ProgramTracerTensor) -> Var:
        var = self.tracer_to_var.get(id(tracer))
        assert var is not None
        return var

    def add_const(self, tracer: ProgramTracerTensor, val: Any) -> Var:
        var = self.add_var(tracer)
        self.const_tracers[id(val)] = tracer
        self.constvals[var] = val
        return var

    def build(self, in_tracers: Any, out_tracers: Any, static_args, name) -> Tuple[Program, List[Any]]:
        constvars, constvals = unzip2(self.constvals.items())
        t2v = lambda t: self.tracer_to_var[id(t)]
        in_binders = constvars + [t2v(t) for t in in_tracers]
        out_vars = [t2v(t) for t in out_tracers]
        program = Program(in_binders, self.instructions, out_vars, len(constvals), static_args, name)
        typecheck_program(program)
        program, constvals = self._inline_literals(program, constvals)
        return program, constvals

    def _inline_literals(self, program: Program, consts: List[Any]) -> Tuple[Program, List[Any]]:
        const_binders, other_binders = split_list(program.in_binders, len(consts))
        scalars = [type(x) in TracerTensor.PYTHON_TYPES and not get_void_tensor_or_tensor(x).shape for x in consts]
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
            new_const_binders + other_binders,
            new_instructions,
            new_outs,
            len(new_consts),
            program.static_args,
            program.name,
        )
        typecheck_program(new_program)
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
        return {"Function": current_function_name, "Module": current_module_name, "Class": current_class_name}


class PrimalProxy(NamedTuple):
    void_tensor: VoidTensor

    @property
    def shape(self):
        return self.void_tensor.shape

    @property
    def dtype(self):
        return self.void_tensor.dtype

    def __repr__(self):
        return f"PrimalProxy: {self.void_tensor}"

    def str_short(self):
        return f"PrimalProxy: {self.dtype}, {self.shape}"


class PartialValue(NamedTuple):
    void_tensor: VoidTensor
    const: Optional[Any]

    is_known = property(lambda self: self.const is not None)
    is_unknown = property(lambda self: self.const is None)


class LambdaBindingDraft(NamedTuple):
    pass


class ConstDraft(NamedTuple):
    val: Any


class InstructionDraft(NamedTuple):
    prim: Operator
    tracers_in: List["PartialRunTracerTensor"]
    params: Dict[str, Any]
    void_tensors_out: List[VoidTensor]
    tracer_refs_out: List[weakref.ReferenceType["PartialRunTracerTensor"]]


ProgramDraft = Union[LambdaBindingDraft, ConstDraft, InstructionDraft]


class PartialRunTracerTensor(TracerTensor):
    def __init__(self, trace, pval, draft):
        self._trace = trace
        self.pval = pval
        self.draft = draft

    void_tensor = property(lambda self: self.pval.void_tensor)
    val = property(lambda self: self.pval.const)

    def full_lower(self):
        if self.pval.is_known:
            return full_lower(self.pval.const)
        return self


class PartialRunTrace(Trace):
    def new_arg(self, pval: PartialValue) -> Any:
        return PartialRunTracerTensor(self, pval, LambdaBindingDraft())

    def pure(self, val: Any) -> PartialRunTracerTensor:
        return PartialRunTracerTensor(self, make_known_pval(val), None)

    def instantiate_const(self, tracer: PartialRunTracerTensor) -> PartialRunTracerTensor:
        if tracer.pval.is_unknown:
            return tracer
        else:
            pval = make_unknown_pval(VoidTensor.like(tracer.void_tensor))
            return PartialRunTracerTensor(self, pval, ConstDraft(tracer.pval.const))

    def run_op(self, op, tracers, params):
        is_knowns = tuple(t.pval.is_known for t in tracers)

        if all(is_knowns):  # and op in backend.impls.keys():
            return bind(op, *list_map(full_lower, tracers), **params)
        return op.partial_run(self, tracers, **params)


trace_stack: List[MainTrace] = []
dynamic_trace: Optional[MainTrace] = None
trace_stack += [MainTrace(0, RunTrace, None)]


class BackendPlaceholder:
    def __getattr__(self, attr):
        raise NotImplementedError("Backend not init yet with slope.core.set_backend(backend)")


backend = BackendPlaceholder()


def set_backend(init_backend):
    global backend
    backend = init_backend
    dblog(f"slope backend is {backend}", enable=backend.LOG_INIT)


def stack_str():
    ret = ""
    for trace in trace_stack:
        ret += f"{trace.level}: {trace.trace_type.__name__}\t{trace.global_data=}\n"
    return ret


def make_known_pval(val: Any):
    return PartialValue(get_void_tensor_or_tensor(val), val)


def make_unknown_pval(void_tensor: VoidTensor):
    return PartialValue(void_tensor, None)


def get_void_tensor_or_tensor(x):
    if isinstance(x, TracerTensor):
        return x.void_tensor
    elif type(x) in TracerTensor.PYTHON_TYPES:
        return backend.tensor(x)
    elif isinstance(x, Tensor):
        return x
    elif isinstance(x, VoidTensor):
        return x
    else:
        raise TypeError(type(x))


def tree_flatten(x: Any) -> Any:
    def _tree_flatten(x_: Any) -> Tuple[Iterable, Union[PyTreeDef, Leaf]]:
        node_type = None
        for k in backend.node_types.keys():
            if isinstance(x_, k):
                node_type = backend.node_types[k]

        if node_type is not None:
            node_metadata, children = node_type.flatten(x_)
            children_flat, child_trees = unzip2(list_map(_tree_flatten, children))
            children_iter = itertools.chain.from_iterable(children_flat)
            treedef = PyTreeDef(node_type, node_metadata, tuple(child_trees))
            return children_iter, treedef
        else:
            return (x_,), Leaf(x_)

    children_iter, treedef = _tree_flatten(x)
    return tuple(children_iter), treedef


def tree_unflatten(treedef: PyTreeDef, xs: Tuple[Any]) -> Any:
    def _tree_unflatten(treedef_: PyTreeDef, xs_: Iterator) -> Any:
        if isinstance(treedef_, Leaf):
            dblog(f"    tree leaf found: {xs_}\n", enable=backend.LOG_PYTREE)
            return next(xs_)
        else:
            dblog(f"    now\n  {treedef_}", enable=backend.LOG_PYTREE)
            children = (_tree_unflatten(t, xs_) for t in treedef_.child_treedefs)
            dblog(f"{children=}\n", enable=backend.LOG_PYTREE)
            return treedef_.node_type.unflatten(treedef_.node_metadata, children)

    dblog(f"unflattening {treedef}", enable=backend.LOG_PYTREE)
    return _tree_unflatten(treedef, iter(xs))


def tree_transpose(
    outer_treedef: PyTreeDef,
    inner_treedef: PyTreeDef,
    pytree_to_transpose: Any,
) -> Any:
    flat, treedef = tree_flatten(pytree_to_transpose)
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
        pytree_args = tree_unflatten(in_tree, args_flat)
        out = f(*pytree_args, **params)
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
def new_main(trace_type: Type["Trace"], global_data=None):
    level = len(trace_stack)
    main = MainTrace(level, trace_type, global_data)
    trace_stack.append(main)

    try:
        yield main
    finally:
        trace_stack.pop()


@contextmanager
def new_dynamic(main: MainTrace):
    global dynamic_trace
    prev_dynamic_trace, dynamic_trace = dynamic_trace, main
    try:
        yield
    finally:
        dynamic_trace = prev_dynamic_trace


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
            elif isinstance(x, TracerTensor):
                arrs += [x]

    get_arr_from_seq(xs)
    arrs = tuple(arrs)
    top_main = max(
        (x._trace.main for x in arrs),
        default=trace_stack[0],
        key=operator_py.attrgetter("level"),
    )
    if dynamic_trace and dynamic_trace.level > top_main.level:
        top_main = dynamic_trace
    return top_main.trace_type(top_main)


def full_raise(trace: Trace, val: Any) -> TracerTensor:
    if not isinstance(val, TracerTensor):
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
    if isinstance(val, TracerTensor):
        return val.full_lower()
    elif type(val) in (list, tuple):
        return tuple(full_lower(v) for v in val)
    else:
        return val


def typecheck_program(program: Program) -> ProgramType:
    backend: Set[Var] = set()

    for v in program.in_binders:
        if v in backend:
            raise TypeError
        backend.add(v)

    for instruction in program.instructions:
        in_types = [typecheck_atom(backend, x) for x in instruction.inputs]
        out_types = instruction.op.typecheck(*in_types, **instruction.params)
        for out_binder, out_type in list_zip(instruction.out_binders, out_types):
            if not out_type == out_binder.void_tensor:
                raise TypeError
        for out_binder in instruction.out_binders:
            if out_binder in backend:
                raise TypeError
            backend.add(out_binder)

    in_types = [v.void_tensor for v in program.in_binders]
    out_types = [typecheck_atom(backend, x) for x in program.outs]
    return ProgramType(tuple(in_types), tuple(out_types))


def typecheck_atom(backend: Set[Var], x: Atom) -> VoidTensor:
    if isinstance(x, Var):
        if x not in backend:
            raise TypeError("unbound variable")
        return x.void_tensor
    elif isinstance(x, Lit):
        return get_void_tensor_or_tensor(x.val)
    else:
        assert False


def run_program(program: Program, args: List[Any]) -> List[Any]:
    backend: Dict[Var, Any] = {}

    def read(x: Atom) -> Any:
        return backend[x] if type(x) is Var else x.val

    def write(v: Var, val: Any) -> None:
        assert v not in backend  # single-assignment
        backend[v] = val

    list_map(write, program.in_binders, args)
    for instruction in program.instructions:
        in_vals = list_map(read, instruction.inputs)
        outs = bind(instruction.op, *in_vals, **instruction.params)
        list_map(write, instruction.out_binders, outs)
    return list_map(read, program.outs)


def program_as_fun(program: Program):
    return lambda *args: run_program(program, args)


def vmap_flat(f, in_dim, *args):
    axi_set = {x.shape[ax] for x, ax in list_zip(args, in_dim) if ax is not None}
    assert len(axi_set) == 1
    (dim_size,) = axi_set
    with new_main(BatchTrace, dim_size) as main:
        trace = BatchTrace(main)
        tracers_in = [BatchTracerTensor(trace, x, ax) if ax is not None else x for x, ax in list_zip(args, in_dim)]
        outs = f(*tracers_in)
        tracers_out = [full_raise(trace, out) for out in outs]
        vals_out, bdims_out = unzip2((t.val, t.batch_dim) for t in tracers_out)
    outs_permuted = [
        BatchTrace.move_batch_dim(dim_size, bdim, 0, val_out) for val_out, bdim in list_zip(vals_out, bdims_out)
    ]
    return outs_permuted


def vmap(f, in_dim=0, out_dim=0):
    def batched_f(*args):
        args_flat, in_tree = tree_flatten(args)
        in_dim_flat, in_tree2 = tree_flatten(in_dim)
        if in_tree != in_tree2:
            raise TypeError(f"{in_tree}\n!=\n{in_tree2}")
        f_flat, out_tree_store = flatten_fn(f, in_tree)
        outs_flat = vmap_flat(f_flat, in_dim_flat, *args_flat)
        return tree_unflatten(out_tree_store(), outs_flat)

    return batched_f


def jvp_flat(f, primals, tangents, *, has_aux, global_data, **static_args):
    with new_main(JVPTrace, global_data) as main:
        trace = JVPTrace(main)
        tracers_in = [JVPTracerTensor(trace, x, t) for x, t in list_zip(primals, tangents)]
        jvp_flat_ret = f(*tracers_in, **static_args)
        if has_aux:
            (outs, aux) = jvp_flat_ret
            # aux_ = aux
            aux = tree_map(lambda x: x.primal, aux)
            # aux = tree_map(lambda x: x.full_lower(), aux)
            # breakpoint()
        else:
            outs = jvp_flat_ret
        tracers_out = [full_raise(trace, out) for out in outs]
        primals_out, tangents_out = unzip2((t.primal, t.tangent) for t in tracers_out)
    return ((primals_out, tangents_out), aux) if has_aux else (primals_out, tangents_out)


def jvp(f, primals, tangents, *, has_aux=False, global_data=None, **static_args):
    primals_flat, in_tree = tree_flatten(primals)
    tangents_flat, in_tree2 = tree_flatten(tangents)
    if in_tree != in_tree2:
        raise TypeError
    f, out_tree_store = flatten_fn(f, in_tree, has_aux=has_aux)
    jvp_ret = jvp_flat(f, primals_flat, tangents_flat, has_aux=has_aux, global_data=global_data, **static_args)
    if has_aux:
        (primals_out_flat, tangents_out_flat), aux = jvp_ret
    else:
        (primals_out_flat, tangents_out_flat) = jvp_ret
    primals_out = tree_unflatten(out_tree_store(), primals_out_flat)
    tangents_out = tree_unflatten(out_tree_store(), tangents_out_flat)
    return ((primals_out, tangents_out), aux) if has_aux else (primals_out, tangents_out)


# def jacfwd(f, x):
#     pushfwd = lambda v: jvp(f, (x,), (v,))[1]
#     vecs_in = eye(math.prod(x.shape)).reshape(x.shape * 2)
#     return vmap(pushfwd, (0,))(vecs_in)


@lru_cache_verbose()
def make_program(f: Callable, *void_tensors_in: VoidTensor, static_args, name) -> Tuple[Program, List[Any], PyTreeDef]:
    void_tensors_in, in_tree = tree_flatten(void_tensors_in)
    f, out_tree_store = flatten_fn(f, in_tree)
    builder = ProgramBuilder()
    with new_main(ProgramTrace, builder) as main:
        with new_dynamic(main):
            trace = ProgramTrace(main)
            tracers_in = [trace.new_arg(void_tensor) for void_tensor in void_tensors_in]
            outs = f(*tracers_in, **{k: v for k, v in static_args})
            # tracers_out = [full_raise(trace, out) for out in outs]
            # raise check because of aux is not ProgramTracerTensor
            tracers_out = [full_raise(trace, out) if isinstance(out, ProgramTracerTensor) else out.val for out in outs]
            program, consts = builder.build(tracers_in, tracers_out, static_args, name)

    return program, consts, out_tree_store()


@lru_cache_verbose()
def jvp_program(program: Program, static_args=()) -> Tuple[Program, List[Any]]:
    def jvp_traceable(*primals_and_tangents):
        n = len(primals_and_tangents) // 2
        primals, tangents = primals_and_tangents[:n], primals_and_tangents[n:]
        return jvp(program_as_fun(program), primals, tangents)

    in_void_tensors = tree_map(lambda v: v.void_tensor, program.in_binders)
    new_program, new_consts, _ = make_program(
        jvp_traceable,
        *in_void_tensors,
        *in_void_tensors,
        static_args=static_args,
        name=f"{program.name}_jvp",
    )
    return new_program, new_consts


def partial_run_flat(
    f: Callable, pvals_in: List["PartialValue"], has_aux, global_data=None
) -> Tuple[Program, List["PartialValue"], List[Any]]:
    with new_main(PartialRunTrace, global_data) as main:
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
    self,
    program: Program,
    in_unknowns: List[bool],
    instantiate: Optional[List[bool]] = None,
) -> Tuple[Program, Program, List[bool], int]:
    backend: Dict[Var, bool] = {}
    residuals: Set[Var] = set()

    def read(x: Atom) -> bool:
        return type(x) is Var and backend[x]

    def write(unk: bool, v: Var) -> None:
        backend[v] = unk

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
    pvals_in = [make_known_pval(x) for x in primals_in] + [
        make_unknown_pval(VoidTensor.like(get_void_tensor_or_tensor(x))) for x in primals_in
    ]

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
    tracers_in: List["PartialRunTracerTensor"],
    tracers_out: List["PartialRunTracerTensor"],
):
    def tracer_parents(t: PartialRunTracerTensor) -> List[PartialRunTracerTensor]:
        return t.draft.tracers_in if isinstance(t.draft, InstructionDraft) else []

    def draft_to_instruction(tracer_to_var: Dict[int, Var], draft: InstructionDraft) -> Instruction:
        inputs = [tracer_to_var[id(t)] for t in draft.tracers_in]
        out_binders = [Var(void_tensor) for void_tensor in draft.void_tensors_out]
        for t_ref, var in list_zip(draft.tracer_refs_out, out_binders):
            if t_ref() is not None:
                tracer_to_var[id(t_ref())] = var
        return Instruction(draft.prim, inputs, draft.params, out_binders)

    tracer_to_var: Dict[int, Var] = {id(t): Var(VoidTensor.like(t.void_tensor)) for t in tracers_in}
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
                void_tensor = VoidTensor.like(get_void_tensor_or_tensor(val))
                var = constid_to_var[id(val)] = Var(void_tensor)
                constvar_to_val[var] = val
            tracer_to_var[id(t)] = var
        elif isinstance(t.draft, InstructionDraft):
            if id(t.draft) not in processed_instructions:
                instructions.append(draft_to_instruction(tracer_to_var, t.draft))
                processed_instructions.add(id(t.draft))
        else:
            raise TypeError(t.draft)

    constvars, constvals = unzip2(constvar_to_val.items())
    in_binders = constvars + [tracer_to_var[id(t)] for t in tracers_in]
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
        sorted_nodes.append(node)
        for parent in parents(node):
            if child_counts[id(parent)] == 1:
                childless_nodes.append(parent)
            else:
                child_counts[id(parent)] -= 1

    sorted_nodes = sorted_nodes[::-1]
    check_toposort(sorted_nodes, parents)
    return sorted_nodes


def vjp_flat(f, *primals_in, has_aux=False, **static_args):
    pvals_in = [make_known_pval(x) for x in primals_in] + [
        make_unknown_pval(VoidTensor.like(get_void_tensor_or_tensor(x))) for x in primals_in
    ]
    _, tangent_pvals_in = split_half(pvals_in)

    def f_jvp(*primals_tangents_in):
        jvp_ret = jvp(f, *split_half(primals_tangents_in), has_aux=has_aux, global_data="vjp", **static_args)
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

    primal_pvals, _ = split_half(pvals_out)
    assert all(pval.is_known for pval in primal_pvals)
    primals_out_flat = [pval.const for pval in primal_pvals]
    transpose_inputs = consts + [PrimalProxy(p.void_tensor) for p in tangent_pvals_in]
    f_vjp_flat = lambda *cotangents: run_program_transposed(program, transpose_inputs, cotangents)
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


def run_program_transposed(program: Program, args: List[Any], cotangents: List[Any], **others) -> List[Any]:
    primal_env: Dict[Var, Any] = {}
    ct_env: Dict[Var, Any] = {}

    def read_primal(x: Atom) -> Any:
        return primal_env.get(x, PrimalProxy(x.void_tensor)) if type(x) is Var else x.val

    def write_primal(v: Var, val: Any) -> None:
        if type(val) is not PrimalProxy:
            primal_env[v] = val

    def read_cotangent(v: Var) -> Any:
        return ct_env.pop(v, backend.zeros(v.void_tensor.shape, v.void_tensor.dtype))

    def write_cotangent(x: Atom, val: Any):
        if type(x) is Var and val is not None:
            ct_env[x] = ct_env[x] + val if x in ct_env else val

    list_map(write_primal, program.in_binders, args)
    list_map(write_cotangent, program.outs, cotangents)
    for instruction in program.instructions[::-1]:
        primals_in = list_map(read_primal, instruction.inputs)
        cotangents_in = list_map(read_cotangent, instruction.out_binders)
        inp, params = primals_in, instruction.params
        inp, params = instruction.op.reorg_args(inp, params)
        inp, params = instruction.op.args_fixer(*inp, **params)
        cotangents_out = instruction.op.T(cotangents_in, *inp, **params)
        list_map(write_cotangent, instruction.inputs, cotangents_out)

    ret = [read_cotangent(v) for v, x in list_zip(program.in_binders, args) if type(x) is PrimalProxy]

    return ret


@lru_cache_verbose()
def transpose_program(program: Program, undef_primals: tuple[bool, ...]) -> tuple[Program, list[Any]]:
    void_tensors_in, void_tensors_out = typecheck_program(program)
    traceable = partial(run_program_transposed, program)
    args = [PrimalProxy(a) if u else a for a, u in zip(void_tensors_in, undef_primals)]
    trans_program, consts, _ = make_program(
        traceable,
        tuple(args),
        tuple(void_tensors_out),
        static_args=program.static_args,
        name=f"{program.name}_T",
    )
    typecheck_program(trans_program)

    return trans_program, consts


def grad(f, argnums=(0,), argnames="", has_aux=False, return_value=False):
    f, rejit = (f, False) if not isinstance(f, jit) else (f.f, True)
    if isinstance(argnums, int):
        argnums = (argnums,)

    def grad_fn(x, *xs, **static_args):
        vjp_ret = vjp(f, x, *xs, has_aux=has_aux, **static_args)
        if has_aux:
            y, f_vjp, aux = vjp_ret
        else:
            y, f_vjp = vjp_ret
        if np.shape(y) != ():
            raise TypeError("grad output must be 0-dim scalar with shape ()")
        grad_L_xs = f_vjp(backend.ones(()))
        grad_L_xs = tuple(grad_L_xs[i] for i in argnums) if len(argnums) > 1 else grad_L_xs[argnums[0]]
        if return_value:
            return ((y, aux), grad_L_xs) if has_aux else (y, grad_L_xs)
        else:
            return (grad_L_xs, aux) if has_aux else grad_L_xs

    return jit(grad_fn) if rejit else grad_fn


def value_and_grad(f, argnums=(0,), argnames="", has_aux=False):
    return grad(f, argnums=argnums, argnames=argnames, has_aux=has_aux, return_value=True)


def jit_partial_run(trace, tracers, *, program):
    in_unknowns = [not t.pval.is_known for t in tracers]
    program1, program2, out_unknowns, num_res = partial_run_program(program, in_unknowns)
    known_tracers, unknown_tracers = partition_list(in_unknowns, tracers)
    known_vals = [t.pval.const for t in known_tracers]
    outs1_res = backend.jit_op(*known_vals, program=program)
    outs1, res = split_list(outs1_res, len(program1.outs) - num_res)
    res_tracers = [trace.instantiate_const(full_raise(trace, x)) for x in res]
    outs2 = [PartialRunTracerTensor(trace, PartialValue.unknown(v.void_tensor), None) for v in program2.outs]
    draft = InstructionDraft(
        backend.jit_op,
        res_tracers + unknown_tracers,
        dict(program=program2),
        [v.void_tensor for v in program2.outs],
        map(weakref.ref, outs2),
    )
    for t in outs2:
        t.draft = draft
    return merge_lists(out_unknowns, outs1, outs2)


class jit:
    def __init__(self, f, static_argnames=(), name=None):
        assert type(static_argnames) is tuple and all(type(s) is str for s in static_argnames)
        self.f = f
        self.name = name
        self.static_argnames = static_argnames

    @classmethod
    def with_options(cls, static_argnames=(), name=None):
        return partial(cls, static_argnames=static_argnames, name=name)

    def get_program(self, *args, **static_args):
        sig = inspect.signature(self.f)
        if all("*" not in repr(v) for v in sig.parameters.values()):
            args_strs = [k for k, v in sig.parameters.items() if k != "self" and k not in self.static_argnames]
            static_args_strs = [k for k, v in sig.parameters.items() if k != "self" and k in self.static_argnames]

            if args:
                if len(args) > len(args_strs):
                    assert static_args_strs
                    args, rest = args[: len(args_strs)], args[len(args_strs) :]
                    new_static_args = {
                        k: rest_arg for k, rest_arg in zip(static_args_strs, rest) if k not in static_args
                    }
                    static_args = {**new_static_args, **static_args}
            else:
                args = tuple([static_args[k] if k in static_args else arg for k, arg in zip(args_strs, args)])

        static_args = tuple(static_args.items())

        void_tensors_in = tree_map(lambda x: VoidTensor.like(get_void_tensor_or_tensor(x)), args)
        if self.name is None:
            self.name = f"jit_{str(hash((self.f, void_tensors_in, static_args)))[1:5]}"
        program, consts, out_tree = make_program(self.f, *void_tensors_in, static_args=static_args, name=self.name)
        return program, consts, out_tree

    def __call__(self, *args, **static_args):
        program, consts, out_tree = self.get_program(*args, **static_args)
        args, in_tree = tree_flatten(args)
        outs = bind(backend.jit_op, *consts, *args, program=program)
        return tree_unflatten(out_tree, outs)

    def get_jit_object(self, *args, **static_args):
        program, consts, out_tree = self.get_program(*args, **static_args)
        args, in_tree = tree_flatten(args)
        hashed_program = Hashed(program)
        num_consts = program.num_consts
        consts, args = args[:num_consts], args[num_consts:]
        hashed_consts = tuple(map(Hashed, consts))
        jit_object = backend.gen_jit_object(hashed_program, hashed_consts)
        return jit_object


#
