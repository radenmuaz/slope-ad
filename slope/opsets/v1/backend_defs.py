import slope as sp
from slope.opsets.v1.ops_defs import ops
from slope.opsets.v1.procs_defs import procs
from slope.core import Backend, BaseArray, VoidArray, list_zip, list_map
import numpy as np
from typing import (
    List,
    Dict,
    Any,
)
import pickle
import math
import inspect
import re

numpy_backend = Backend("numpy")
numpy_dtype_map = {
    BaseArray.float32: np.dtype("float32"),
    BaseArray.int64: np.dtype("int64"),
    BaseArray.int8: np.dtype("int8"),
    BaseArray.bool: bool,
}
default_dtype = BaseArray.default_dtype
numpy_backend.set_dtype_map(numpy_dtype_map)


@numpy_backend.set_compile
def f(self, prog, consts, name) -> List[Any]:
    # def f(self, prog, args, consts, name) -> List[Any]:
    def indent(code_line, amount=4):
        spaces = " " * (len(code_line) - len(code_line.lstrip()))
        spaces += " " * amount
        return "\n".join([spaces + line for line in code_line.strip().split("\n")])

    # def process_arg(a):
    #     return self.dtype_map[a] if isinstance(a, sp.core.DType) else a

    safe_builtins = {"math": math, "np": np, "pickle": pickle}

    exec_locals = {}
    env: Dict[sp.Var, Any] = {}
    ncs = 0
    nxs = 0
    nzs = 0
    inb_args = []
    inb_consts = []

    for inb in prog.in_binders:
        if type(inb.aval) is not VoidArray:
            env[inb] = f"c{ncs}"
            inb_consts += [env[inb]]
            ncs += 1
        else:
            env[inb] = f"x{nxs}"
            inb_args += [env[inb]]
            nxs += 1
    code_lines = []
    fn_args_strs = f""
    if inb_consts:
        fn_args_strs += f"{', '.join(inb_consts)}, "
    fn_args_strs += f"{', '.join(inb_args)}"
    code_lines += [f"def {name}({fn_args_strs}):"]
    code_lines += [f"    float32 = np.float32"]  # TODO: cleanup dtype translation
    multiline_op_impl_set = set()
    multiline_op_impl_defs = []
    for instr in prog.instrs:
        in_vals = list_map(lambda x: env[x], instr.inputs)
        for outb in instr.out_binders:
            env[outb] = f"z{nzs}"
            nzs += 1
        out_vals = list_map(lambda z: env[z], instr.out_binders)
        assert not len(out_vals) > 1, "Op with >1 output not supported"
        impl = self.rt.backend.impls[instr.op]
        op_impl_code_lines = inspect.getsourcelines(impl)[0]
        if op_impl_code_lines[0][0] == "@":  # skip decorator
            op_impl_code_lines = op_impl_code_lines[1:]
        args_str = ", ".join(in_vals)
        kwargs_str = ", ".join([f"{k}={v}" for k, v in instr.params.items()])
        if len(op_impl_code_lines) > 2:
            code_line = ""
            if instr.op.name not in multiline_op_impl_set:
                multiline_op_impl_set.add(instr.op.name)
                def_str = op_impl_code_lines[0]
                op_impl_code_lines[
                    0
                ] = f"def {instr.op.name}{def_str[def_str.find('('):]}"
                multiline_op_impl_defs += [op_impl_code_lines]
            # if instr.op.name == "convert":breakpoint()

            code_line = f"    {out_vals[0]} = {instr.op.name}({args_str}, {kwargs_str})"
        else:
            sig = inspect.signature(impl)
            args_strs = [
                k
                for k, v in sig.parameters.items()
                if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and k != "self"
            ]
            op_str = op_impl_code_lines[1].replace("return", "").strip()

            # print(f"orig\t{op_str}")
            for argname, arg in list_zip(args_strs, in_vals):
                mark = "," if argname != args_strs[-1] or len(instr.params) > 0 else ")"
                op_str = op_str.replace(f"{argname}{mark}", f"{arg}{mark}")
                # print(f"{argname}->{arg}\t{op_str}")
            for kwargname, kwarg in instr.params.items():
                if isinstance(kwarg, sp.core.DType):
                    kwarg = self.dtype_map[kwarg]
                op_str = op_str.replace(f"={kwargname}", f"={kwarg}")
                # print(f"{kwargname}=>{kwarg} {op_str}")
            # print(f"mod\t{op_str}\n")

            code_line = f"{out_vals[0]} = {op_str}"
            code_line = indent(code_line, 4)
        code_lines += [code_line]
        # out_vals = instr.op.jit(in_avals, in_vals, **instr.params)

    outs = list_map(lambda y: env[y], prog.outs)
    # ops_code += [f"    outs[0]}"]
    code_lines += [f"    return {', '.join(outs)}{',' if len(outs)==1 else ''}"]
    if len(multiline_op_impl_defs) > 0:
        a = [
            indent(line) for impl_lines in multiline_op_impl_defs for line in impl_lines
        ]
        code_lines = (
            code_lines[0:1]
            + [
                indent(line)
                for impl_lines in multiline_op_impl_defs
                for line in impl_lines
            ]
            + code_lines[1:]
        )

    code = "\n".join(code_lines)
    exec(compile(code, "<string>", "exec"), safe_builtins, exec_locals)
    fn = exec_locals[name]
    return sp.core.JitFn(self.rt, code, fn, consts)


### Op Impls


@numpy_backend.set_impl(ops.convert)
def f(
    x,
    *,
    dtype,
):
    return x.astype(dtype=dtype)


@numpy_backend.set_impl(ops.stop_gradient)
def f(x, *, dtype):
    return x


@numpy_backend.set_impl(ops.neg)
def f(
    x,
):
    return np.negative(x)


@numpy_backend.set_impl(ops.sqrt)
def f(x):
    return np.sqrt(x)


@numpy_backend.set_impl(ops.exp)
def f(x):
    return np.exp(x)


@numpy_backend.set_impl(ops.log)
def f(x):
    return np.log(x)


@numpy_backend.set_impl(ops.sin)
def f(x):
    return np.sin(x)


@numpy_backend.set_impl(ops.add)
def f(x, y):
    return np.add(x, y)


@numpy_backend.set_impl(ops.sub)
def f(x, y):
    return np.subtract(x, y)


@numpy_backend.set_impl(ops.mul)
def f(x, y):
    return np.multiply(x, y)


@numpy_backend.set_impl(ops.div)
def f(x, y):
    return np.divide(x, y)


@numpy_backend.set_impl(ops.equal)
def f(x, y):
    return np.equal(x, y)


@numpy_backend.set_impl(ops.not_equal)
def f(x, y):
    return np.not_equal(x, y)


@numpy_backend.set_impl(ops.maximum)
def f(x, y):
    return np.maximum(x, y)


@numpy_backend.set_impl(ops.sum)
def f(x, *, axes=None, keepdims=False):
    return np.sum(x, axis=axes, keepdims=keepdims)


@numpy_backend.set_impl(ops.max)
def f(x, *, axes=None, keepdims=False):
    return np.max(x, axis=axes, keepdims=keepdims)


@numpy_backend.set_impl(ops.constant)
def f(val, *, dtype=default_dtype):
    return np.array(val, dtype=dtype)


@numpy_backend.set_impl(ops.arange)
def f(*, start, stop, stride, dtype=default_dtype):
    return np.arange(start=start, stop=stop, stride=stride, dtype=dtype)


@numpy_backend.set_impl(ops.full)
def f(*, shape, fill_value, dtype=default_dtype):
    return np.full(shape=shape, fill_value=fill_value, dtype=dtype)


@numpy_backend.set_impl(ops.random_uniform)
def f(*, shape, dtype=default_dtype):
    return np.random.uniform(size=shape).astype(dtype=dtype)


@numpy_backend.set_impl(ops.random_normal)
def f(*, shape, dtype=default_dtype):
    return np.random.normal(loc=np.zeros(shape=shape)).astype(dtype=dtype)


@numpy_backend.set_impl(ops.broadcast)
def f(x, *, shape, axes=None):
    ret = x
    if not axes is None:
        for a in sorted(axes):
            ret = np.expand_dims(ret, a)
    ret = np.broadcast_to(ret, shape)
    return ret


@numpy_backend.set_impl(ops.reshape)
def f(x, *, shape):
    return np.reshape(x, newshape=shape)


@numpy_backend.set_impl(ops.pad)
def f(x, *, lo, hi, interior, value):
    # TODO: implement interior pad
    return np.pad(x, list(zip(lo, hi)), constant_values=value)


@numpy_backend.set_impl(ops.slice)
def f(x, *, starts, limits, strides):
    slices = tuple(slice(s, l, st) for s, l, st in zip(starts, limits, strides))
    return x[slices]


@numpy_backend.set_impl(ops.concatenate)
def f(xs, *, axes):
    return np.concatenate(xs, axes)


@numpy_backend.set_impl(ops.transpose)
def f(x, *, perm):  # NOTE: np.transpose is like torch.permute
    return np.transpose(x, axes=perm)


@numpy_backend.set_impl(ops.flip)
def f(x, *, axes):
    return np.flip(x, axes)
