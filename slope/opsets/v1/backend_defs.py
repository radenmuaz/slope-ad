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

numpy_backend = Backend(name="numpy", deps=("numpy as np", "math"))
numpy_dtype_map = {
    BaseArray.float32: np.dtype("float32"),
    BaseArray.int64: np.dtype("int64"),
    BaseArray.int8: np.dtype("int8"),
    BaseArray.bool: bool,
}
default_dtype = BaseArray.default_dtype
numpy_backend.set_dtype_map(numpy_dtype_map)


@numpy_backend.set_compile
def f(self, prog, codegen_out, fn_name):
    def indent(code_line, amount=4):
        spaces = " " * (len(code_line) - len(code_line.lstrip()))
        spaces += " " * amount
        return "\n".join([spaces + line for line in code_line.strip().split("\n")])

    code_lines = []

    fn_args_strs = f""
    inb_args = codegen_out["inb_args"]
    inb_consts = codegen_out["inb_consts"]
    if inb_consts:
        fn_args_strs += f"{', '.join(inb_consts)}, "
    fn_args_strs += f"{', '.join(inb_args)}"
    code_lines += [f"def {fn_name}({fn_args_strs}):"]

    code_lines += [indent(f"float32 = np.float32")]  # TODO: cleanup dtype translation

    code_lines += [indent(cl) for cl in codegen_out["code_lines"]]

    outs = codegen_out["outs"]
    code_lines += [indent(f"return {', '.join(outs)}{',' if len(outs)==1 else ''}")]

    multiline_op_impl_set = set()
    multiline_op_impl_defs = []
    for instr in prog.instrs:
        impl = self.rt.backend.impls[instr.op]
        op_impl_code_lines = inspect.getsourcelines(impl)[0]
        if op_impl_code_lines[0][0] == "@":  # skip decorator line
            op_impl_code_lines = op_impl_code_lines[1:]
        if len(op_impl_code_lines) > 2:
            if instr.op.name not in multiline_op_impl_set:
                multiline_op_impl_set.add(instr.op.name)
                def_str = op_impl_code_lines[0]
                op_impl_code_lines[
                    0
                ] = f"def {instr.op.name}{def_str[def_str.find('('):]}"
                multiline_op_impl_defs += [op_impl_code_lines]

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

    exec_locals = {}
    code = "\n".join(code_lines)
    print(code)
    exec(compile(code, "<string>", "exec"), self.deps_dict, exec_locals)
    fn = exec_locals[fn_name]
    return fn


@numpy_backend.set_codegen
def f(self, prog, args, codegen_idx=0, codegen_depth=0) -> List[Any]:
    codegen_idx = codegen_idx
    codegen_depth = codegen_depth
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
    for instr in prog.instrs:
        in_vals = list_map(lambda x: env[x], instr.inputs)
        in_avals = [x.aval for x in instr.inputs]
        for outb in instr.out_binders:
            env[outb] = f"z{nzs}"
            nzs += 1
        out_vals = list_map(lambda z: env[z], instr.out_binders)

        impl = self.rt.backend.impls[instr.op]
        if instr.op is sp.core.jit_op:
            # TODO: generalize interface to other than jit_op
            codegen_idx += 1
            jit_op_code_lines = impl(
                args,
                params=instr.params,
                backend=self,
                codegen_idx=codegen_idx,
                codegen_depth=codegen_depth + 1,
            )
        op_impl_code_lines = inspect.getsourcelines(impl)[0]
        if op_impl_code_lines[0][0] == "@":  # skip decorator line
            op_impl_code_lines = op_impl_code_lines[1:]

        args_str = ", ".join(in_vals)
        lhs = (
            f"{out_vals[0] if len(out_vals) == 1 else ', '.join([o for o in out_vals])}"
        )
        if len(op_impl_code_lines) > 2:
            kwargs_str = ", ".join([f"{k}={v}" for k, v in instr.params.items()])
            rhs = f"{instr.op.name}({args_str}, {kwargs_str})"
            code_line = f"{lhs} = {rhs}"
        else:
            sig = inspect.signature(impl)
            args_strs = [
                k
                for k, v in sig.parameters.items()
                if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and k != "self"
            ]
            rhs = op_impl_code_lines[1].replace("return", "").strip()

            for argname, arg in list_zip(args_strs, in_vals):
                mark = "," if argname != args_strs[-1] or len(instr.params) > 0 else ")"
                rhs = rhs.replace(f"{argname}{mark}", f"{arg}{mark}")
            for kwargname, kwarg in instr.params.items():
                if isinstance(kwarg, sp.core.DType):
                    kwarg = self.dtype_map[kwarg]
                rhs = rhs.replace(f"={kwargname}", f"={kwarg}")
            code_line = f"{lhs} = {rhs}"
        code_lines += [code_line]
        # out_vals = instr.op.jit(in_avals, in_vals, **instr.params)

    outs = list_map(lambda y: env[y], prog.outs)
    # code_lines += [f"out = ({', '.join(outs)}{',' if len(outs)==1 else ''})"]
    return dict(
        code_lines=code_lines, inb_args=inb_args, inb_consts=inb_consts, outs=outs
    )


### Op Impls


@numpy_backend.set_impl(sp.core.jit_op)
def f(self, args, *, params, backend, codegen_idx, codegen_depth):
    prog = params["prog"]
    out_vals = prog.out_vals
    codegen_out = backend.codegen(
        prog, args, codegen_idx=codegen_idx, codegen_depth=codegen_depth
    )
    code_lines = codegen_out["code_lines"]
    outs = codegen_out["outs"]
    code_lines += [f"return {', '.join(outs)}{',' if len(outs)==1 else ''}"]
    lhs = f"{out_vals[0] if len(out_vals) == 1 else ', '.join([o for o in out_vals])}"

    return code_lines


@numpy_backend.set_impl(ops.convert)
def f(
    self,
    x,
    *,
    dtype,
):
    return x.astype(dtype=dtype)


@numpy_backend.set_impl(ops.stop_gradient)
def f(self, x, *, dtype):
    return x


@numpy_backend.set_impl(ops.neg)
def f(
    self,
    x,
):
    return np.negative(x)


@numpy_backend.set_impl(ops.sqrt)
def f(self, x):
    return np.sqrt(x)


@numpy_backend.set_impl(ops.exp)
def f(self, x):
    return np.exp(x)


@numpy_backend.set_impl(ops.log)
def f(self, x):
    return np.log(x)


@numpy_backend.set_impl(ops.sin)
def f(self, x):
    return np.sin(x)


@numpy_backend.set_impl(ops.add)
def f(self, x, y):
    return np.add(x, y)


@numpy_backend.set_impl(ops.sub)
def f(self, x, y):
    return np.subtract(x, y)


@numpy_backend.set_impl(ops.mul)
def f(self, x, y):
    return np.multiply(x, y)


@numpy_backend.set_impl(ops.div)
def f(self, x, y):
    return np.divide(x, y)


@numpy_backend.set_impl(ops.equal)
def f(self, x, y):
    return np.equal(x, y)


@numpy_backend.set_impl(ops.not_equal)
def f(self, x, y):
    return np.not_equal(x, y)


@numpy_backend.set_impl(ops.maximum)
def f(self, x, y):
    return np.maximum(x, y)


@numpy_backend.set_impl(ops.sum)
def f(self, x, *, axes=None, keepdims=False):
    return np.sum(x, axis=axes, keepdims=keepdims)


@numpy_backend.set_impl(ops.max)
def f(self, x, *, axes=None, keepdims=False):
    return np.max(x, axis=axes, keepdims=keepdims)


@numpy_backend.set_impl(ops.constant)
def f(self, val, *, dtype=default_dtype):
    return np.array(val, dtype=dtype)


@numpy_backend.set_impl(ops.arange)
def f(self, *, start, stop, stride, dtype=default_dtype):
    return np.arange(start=start, stop=stop, stride=stride, dtype=dtype)


@numpy_backend.set_impl(ops.full)
def f(self, *, shape, fill_value, dtype=default_dtype):
    return np.full(shape=shape, fill_value=fill_value, dtype=dtype)


@numpy_backend.set_impl(ops.random_uniform)
def f(self, *, shape, dtype=default_dtype):
    return np.random.uniform(size=shape).astype(dtype=dtype)


@numpy_backend.set_impl(ops.random_normal)
def f(self, *, shape, dtype=default_dtype):
    return np.random.normal(loc=np.zeros(shape=shape)).astype(dtype=dtype)


@numpy_backend.set_impl(ops.broadcast)
def f(self, x, *, shape, axes=None):
    ret = x
    if not axes is None:
        for a in sorted(axes):
            ret = np.expand_dims(ret, a)
    ret = np.broadcast_to(ret, shape)
    return ret


@numpy_backend.set_impl(ops.reshape)
def f(self, x, *, shape):
    return np.reshape(x, newshape=shape)


@numpy_backend.set_impl(ops.pad)
def f(self, x, *, lo, hi, interior, value):
    # TODO: implement interior pad
    return np.pad(x, list(zip(lo, hi)), constant_values=value)


@numpy_backend.set_impl(ops.slice)
def f(self, x, *, starts, limits, strides):
    slices = tuple(slice(s, l, st) for s, l, st in zip(starts, limits, strides))
    return x[slices]


@numpy_backend.set_impl(ops.concatenate)
def f(self, xs, *, axes):
    return np.concatenate(xs, axes)


@numpy_backend.set_impl(ops.transpose)
def f(self, x, *, perm):  # NOTE: np.transpose is like torch.permute
    return np.transpose(x, axes=perm)


@numpy_backend.set_impl(ops.flip)
def f(self, x, *, axes):
    return np.flip(x, axes)


#   subc = subc.build(xops.Tuple(subc, outs))
#   return destructure_tuple(c, xops.Call(c, subc, in_vals))


# def direct_translation(op, c, in_avals, in_vals):
#     del c, in_avals
#     return [op(*in_vals)]
