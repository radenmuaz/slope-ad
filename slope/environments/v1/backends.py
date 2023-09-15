import slope
from slope.environments.v1.operators import operator_set
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

numpy_backend = Backend(
    name="numpy", default_dtype=BaseArray.float32, deps=("numpy as np", "math")
)
numpy_dtype_map = {
    BaseArray.float32: np.dtype("float32"),
    BaseArray.int64: np.dtype("int64"),
    BaseArray.int8: np.dtype("int8"),
    BaseArray.bool: np.dtype("bool"),
}
numpy_backend.set_dtype_map(numpy_dtype_map)

default_dtype = numpy_backend.default_dtype


@numpy_backend.set_numpy_of
def f(self, array):
    return array.buf.val


@numpy_backend.set_device_of
def f(self, array):
    return "cpu"


@numpy_backend.set_compile
def f(self, program, codegen_out, fn_name):
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
    for instruction in program.instructions:
        if instruction.op == slope.core.jit_op:
            continue
        impl = slope.M().backend.impls[instruction.op]
        op_impl_code_lines = inspect.getsourcelines(impl)[0]
        if op_impl_code_lines[0][0] == "@":  # skip decorator line
            op_impl_code_lines = op_impl_code_lines[1:]
        if len(op_impl_code_lines) > 2:
            if instruction.op.name not in multiline_op_impl_set:
                multiline_op_impl_set.add(instruction.op.name)
                def_str = op_impl_code_lines[0]
                op_impl_code_lines[
                    0
                ] = f"def {instruction.op.name}{def_str[def_str.find('('):]}"
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
    code = "\n".join(code_lines).replace("self, ", "")
    exec(compile(code, "<string>", "exec"), self.deps_dict, exec_locals)
    fn = exec_locals[fn_name]
    return fn, code


@numpy_backend.set_codegen
def f(self, program, args) -> List[Any]:
    # codegen is recursive if jit-of-jit happens
    environment: Dict[slope.Var, Any] = {}
    ncs = 0
    nxs = 0
    nzs = 0
    inb_args = []
    inb_consts = []
    affix = (
        f"_d{self.codegen_depth}_i{self.codegen_idx}" if self.codegen_depth != 0 else ""
    )
    for inb in program.in_binders:
        if type(inb.aval) is not VoidArray:
            environment[inb] = f"c{ncs}{affix}"
            inb_consts += [environment[inb]]
            ncs += 1
        else:
            environment[inb] = f"x{nxs}{affix}"
            inb_args += [environment[inb]]
            nxs += 1

    code_lines = []
    for instruction in program.instructions:
        in_vals = list_map(lambda x: environment[x], instruction.inputs)
        in_avals = [x.aval for x in instruction.inputs]
        for outb in instruction.out_binders:
            environment[outb] = f"z{nzs}{affix}"
            nzs += 1
        out_vals = list_map(lambda z: environment[z], instruction.out_binders)

        impl = slope.M().backend.impls[instruction.op]
        args_str = ", ".join(in_vals)
        lhs = (
            f"{out_vals[0] if len(out_vals) == 1 else ', '.join([o for o in out_vals])}"
        )
        if instruction.op is slope.core.jit_op:
            # TODO: generalize interface to other than jit_op
            op_out = impl(in_vals, in_avals, params=instruction.params)
            co = op_out["codegen_out"]
            outs = co["outs"]
            rhs = f"{outs[0] if len(outs) == 1 else ', '.join([o for o in outs])}"
            op_code_lines = co["code_lines"]
            input_lhs = ", ".join((co["inb_args"] + co["inb_consts"]))
            input_code_line = f"{input_lhs} = {args_str}"
            output_code_line = f"{lhs} = {rhs}"
            code_lines += [input_code_line] + op_code_lines + [output_code_line]
            continue
        op_impl_code_lines = inspect.getsourcelines(impl)[0]
        if op_impl_code_lines[0][0] == "@":  # skip decorator line
            op_impl_code_lines = op_impl_code_lines[1:]

        if len(op_impl_code_lines) > 2:
            # kwargs_str = ", ".join([f"{k}={v}" for k, v in instruction.params.items()])
            params = {
                k: v if not isinstance(v, slope.core.DType) else self.dtype_map[v]
                for k, v in instruction.params.items()
            }
            kwargs_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            rhs = f"{instruction.op.name}({args_str}, {kwargs_str})"
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
                mark = (
                    ","
                    if argname != args_strs[-1] or len(instruction.params) > 0
                    else ")"
                )
                rhs = rhs.replace(f"{argname}{mark}", f"{arg}{mark}")
            for kwargname, kwarg in instruction.params.items():
                if isinstance(kwarg, slope.core.DType):
                    kwarg = self.dtype_map[kwarg]
                rhs = rhs.replace(f"={kwargname}", f"={kwarg}")
            code_line = f"{lhs} = {rhs}"
        code_lines += [code_line]

    outs = list_map(lambda y: environment[y], program.outs)
    return dict(
        code_lines=code_lines, inb_args=inb_args, inb_consts=inb_consts, outs=outs
    )


### Operator Impls


@numpy_backend.set_impl(slope.core.jit_op)
def f(self, in_vals, in_avals, *, params):
    program = params["program"]
    self.codegen_depth += 1
    codegen_out = self.codegen(program, in_vals + in_avals)
    self.codegen_idx += 1
    self.codegen_depth -= 1

    return dict(codegen_out=codegen_out)


@numpy_backend.set_impl(operator_set.convert)
def f(self, x, *, dtype):
    ret = x
    return ret.astype(dtype=dtype)


@numpy_backend.set_impl(operator_set.stop_gradient)
def f(self, x, *, dtype):
    return x


@numpy_backend.set_impl(operator_set.neg)
def f(self, x):
    return np.negative(x)


@numpy_backend.set_impl(operator_set.sqrt)
def f(self, x):
    return np.sqrt(x)


@numpy_backend.set_impl(operator_set.exp)
def f(self, x):
    return np.exp(x)


@numpy_backend.set_impl(operator_set.log)
def f(self, x):
    return np.log(x)


@numpy_backend.set_impl(operator_set.sin)
def f(self, x):
    return np.sin(x)


@numpy_backend.set_impl(operator_set.add)
def f(self, x, y):
    return np.add(x, y)


@numpy_backend.set_impl(operator_set.sub)
def f(self, x, y):
    return np.subtract(x, y)


@numpy_backend.set_impl(operator_set.mul)
def f(self, x, y):
    return np.multiply(x, y)


@numpy_backend.set_impl(operator_set.div)
def f(self, x, y):
    return np.divide(x, y)


@numpy_backend.set_impl(operator_set.equal)
def f(self, x, y):
    return np.equal(x, y)


@numpy_backend.set_impl(operator_set.not_equal)
def f(self, x, y):
    return np.not_equal(x, y)


@numpy_backend.set_impl(operator_set.maximum)
def f(self, x, y):
    return np.maximum(x, y)


@numpy_backend.set_impl(operator_set.sum)
def f(self, x, *, axes=None, keepdims=False):
    return np.sum(x, axis=axes, keepdims=keepdims)


@numpy_backend.set_impl(operator_set.max)
def f(self, x, *, axes=None, keepdims=False):
    return np.max(x, axis=axes, keepdims=keepdims)


@numpy_backend.set_impl(operator_set.constant)
def f(self, val, *, dtype=default_dtype):
    return np.array(val, dtype=dtype)


@numpy_backend.set_impl(operator_set.arange)
def f(self, *, start, stop, stride, dtype=default_dtype):
    return np.arange(start=start, stop=stop, stride=stride, dtype=dtype)


@numpy_backend.set_impl(operator_set.full)
def f(self, *, shape, fill_value, dtype=default_dtype):
    return np.full(shape=shape, fill_value=fill_value, dtype=dtype)


@numpy_backend.set_impl(operator_set.random_uniform)
def f(self, *, shape, dtype=default_dtype):
    return np.random.uniform(size=shape).astype(dtype=dtype)


@numpy_backend.set_impl(operator_set.random_normal)
def f(self, *, shape, dtype=default_dtype):
    return np.random.normal(loc=np.zeros(shape=shape)).astype(dtype=dtype)


@numpy_backend.set_impl(operator_set.broadcast_in_dim)
def f(self, x, *, shape, axes=None):
    ret = x
    if not axes is None:
        for a in sorted(axes):
            ret = np.expand_dims(ret, a)
    ret = np.broadcast_to(ret, shape)
    return ret


@numpy_backend.set_impl(operator_set.reshape)
def f(self, x, *, shape):
    return np.reshape(x, newshape=shape)


# @numpy_backend.set_impl(operator_set.pad)
# def f(self, x, *,  pad_width, mode="constant", constant_values=0.0):
#     # TODO: implement interior pad
#     return np.pad(x, pad_width=pad_width, mode=mode, constant_values=constant_values)


@numpy_backend.set_impl(operator_set.pad_hlo)
def f(self, x, *, lo, hi, interior, value):
    # TODO: implement interior pad
    return np.pad(x, list(zip(lo, hi)), constant_values=value)


@numpy_backend.set_impl(operator_set.slice_hlo)
def f(self, x, *, starts, limits, strides):
    slices = tuple(slice(s, l, st) for s, l, st in zip(starts, limits, strides))
    return x[slices]


@numpy_backend.set_impl(operator_set.concatenate)
def f(self, xs, *, axes):
    return np.concatenate(xs, axes)


@numpy_backend.set_impl(operator_set.transpose)
def f(self, x, *, perm):  # NOTE: np.transpose is like torch.permute
    return np.transpose(x, axes=perm)


@numpy_backend.set_impl(operator_set.flip)
def f(self, x, *, axes):
    return np.flip(x, axes)


#   subc = subc.build(xoperator_set.Tuple(subc, outs))
#   return destructure_tuple(c, xoperator_set.Call(c, subc, in_vals))


# def direct_translation(op, c, in_avals, in_vals):
#     del c, in_avals
#     return [op(*in_vals)]
