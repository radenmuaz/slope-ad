import slope as sp
import numpy as np
from typing import (
    List,
    Dict,
    Any,
)
import pickle
import math
import inspect

numpy_backend = sp.Backend("numpy")
for dtype in [bool, int, float, np.ndarray, np.float64, np.float32]:
    numpy_backend.set_input_handler(dtype, np.asarray)
numpy_dtype_map = {
    sp.BaseArray.float32: np.float32,
    sp.BaseArray.int64: np.int64,
    sp.BaseArray.int8: np.int8,
    sp.BaseArray.bool: bool,
}

default_dtype = numpy_dtype_map[sp.BaseArray.default_dtype]
numpy_backend.set_dtype_map(numpy_dtype_map)


@numpy_backend.set_compile
def f(self, prog, consts, in_avals, name) -> List[Any]:
    safe_builtins = {"math": math, "np": np, "pickle": pickle}

    exec_locals = {}
    env: Dict[sp.Var, Any] = {}
    ncs = 0
    nxs = 0
    nzs = 0
    inb_args = []
    inb_consts = []

    for inb in prog.in_binders:
        if type(inb.aval) is not sp.ArrayShape:
            env[inb] = f"c{ncs}"
            inb_consts += [env[inb]]
            ncs += 1
        else:
            env[inb] = f"x{nxs}"
            inb_args += [env[inb]]
            nxs += 1
    code_lines = []
    code_lines += [f"def {name}({', '.join(inb_args)}):"]
    for inb_const, const in sp.list_zip(inb_consts, consts):
        code_lines += [f"    {inb_const} = pickle.loads({pickle.dumps(const.val)})"]
    multiline_op_impl_set = set()
    multiline_op_impl_defs = []
    for eqn in prog.instrs:
        in_vals = sp.list_map(lambda x: env[x], eqn.inputs)
        for outb in eqn.out_binders:
            env[outb] = f"z{nzs}"
            nzs += 1
        out_vals = sp.list_map(lambda z: env[z], eqn.out_binders)
        assert not len(out_vals) > 1, "Op with >1 output not supported"
        # op_ir = self.op_impls[eqn.op.name].ir(*in_vals, **eqn.params, ret=out_vals[0])
        impl = sp.RT.ops[eqn.op.name].impls[self.name]
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
            for argname, arg in sp.list_zip(argspec.args, in_vals):
                op_str.replace(argname, arg)
            for kwargname, kwarg in eqn.params.items():
                op_str.replace(kwargname, kwarg)
            code_line = f"{out_vals[0]} = {op_str}"

        code_line = "\n".join(["    " + line for line in code_line.strip().split("\n")])
        code_lines += [code_line]
        # out_vals = eqn.op.jit(in_avals, in_vals, **eqn.params)

    outs = sp.list_map(lambda y: env[y], prog.outs)
    # ops_code += [f"    outs[0]}"]
    code_lines += [f"    return {', '.join(outs)}{',' if len(outs)==1 else ''}"]
    code_lines = multiline_op_impl_defs + code_lines
    code = "\n".join(code_lines)
    exec(compile(code, "<string>", "exec"), safe_builtins, exec_locals)
    fn = exec_locals[name]
    # exec('\n'.join(ops_code), safe_builtins, exec_locals)
    return sp.JitFn(code, fn)


### Op Impls


@numpy_backend.set_impl(sp.ops.convert)
def f(
    x,
    dtype,
):
    return x.astype(dtype)


@numpy_backend.set_impl(sp.ops.stop_gradient)
def f(x, dtype):
    return x


@numpy_backend.set_impl(sp.ops.neg)
def f(
    x,
):
    return np.negative(x)


@numpy_backend.set_impl(sp.ops.sqrt)
def f(x):
    return np.sqrt(x)


@numpy_backend.set_impl(sp.ops.exp)
def f(x):
    return np.exp(x)


@numpy_backend.set_impl(sp.ops.log)
def f(x):
    return np.log(x)


@numpy_backend.set_impl(sp.ops.add)
def f(x, y):
    return np.add(x, y)


@numpy_backend.set_impl(sp.ops.sub)
def f(x, y):
    return np.subtract(x, y)


@numpy_backend.set_impl(sp.ops.mul)
def f(x, y):
    return np.multiply(x, y)


@numpy_backend.set_impl(sp.ops.div)
def f(x, y):
    return np.divide(x, y)


@numpy_backend.set_impl(sp.ops.equal)
def f(x, y):
    return np.equal(x, y)


@numpy_backend.set_impl(sp.ops.not_equal)
def f(x, y):
    return np.not_equal(x, y)


@numpy_backend.set_impl(sp.ops.maximum)
def f(x, y):
    return np.maximum(x, y)


@numpy_backend.set_impl(sp.ops.sum)
def f(x, axes=None, keepdims=False):
    return np.sum(x, axis=axes, keepdims=keepdims)


@numpy_backend.set_impl(sp.ops.max)
def f(x, axes=None, keepdims=False):
    return np.max(x, axis=axes, keepdims=keepdims)


@numpy_backend.set_impl(sp.ops.constant)
def f(val, dtype=default_dtype):
    return np.array(val, dtype=dtype)


@numpy_backend.set_impl(sp.ops.arange)
def f(start, stop, stride, dtype=default_dtype):
    return np.arange(start, stop, stride, dtype=dtype)


@numpy_backend.set_impl(sp.ops.full)
def f(shape, fill_value, dtype=default_dtype):
    return np.full(shape, fill_value=fill_value, dtype=dtype)


@numpy_backend.set_impl(sp.ops.random_uniform)
def f(shape, dtype=default_dtype):
    return np.random.uniform(size=shape).astype(dtype)


@numpy_backend.set_impl(sp.ops.random_normal)
def f(shape, dtype=default_dtype):
    return np.random.normal(loc=np.zeros(shape)).astype(dtype)


@numpy_backend.set_impl(sp.ops.broadcast)
def f(x, shape, axes=None):
    ret = x
    if not axes is None:
        for a in sorted(axes):
            ret = np.expand_dims(ret, a)
    ret = np.broadcast_to(ret, shape)
    return ret


@numpy_backend.set_impl(sp.ops.reshape)
def f(x, shape):
    return np.reshape(x, shape)


@numpy_backend.set_impl(sp.ops.pad)
def f(x, lo, hi, interior, value):
    # TODO: implement interior pad
    return np.pad({x}, list(zip(lo, hi)), constant_values={value})


@numpy_backend.set_impl(sp.ops.slice)
def f(x, starts, limits, strides):
    return x[[slice(s, l, st) for s, l, st in zip(starts, limits, strides)]]


@numpy_backend.set_impl(sp.ops.concatenate)
def f(xs, axes):
    return np.concatenate(xs, axes)


@numpy_backend.set_impl(sp.ops.transpose)
def f(x, axes):  # NOTE: np.transpose is like torch.permute
    return np.transpose(x, axes)


@numpy_backend.set_impl(sp.ops.flip)
def f(x, axes):
    return np.flip(x, axes)