import slope
from slope.core import (
    Compiler,
    Tensor,
    Typecheckor,
    list_map
)

import math
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Sequence, Union, Iterator, Callable
from collections import defaultdict
from operators import operator_set
sum_py = sum
max_py = max
abs_py = abs
slice_py = slice

compile_py = compile
compiler = Compiler(name="example", default_dtype=Tensor.float32, default_device=slope.SLOPE_DEVICE)
compiler.set_dtype_map(
    {
        Tensor.float32: "float",
        Tensor.uint8: "uint8",
        Tensor.int8: "int8",
        Tensor.bool: "bool",
        Tensor.int32: "int32",
        Tensor.int64: "int64",
        Tensor.float16: "float16",
    }
)

@compiler.set_method
def from_numpy(self, val, dtype=compiler.default_dtype_value, device=compiler.default_device):
    raise NotImplementedError

@compiler.set_method
def numpy_of(self, tensor):
    raise NotImplementedError


@compiler.set_method
def device_of(self, tensor):
    raise NotImplementedError


@compiler.set_method
def shape_of(self, tensor):
    raise NotImplementedError


@compiler.set_method
def dtype_of(self, tensor):
    raise NotImplementedError


@compiler.set_method
def export(self, jit_object: slope.core.JitObject, output_path, *args, **kwargs):
    with open(output_path, "w") as f:
        f.write("test")


@compiler.set_method
def compile(self, codegen_out):
    code_lines = codegen_out["code_lines"]
    code = "\n".join(code_lines)
    def fn(*args, **kwargs):
        pass

    return fn, code


@compiler.set_method
def codegen(self, program, args, *, fn_name: str = "main", fn_defs=dict()) -> List[Any]:
    if fn_name == "main":
        assert not hasattr(self, "fn_count")
        self.fn_count = 0

    def indent(code, amount):
        spaces = " " * (len(code) - len(code.lstrip()))
        spaces += " " * amount
        return "\n".join([spaces + line for line in code.strip().split("\n")])

    # codegen is recursive if jit-of-jit happens
    backend: Dict[slope.Var, Any] = {}
    il1 = 4  # indent length
    body_code_lines = []

    for inb in program.in_binders:
        prefix = "x" if type(inb.aval) is Typecheckor else "c"
        idx = sum_py([1 if v["name"][0] == prefix else 0 for v in backend.values()])
        backend[inb] = dict(name=f"{prefix}{idx}", type=inb.aval)

    for instruction in program.instructions:
        if len(instruction.out_binders) == 0:  # skip codegen for function returns nothing
            continue
        in_vals = list_map(lambda x: backend[x]["name"], instruction.inputs)
        for outb in instruction.out_binders:
            prefix = "y" if outb in program.outs else "z"
            idx = sum_py([1 if v["name"][0] == prefix else 0 for v in backend.values()])
            backend[outb] = dict(name=f"{prefix}{idx}", type=outb.aval)

        out_vals = list_map(lambda z: backend[z]["name"], instruction.out_binders)
        if instruction.op.op_type is slope.core.OperatorType.Meta:
            lhs = ", ".join(out_vals)
            rhs, fn_defs = self.impls[instruction.op](program, args, instruction, in_vals, fn_defs)
            impl_code = f"{lhs} = {rhs}"
        else:
            impl_code = self.impls[instruction.op](*in_vals, **instruction.params)
            if len(out_vals) == 1:
                impl_code = impl_code.replace("ret", out_vals[0])
            else:
                raise NotImplementedError
        for impl_code_line in impl_code.split("\n"):  # handle multi-line code
            body_code_lines += [indent(impl_code_line, il1)]

    # inb_consts = [v for v in backend.values() if "c" in v["name"]]
    # const_type_strs = [f"{self.dtype_map[c['type'].dtype]}[{repr(c['type'].shape)[1:-1]}] {c['name']}" for c in inb_consts]

    in_binders = list_map(lambda x: backend[x], program.in_binders)
    arg_type_strs = [
        f"{self.dtype_map[i['type'].dtype]}[{repr(list(i['type'].shape))[1:-1]}] {i['name']}" for i in in_binders
    ]
    fn_args_str = ", ".join(arg_type_strs)

    outs = list_map(lambda x: backend[x], program.outs)  # TODO: input that is output should has identity op
    out_type_strs = [
        f"{self.dtype_map[o['type'].dtype]}[{repr(list(o['type'].shape))[1:-1]}] {o['name']}" for o in outs
    ]
    out_type_str = ", ".join(out_type_strs)

    functions_code_lines = []
    for op, fn_def_code_lines in fn_defs.items():
        functions_code_lines +=  + fn_def_code_lines
    code_lines = body_code_lines + functions_code_lines
    slope.dblog(f"\n---- {program.name} codegen:\n\n" + "\n".join(code_lines) + "\n\n===============\n", enable=slope.LOG_JIT)

    if fn_name == "main":
        del self.fn_count
    assert len(outs) == len(program.outs)
    return dict(code_lines=code_lines, fn_defs=fn_defs, in_binders=in_binders, outs=outs)


### Operator Impls

compiler.set_impl(operator_set.stop_gradient)(lambda self, x, *, dtype: f"ret = Identity({x})")
compiler.set_impl(operator_set.neg)(lambda self, x: f"ret =  Neg({x})")
compiler.set_impl(operator_set.sqrt)(lambda self, x: f"ret = Sqrt({x})")
compiler.set_impl(operator_set.exp)(lambda self, x: f"ret = Exp({x})")
compiler.set_impl(operator_set.log)(lambda self, x: f"ret = Log({x})")
compiler.set_impl(operator_set.sin)(lambda self, x: f"ret = Sin({x})")
compiler.set_impl(operator_set.add)(lambda self, x, w: f"ret = Add({x}, {w})")
compiler.set_impl(operator_set.sub)(lambda self, x, w: f"ret = Sub({x}, {w})")
compiler.set_impl(operator_set.mul)(lambda self, x, w: f"ret = Mul({x}, {w})")
compiler.set_impl(operator_set.div)(lambda self, x, w: f"ret = Div({x}, {w})")
compiler.set_impl(operator_set.invert)(lambda self, x: f"ret = Not({x})")
compiler.set_impl(operator_set.equal)(lambda self, x, w: f"ret = Equal({x}, {w})")
compiler.set_impl(operator_set.maximum)(lambda self, x, w: f"ret = Max({x}, {w})")
compiler.set_impl(operator_set.matmul)(lambda self, x, w: f"ret = MatMul({x}, {w})")


@compiler.set_impl(operator_set.sum)
def sum_impl(self, x, *, axes, keepdims):
    return f"""
ret_axes = Constant <value = int64[{len(axes)}]  {{ {repr(axes)[1:(-1 if len(axes) > 1 else -2)]} }} >()
ret = ReduceSum<keepdims={int(keepdims)}> ({x}, ret_axes)
"""


@compiler.set_impl(slope.core.jit_op)
def jit_op_impl(self, program, args, instruction, in_vals, fn_defs):
    jit_program = instruction.params["program"]
    jit_name = f"{program.name}"
    jit_codegen_out = self.codegen(
        jit_program,
        args,
        fn_name=jit_name,
        fn_defs=fn_defs,
    )
    assert jit_name not in fn_defs.keys()
    fn_defs[jit_name] = jit_codegen_out["code_lines"]
    fn_defs = {**fn_defs, **jit_codegen_out["fn_defs"]}
    args_str = ", ".join(in_vals)
    rhs = f"slope.{jit_name}({args_str})"
    return rhs, fn_defs


@compiler.set_impl(slope.core.procedure_op)
def procedure_op_impl(self, program, args, instruction, in_vals, fn_defs):
    proc_program = instruction.params["program"]
    proc_name = f"{proc_program.name}_{self.fn_count}"
    self.fn_count += 1
    proc_codegen_out = self.codegen(
        proc_program,
        args,
        fn_name=proc_name,
        fn_defs=fn_defs,
    )
    fn_defs[proc_name] = proc_codegen_out["code_lines"]
    fn_defs = {**fn_defs, **proc_codegen_out["fn_defs"]}
    args_str = ", ".join(in_vals)
    rhs = f"slope.{proc_name}({args_str})"
    return rhs, fn_defs
