import slope
from slope.core import Compiler, Tensor, TensorBuffer, Typecheckor, list_map
import os

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
compiler = Compiler(name="example_numpy", default_dtype=Tensor.dtype_names[slope.SLOPE_DTYPE])
compiler.set_dtype_map(
    {
        Tensor.float32: np.dtype("float32"),
        Tensor.int64: np.dtype("int64"),
        Tensor.int32: np.dtype("int32"),
        Tensor.int8: np.dtype("int8"),
        Tensor.bool: np.dtype("bool"),
    }
)


@compiler.set_method
def from_numpy(self, val, dtype=compiler.default_dtype_value):
    val = np.array(val, dtype=compiler.dtype_map[dtype])
    return Tensor(TensorBuffer(val))


@compiler.set_method
def numpy_of(self, tensor):
    return tensor.buf.val


@compiler.set_method
def device_of(self, tensor):
    return "cpu"


@compiler.set_method
def shape_of(self, tensor):
    return tensor.buf.val.shape


@compiler.set_method
def dtype_of(self, tensor):
    return self.dtype_map_inv[tensor.buf.val.dtype]


@compiler.set_method
def export(self, jit_object: slope.core.JitObject, output_path, *args, **kwargs):
    code = jit_object.code
    os.makedirs(output_path, exist_ok=True)
    consts_dir_path = os.path.join(output_path, "consts")
    os.makedirs(consts_dir_path, exist_ok=True)
    in_binders = jit_object.codegen_out["in_binders"]
    outs = jit_object.codegen_out["outs"]
    num_consts = jit_object.program.num_consts
    load_consts_code = ""
    for i in range(num_consts):
        const_name = in_binders[i]["name"]
        const_path = os.path.join(consts_dir_path, f"{const_name}.npy")
        load_consts_code += f"""{const_name} = np.load(os.path.join(consts_dir_path, "{const_name}.npy"))\n"""
        np.save(const_path, in_binders[i]["type"].numpy())
    input_args_code = ", ".join(ib["name"] for ib in in_binders[num_consts:])
    args_code = ", ".join(ib["name"] for ib in in_binders)
    input_arg_names = [ib["name"] for ib in in_binders[num_consts:]]
    input_arg_names_str = ", ".join(input_arg_names)
    outs_names = [out["name"] for out in outs]

    test_input_code = ""
    for i in range(num_consts, len(in_binders)):
        input_name = in_binders[i]["name"]
        input_shape = in_binders[i]["type"].shape
        dtype = in_binders[i]["type"].dtype
        input_dtype = ("np." + dtype.numpy.__name__) if dtype is not Tensor.bool else "bool"
        test_input_code += f"""    {input_name} = np.ones({input_shape}, dtype={input_dtype})\n"""

    module_path = os.path.join(output_path, "__init__.py")
    module_code = f"""import numpy as np
import os
root_path = os.path.dirname(__file__)
consts_dir_path =  os.path.join(root_path, "consts")
{load_consts_code}
{code}

input_arg_names = {input_arg_names}
out_names = {outs_names}

def run({input_args_code}):
    return main({args_code})

if __name__ == "__main__":
{test_input_code}
    for inp_name, inp in zip(input_arg_names, ({input_arg_names_str})):
        print(f"{{inp_name}} = ")
        print(inp)
        print(f"dtype: {{inp.dtype}}")
        print(f"shape: {{inp.shape}}")
        print()

    outs = run({input_arg_names_str})

    print("outputs:")
    for out_name, out in zip(out_names, outs):
        print(f"{{out_name}} = ")
        print(out)
        print(f"dtype: {{out.dtype}}")
        print(f"shape: {{out.shape}}")
        print()
"""
    with open(module_path, "w") as f:
        f.write(module_code)
        slope.dblog(module_code, enable=slope.LOG_JIT)


@compiler.set_method
def compile(self, codegen_out):
    deps_dict = dict()
    deps_dict["numpy"] = importlib.import_module("numpy")
    deps_dict["np"] = deps_dict["numpy"]
    deps_dict["math"] = importlib.import_module("math")
    code_lines = codegen_out["code_lines"]
    exec_locals = dict()
    code = "\n".join(code_lines)
    exec(compile_py(code, "<string>", "exec"), deps_dict, exec_locals)
    fn = exec_locals["main"]
    return fn, code


@compiler.set_method
def codegen(self, program, args, *, fn_name: str = "main", fn_defs=dict()) -> List[Any]:
    if fn_name == "main":
        assert not hasattr(self, "fn_count")
        self.fn_count = 0
        assert not hasattr(self, "depth")
        self.depth = 0

    def indent(code, amount):
        spaces = " " * (len(code) - len(code.lstrip()))
        spaces += " " * amount
        return "\n".join([spaces + line for line in code.strip().split("\n")])

    # codegen is recursive if jit-of-jit happens
    backend: Dict[slope.Var, Any] = {}
    il1 = (self.depth + 1) * 4
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
            self.depth += 1
            rhs, fn_defs = self.impls[instruction.op](program, args, instruction, in_vals, fn_defs)
            self.depth -= 1
            impl_code = f"{lhs} = {rhs}"
        else:
            impl_code = self.impls[instruction.op](*in_vals, **instruction.params)
            if len(out_vals) == 1:
                impl_code = impl_code.replace("ret", out_vals[0])
            else:
                raise NotImplementedError
        for np_dtype in self.dtype_map.values():  # fix dtype kwargs not having 'np.' prefix
            impl_code = impl_code.replace(
                np_dtype.name, "bool" if np_dtype is np.dtype("bool") else f"np.{np_dtype.name}"
            )

        for impl_code_line in impl_code.split("\n"):  # handle multi-line code
            body_code_lines += [indent(impl_code_line, il1)]

    in_binders = list_map(lambda x: backend[x], program.in_binders)
    arg_type_strs = [i["name"] for i in in_binders]
    # arg_type_asserts = [
    #     f"{self.dtype_map[i['type'].dtype]}[{repr(list(i['type'].shape))[1:-1]}] {i['name']}" for i in in_binders
    # ]
    fn_args_str = ", ".join(arg_type_strs)

    outs = list_map(lambda x: backend[x], program.outs)  # TODO: input that is output should has identity op
    out_type_strs = [o["name"] for o in outs]
    # out_type_asserts = [
    #     f"{self.dtype_map[o['type'].dtype]}[{repr(list(o['type'].shape))[1:-1]}] {o['name']}" for o in outs
    # ]

    head_code_lines = []
    head_code_lines += [f"def {fn_name} ({fn_args_str}):"]
    out_type_str = ", ".join(out_type_strs) + ("," if len(outs) == 1 else "")
    return_line = [indent(f"return {out_type_str}", il1)]

    functions_code_lines = []
    for op, fn_def_code_lines in fn_defs.items():
        # functions_code_lines += fn_def_code_lines
        functions_code_lines += fn_def_code_lines

    code_lines = head_code_lines + [indent(l, il1) for l in functions_code_lines] + body_code_lines + return_line
    slope.dblog(f"\n-- {program.name} codegen:\n\n" + "\n".join(code_lines) + "\n\n==\n", enable=slope.LOG_JIT)

    if fn_name == "main":
        del self.fn_count
        del self.depth
    assert len(outs) == len(program.outs)
    return dict(code_lines=code_lines, fn_defs=fn_defs, in_binders=in_binders, outs=outs)


### Operator Impls

compiler.set_impl(operator_set.cast)(lambda self, x, *, dtype: f"ret = {x}.astype(dtype={dtype})")
compiler.set_impl(operator_set.stop_gradient)(lambda self, x: f"ret = {x}")
compiler.set_impl(operator_set.neg)(lambda self, x: f"ret = np.negative({x})")
compiler.set_impl(operator_set.sqrt)(lambda self, x: f"ret = np.sqrt({x})")
compiler.set_impl(operator_set.exp)(lambda self, x: f"ret = np.exp({x})")
compiler.set_impl(operator_set.log)(lambda self, x: f"ret = np.log({x})")
compiler.set_impl(operator_set.sin)(lambda self, x: f"ret = np.sin({x})")
compiler.set_impl(operator_set.add)(lambda self, x, w: f"ret = np.add({x}, {w})")
compiler.set_impl(operator_set.sub)(lambda self, x, w: f"ret = np.subtract({x}, {w})")
compiler.set_impl(operator_set.mul)(lambda self, x, w: f"ret = np.multiply({x}, {w})")
compiler.set_impl(operator_set.div)(lambda self, x, w: f"ret = np.divide({x}, {w})")
compiler.set_impl(operator_set.pow)(lambda self, x, w: f"ret = np.power({x}, {w})")
compiler.set_impl(operator_set.invert)(lambda self, x: f"ret = np.invert({x})")
compiler.set_impl(operator_set.equal)(lambda self, x, w: f"ret = np.equal({x}, {w})")
compiler.set_impl(operator_set.maximum)(lambda self, x, w: f"ret = np.maximum({x}, {w})")
compiler.set_impl(operator_set.sum)(
    lambda self, x, *, axes, keepdims: f"ret = np.sum({x}, axis={axes}, keepdims={keepdims})"
)
compiler.set_impl(operator_set.max)(
    lambda self, x, *, axes, keepdims: f"ret = np.max({x}, axis={axes}, keepdims={keepdims})"
)
compiler.set_impl(operator_set.arange)(
    lambda self, *, start, stop, stride, dtype: f"ret = np.arange(start={start}, stop={stop}, step={stride}, dtype={dtype})"
)
compiler.set_impl(operator_set.full)(
    lambda self, *, shape, fill_value, dtype: f"ret = np.full(shape={shape}, fill_value={fill_value}, dtype={dtype})"
)

compiler.set_impl(operator_set.random_uniform)(
    lambda self, *, shape, dtype: (
        f"ret = {'np.array(' if shape == () else ''}np.random.uniform(low=np.zeros(shape={shape})){')' if shape == () else ''}.astype(dtype={dtype})"
    )
)
compiler.set_impl(operator_set.random_normal)(
    lambda self, *, shape, dtype: (
        f"ret = {'np.array(' if shape == () else ''}np.random.normal(loc=np.zeros(shape={shape})){')' if shape == () else ''}.astype(dtype={dtype})"
    )
)
compiler.set_impl(operator_set.expand)(lambda self, x, *, shape: f"ret = np.broadcast_to({x}, shape={shape})")

compiler.set_impl(operator_set.reshape)(lambda self, x, *, shape: f"ret = np.reshape({x}, newshape={shape})")


@compiler.set_impl(operator_set.pad)
def pad_impl(self, x, *, padding, mode, value):
    pad_width = [(lo, hi) for lo, hi in zip(padding[0::2], padding[1::2])]
    return f"ret = np.pad({x}, {pad_width}, constant_values={value})"


compiler.set_impl(operator_set.slice)(
    lambda self, x, *, starts, limits, strides: f"ret = {x}[tuple(slice(s, l, st) for s, l, st in zip({starts}, {limits}, {strides}))]"
)

compiler.set_impl(operator_set.cat)(lambda self, *xs, axis: f"ret = np.cat({xs}, axis={axis})")
compiler.set_impl(operator_set.permute)(lambda self, x, *, perm: f"ret = np.transpose({x}, axes={perm})")
compiler.set_impl(operator_set.flip)(lambda self, x, *, axes: f"ret = np.flip({x}, axis={axes})")


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
    rhs = f"{jit_name}({args_str})"
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
    rhs = f"{proc_name}({args_str})"
    return rhs, fn_defs
