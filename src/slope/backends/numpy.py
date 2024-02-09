import slope
import slope.core
from slope.core import (
    Backend,
    Operator,
    Instruction,
    Program,
    MetaOperator,
    ProcedureSet,
    Tensor,
    TensorBuffer,
    SymbolicTensor,
    list_map,
    dtypes,
    devices,
    DType,
    Device,
    CodegenOutput,
)

import math
import numpy as np
from typing import (
    List,
    Any,
)
import os

from slope.operators import operator_set
from slope.procedures import procedure_set
import tempfile
import importlib


def annotate_shape(symval):
    xdtype = symval.dtype.mlir
    if len(symval.shape) > 0:
        xshape = f"{'x'.join((repr(i) for i in symval.shape))}"
        return f"tensor<{xshape}x{xdtype}>"
    else:
        return f"tensor<{xdtype}>"


def annotate_sig(in_symvals, out_symvals):
    if isinstance(in_symvals, SymbolicTensor):
        in_symvals = (in_symvals,)
    if isinstance(out_symvals, SymbolicTensor):
        out_symvals = (out_symvals,)
    in_sig = f"({','.join(annotate_shape(t) for t in in_symvals)})"
    out_sig = f"({','.join(annotate_shape(t) for t in out_symvals)})"
    sig = f"{in_sig} -> {out_sig}"
    return sig


class NumpyBackend(Backend):
    dtype_for_indices = dtypes.int64
    dtype_map = {
        dtypes.float32: np.dtypes.Float32DType(),
        dtypes.uint8: np.dtypes.UInt8DType(),
        dtypes.int8: np.dtypes.Int8DType(),
        dtypes.bool: np.dtypes.BoolDType(),
        dtypes.int32: np.dtypes.Int32DType(),
        dtypes.int64: np.dtypes.Int64DType(),
        dtypes.uint64: np.dtypes.UInt64DType(),
        dtypes.float16: np.dtypes.Float16DType(),
    }
    device_map = {
        devices.cpu: "cpu",
    }

    dtype_map_inv = {v: k for k, v in dtype_map.items()}
    device_map_inv = {v: k for k, v in device_map.items()}

    def from_numpy(self, val, dtype=None, device=None):
        dtype = dtype or self.DEFAULT_DTYPE
        return Tensor(TensorBuffer(np.array(val, dtype=self.dtype_map[dtype])))

    def numpy_of(self, tensor: Tensor, memmap=False):
        if not memmap:
            return tensor.buf.val
        with tempfile.NamedTemporaryFile() as arr_file:
            arr = np.memmap(arr_file.name, dtype=tensor.dtype.numpy, shape=tensor.shape, mode="w+")
            arr[:] = tensor.buf.val
            return arr

    def shape_of(self, tensor):
        return tuple(int(i) for i in tensor.buf.val.shape)

    def dtype_of(self, tensor: Tensor):
        return self.dtype_map_inv[tensor.buf.val.dtype]

    def device_of(self, tensor: Tensor):
        return devices.cpu

    def codegen(self, program: Program, args, *, fn_name: str = "main", fn_defs=dict()) -> List[Any]:
        if fn_name == "main":
            assert not hasattr(self, "fn_count")
            self.fn_count = 0

        def indent(code):
            spaces = " " * (len(code) - len(code.lstrip()))
            spaces += " " * 4
            return "\n".join([spaces + line for line in code.strip().split("\n")])

        # codegen is recursive if jit-of-jit happens
        body_code_lines = []

        for instruction in program.instructions:
            if len(instruction.out_binders) == 0:  # skip codegen for function returns nothing
                continue
            in_vals = list_map(lambda x: program.env[x], instruction.inputs)
            out_vals = list_map(lambda z: program.env[z], instruction.out_binders)
            if isinstance(instruction.op, MetaOperator):
                impl_code, fn_defs = self.impls[instruction.op](args, instruction, fn_defs, in_vals, out_vals)
            else:
                if instruction.op in self.impls.keys():
                    impl_code = self.impls[instruction.op](*in_vals, *out_vals, **instruction.params)
                else:
                    # No impl is defined, fallback to procedure
                    impl_code, fn_defs = self.codegen_impl_as_procedure(args, instruction, fn_defs, in_vals, out_vals)

            for impl_code_line in impl_code.split("\n"):  # handle multi-line code
                body_code_lines += [indent(impl_code_line)]

        in_binders = list_map(lambda x: program.env[x], program.in_binders)
        arg_type_strs = [f"{inb.name}" for inb in in_binders]
        fn_args_str = ", ".join(arg_type_strs)

        functions_code_lines = []
        if fn_name == "main":
            for fn_def_code_lines in fn_defs.values():
                functions_code_lines += [indent(line) for line in fn_def_code_lines]

        outs = list_map(lambda x: program.env[x], program.outs)
        out_type_strs = [f"{out.name}" for out in outs]
        out_type_str = ", ".join(out_type_strs)
        head_code_line = [f"def {fn_name}({fn_args_str}): # ({out_type_str})"]
        return_line = [indent(f"return {out_type_str}")]
        code_lines = head_code_line + functions_code_lines + body_code_lines + return_line
        slope.dblog(
            f"\n---- {program.name} codegen:\n\n" + "\n".join(code_lines) + "\n\n===============\n",
            enable=slope.LOG_JIT,
        )

        if fn_name == "main":
            del self.fn_count
        assert len(outs) == len(program.outs)
        return CodegenOutput(
            code_lines=code_lines,
            fn_defs=fn_defs,
            in_binders=in_binders,
            outs=outs,
        )

    def codegen_impl_as_procedure(self, args, instruction: Instruction, fn_defs, in_vals, out_vals):
        op: Operator = instruction.op
        op_name = {v: k for k, v in vars(self.operator_set).items()}[op]
        op_procedure = getattr(self.procedure_set, op_name)
        symvals_in = tuple(inp.symval for inp in instruction.inputs)
        params = instruction.params
        op_program, consts, _ = slope.core.make_program(
            op_procedure,
            *symvals_in,
            static_args=tuple(params.items()),
            name=op.name,
        )
        name = slope.core.jit.get_jit_name(tuple(symvals_in), params)
        if name not in fn_defs.keys():
            op_codegen_output: CodegenOutput = self.codegen(
                op_program,
                args,
                fn_name=name,
                fn_defs=fn_defs,
            )
            fn_defs = {**fn_defs, **op_codegen_output.fn_defs}
            fn_defs[name] = op_codegen_output.code_lines
        in_names = ", ".join(i.name for i in in_vals)
        out_names = ", ".join(o.name for o in out_vals)
        impl_code = f"{out_names} = {name}({in_names})"
        return impl_code, fn_defs

    def compile(self, codegen_output):
        deps_dict = dict()
        deps_dict["numpy"] = importlib.import_module("numpy")
        deps_dict["np"] = deps_dict["numpy"]
        deps_dict["math"] = importlib.import_module("math")
        code_lines = codegen_output.code_lines
        exec_locals = dict()
        code = "\n".join(code_lines)
        exec(compile(code, "<string>", "exec"), deps_dict, exec_locals)
        fn = exec_locals["main"]
        return fn, code

    def export(self, jit_output: slope.core.JitOutput, output_path, export_params, input_names, output_names, **kwargs):
        os.makedirs(output_path, exist_ok=True)
        consts_dir_path = os.path.join(output_path, "consts")
        os.makedirs(consts_dir_path, exist_ok=True)
        in_binders = jit_output.codegen_output.in_binders
        outs = jit_output.codegen_output.outs
        num_consts = jit_output.program.num_consts
        load_consts_code = ""
        for i in range(num_consts):
            const_name = in_binders[i].name
            const_path = os.path.join(consts_dir_path, f"{const_name}.npy")
            load_consts_code += f"""{const_name} = np.load(os.path.join(consts_dir_path, "{const_name}.npy"))\n"""
            np.save(const_path, in_binders[i].symval.dtype.numpy)
        input_args_code = ", ".join(inb.name for inb in in_binders[num_consts:])
        args_code = ", ".join(inb.name for inb in in_binders)
        input_arg_names = [inb.name for inb in in_binders[num_consts:]]
        input_arg_names_str = ", ".join(input_arg_names)
        outs_names = [out.name for out in outs]

        test_input_code = ""
        for i in range(num_consts, len(in_binders)):
            input_name = in_binders[i].name
            input_shape = in_binders[i]["type"].shape
            dtype = in_binders[i]["type"].dtype
            input_dtype = f"{'np.' if dtype is not dtypes.bool else ''}{self.dtype_map(dtype)}"
            test_input_code += f"""    {input_name} = np.ones({input_shape}, dtype={input_dtype})\n"""

        module_path = os.path.join(output_path, "__init__.py")
        module_code = f"""import numpy as np
    import os
    root_path = os.path.dirname(__file__)
    consts_dir_path =  os.path.join(root_path, "consts")
    {load_consts_code}
    {jit_output.code}

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


backend = NumpyBackend(operator_set, procedure_set)


@backend.set_impl(backend.operator_set.jit_op)
def jit_op_impl(self, args, instruction, fn_defs, in_vals, out_vals):
    jit_program = instruction.params["program"]
    jit_name = f"{jit_program.name}"
    if jit_name not in fn_defs.keys():
        jit_codegen_output = self.codegen(
            jit_program,
            args,
            fn_name=jit_name,
            fn_defs=fn_defs,
        )
        fn_defs[jit_name] = jit_codegen_output.code_lines
        fn_defs = {**fn_defs, **jit_codegen_output.fn_defs}
    args_str = ", ".join(i.name for i in in_vals)
    impl_code = f"{', '.join(o.name for o in out_vals)} = slope.{jit_name}({args_str})"
    return impl_code, fn_defs


### Operator Impls


@backend.set_impl(operator_set.cast)
def cast_impl(self, x, y, *, dtype):
    return f"{y.name} = {x.name}.astype({'np.' if dtype is not dtypes.bool else ''}{self.dtype_map[dtype]})"


@backend.set_impl(backend.operator_set.stop_gradient)
def stop_gradient_impl(self, x, y):
    return f"{y.name} = {x.name}"


@backend.set_impl(backend.operator_set.sqrt)
def sqrt_impl(self, x, y):
    return f"{y.name} = np.sqrt({x.name})"


@backend.set_impl(backend.operator_set.exp)
def exp_impl(self, x, y):
    return f"{y.name} = np.exp({x.name})"


@backend.set_impl(backend.operator_set.log)
def log_impl(self, x, y):
    return f"{y.name} = np.log({x.name})"


@backend.set_impl(backend.operator_set.sin)
def sin_impl(self, x, y):
    return f"{y.name} = np.sin({x.name})"


@backend.set_impl(backend.operator_set.invert)
def invert_impl(self, x, y):
    return f"{y.name} = ~{x.name}"


@backend.set_impl(backend.operator_set.add)
def add_impl(self, x, w, y):
    return f"{y.name} = {x.name} + {w.name}"


@backend.set_impl(backend.operator_set.sub)
def sub_impl(self, x, w, y):
    return f"{y.name} = {x.name} - {w.name}"


@backend.set_impl(backend.operator_set.mul)
def mul_impl(self, x, w, y):
    return f"{y.name} = {x.name} * {w.name}"


@backend.set_impl(backend.operator_set.div)
def div_impl(self, x, w, y):
    return f"{y.name} = {x.name} / {w.name}"


@backend.set_impl(backend.operator_set.pow)
def pow_impl(self, x, w, y):
    return f"{y.name} = {x.name} ** {w.name}"


@backend.set_impl(backend.operator_set.equal)
def equal_impl(self, x, w, y):
    return f"{y.name} = {x.name} == {w.name}"


@backend.set_impl(backend.operator_set.less)
def less_impl(self, x, w, y):
    return f"{y.name} = {x.name} < {w.name}"


@backend.set_impl(backend.operator_set.greater)
def greater_impl(self, x, w, y):
    return f"{y.name} = {x.name} > {w.name}"


@backend.set_impl(backend.operator_set.maximum)
def maximum_impl(self, x, w, y):
    return f"{y.name} = np.maximum({x.name}, {w.name})"


@backend.set_impl(backend.operator_set.matmul)
def matmul_impl(self, x, w, y):
    return f"{y.name} = {x.name} @ {w.name}"


# @backend.set_impl(backend.operator_set.where)
# def where_impl(self, x, w, u, y):
#     return f"""{y.name} = np.where({x.name}, %{w.name}, %{u.name})"""


@backend.set_impl(operator_set.sum)
def sum_impl(self, x, y, *, dim, keepdim):
    return f"{y.name} = np.sum({x.name}, axis={dim}, keepdims={keepdim})"


@backend.set_impl(operator_set.max)
def max_impl(self, x, y, *, dim, keepdim):
    return f"{y.name} = np.max({x.name}, axis={dim}, keepdims={keepdim})"


@backend.set_impl(operator_set.arange)
def arange_impl(self, y, *, start, stop, stride, dtype, device):
    return f"{y.name} = np.arange({start}, {stop}, {stride}, dtype={'np.' if dtype is not dtypes.bool else ''}{self.dtype_map[dtype]})"


@backend.set_impl(operator_set.full)
def full_impl(self, y, *, shape, fill_value, dtype, device):
    return f"{y.name} = np.full({shape}, {fill_value}, dtype={'np.' if dtype is not dtypes.bool else ''}{self.dtype_map[dtype]})"


@backend.set_impl(operator_set.random_uniform)
def random_uniform_impl(self, y, *, shape, dtype, device):
    if len(shape) > 0:
        return f"{y.name} = {'np.array(' if shape == () else ''}np.random.uniform(low=np.zeros(shape={shape})){')' if shape == () else ''}.astype(dtype={'np.' if dtype is not dtypes.bool else ''}{self.dtype_map[dtype]})"


@backend.set_impl(operator_set.random_normal)
def random_normal_impl(self, y, *, shape, dtype, device):
    return f"{y.name} = {'np.array(' if shape == () else ''}np.random.normal(loc=np.zeros(shape={shape})){')' if shape == () else ''}.astype(dtype={'np.' if dtype is not dtypes.bool else ''}{self.dtype_map[dtype]})"


@backend.set_impl(operator_set.pad)
def pad_impl(self, x, *, padding, mode, value):
    pad_width = [(lo, hi) for lo, hi in zip(padding[0::2], padding[1::2])]
    return f"ret = np.pad({x}, {pad_width}, constant_values={value})"


@backend.set_impl(operator_set.expand)
def expand_impl(self, x, y, *, shape):
    return f"{y.name} = np.broadcast_to({x.name}, shape={shape})"


@backend.set_impl(operator_set.reshape)
def reshape_impl(self, x, y, *, shape):
    return f"{y.name} = np.reshape({x.name}, newshape={shape})"


@backend.set_impl(operator_set.pad)
def pad_impl(self, x, y, *, padding, mode, value):
    padding = padding[::-1]
    pad_width = [(lo, hi) for lo, hi in zip(padding[0::2], padding[1::2])]
    return f"{y.name} = np.pad({x.name}, {pad_width}, constant_values={value})"


@backend.set_impl(operator_set.slice)
def slice_impl(self, x, y, *, starts, limits, strides):
    return f"{y.name} = {x.name}[tuple(slice(s, l, st) for s, l, st in zip({starts}, {limits}, {strides}))]"
    # slice_strs = [f"{s}:{e}:{st}" for s, e ,st in zip(starts, limits, strides)]
    # slices = ", ".join([f"np.s_[{sl}]" for sl in slice_strs])
    # return f"{y.name} = {x.name}[{slices}]"


@backend.set_impl(operator_set.cat)
def cat_impl(self, *xs, dim):
    xs, y = xs[:-1], xs[-1]
    return f"{y.name} = np.concatenate(({', '.join([x.name for x in xs])}), axis={dim})"


@backend.set_impl(operator_set.permute)
def permute_impl(self, x, y, *, perm):
    return f"{y.name} = np.transpose({x.name}, axes={perm})"


@backend.set_impl(operator_set.flip)
def flip_impl(self, x, y, *, dim):
    return f"{y.name} = np.flip({x.name}, axis={dim})"
