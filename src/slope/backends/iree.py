import slope
import slope.core
from slope.core import (
    Backend,
    Operator,
    Instruction,
    Program,
    MetaOperator,
    OperatorSet,
    ProcedureSet,
    Tensor,
    TensorBuffer,
    SymbolicTensor,
    list_map,
    dtypes,
    devices,
    DType,
    Device,
    CodegenOut,
)

import math
import numpy as np
from typing import (
    List,
    Any,
)
import iree.compiler
import iree.runtime
import os

from slope.operators import operator_set
from slope.procedures import procedure_set
import tempfile


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


class IREEBackend(Backend):
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
        devices.cpu: "local-task",
        devices.cuda0: "cuda",
    }

    target_map = {
        devices.cpu: "llvm-cpu",
        devices.cuda0: "cuda",
    }

    dtype_map_inv = {v: k for k, v in dtype_map.items()}
    device_map_inv = {v: k for k, v in device_map.items()}

    def from_numpy(self, val, dtype=None, device=None):
        dtype = dtype or self.DEFAULT_DTYPE
        device = device or self.DEFAULT_DEVICE
        np_val = np.array(val, dtype=dtype.numpy)
        iree_device = iree.runtime.get_device(self.device_map[device])
        val = iree.runtime.asdevicearray(iree_device, np_val)
        return Tensor(TensorBuffer(val))

    def numpy_of(self, tensor: Tensor, memmap=False):
        if not memmap:
            return tensor.buf.val.to_host()
        with tempfile.NamedTemporaryFile() as arr_file:
            arr = np.memmap(arr_file.name, dtype=tensor.dtype.numpy, shape=tensor.shape, mode="w+")
            arr[:] = tensor.buf.val.to_host()
            return arr

    def device_of(self, tensor: Tensor):
        return self.device_map_inv[str(tensor.buf.val._device)]

    def shape_of(self, tensor: Tensor):
        return tuple(tensor.buf.val.shape)

    def dtype_of(self, tensor: Tensor):
        return self.dtype_map_inv[tensor.buf.val.dtype]

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
        fn_args_str = ", ".join([f"%{i.name}: {annotate_shape(i.symval)}" for i in in_binders])

        outs = list_map(lambda x: program.env[x], program.outs)
        out_str = ", ".join([f"%{o.name}" for o in outs])
        out_type_str = ", ".join([f"{annotate_shape(o.symval)}" for o in outs])

        head_code_line = [f"func.func @{fn_name} ({fn_args_str}) -> ({out_type_str})"]
        tail_code_line = [indent(f'"func.return"({out_str}): ({out_type_str}) -> ()')]
        model_code_lines = head_code_line + ["{"] + body_code_lines + tail_code_line + ["}"]

        functions_code_lines = []
        for fn_def_code_lines in fn_defs.values():
            functions_code_lines += fn_def_code_lines
        code_lines = model_code_lines + functions_code_lines
        slope.core.dblog(
            f"\n---- {program.name} codegen:\n\n" + "\n".join(code_lines) + "\n\n===============\n",
            enable=slope.LOG_JIT,
        )

        if fn_name == "main":
            del self.fn_count
        assert len(outs) == len(program.outs)
        return CodegenOut(
            code_lines=code_lines,
            fn_defs=fn_defs,
            in_binders=in_binders,
            outs=outs,
        )

    def codegen_impl_as_procedure(self, args, instruction: Instruction, fn_defs, in_vals, out_vals):
        op: Operator = instruction.op
        op_name = {v: k for (k, v) in vars(self.operator_set).items()}[op]
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
            op_codegen_out: CodegenOut = self.codegen(
                op_program,
                args,
                fn_name=name,
                fn_defs=fn_defs,
            )
            fn_defs = {**fn_defs, **op_codegen_out.fn_defs}
            fn_defs[name] = op_codegen_out.code_lines
        in_names = ", ".join(f"%{i.name}" for i in in_vals)
        out_names = ", ".join(f"%{o.name}" for o in out_vals)
        sig = annotate_sig(tuple(i.symval for i in in_vals), out_vals[0].symval)
        impl_code = f"{out_names} = func.call @{name}({in_names}) : {sig}"
        return impl_code, fn_defs

    def compile(self, codegen_out):
        code_lines = codegen_out.code_lines
        code = "\n".join(code_lines)
        instance = iree.runtime.VmInstance()
        device = codegen_out.outs[0].device
        iree_device = iree.runtime.get_device(self.device_map[device])
        hal_module = iree.runtime.create_hal_module(instance, iree_device)
        binary = iree.compiler.compile_str(
            code,
            target_backends=(self.target_map[device],),
        )
        m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
        context = iree.runtime.VmContext(instance, modules=[hal_module, m])
        f = m.lookup_function("main")
        finv = iree.runtime.FunctionInvoker(context, iree_device, f, tracer=None)
        return finv, code

    def export(self, jit_object, output_path, *args, **kwargs):
        # iree.compiler.core.DEFAULT_TESTING_BACKENDS
        target_backends = kwargs.get("target_backends", "llvm-cpu")
        if isinstance(target_backends, str):
            target_backends = (target_backends,)
        os.makedirs(output_path, exist_ok=True)

        code_lines = jit_object.codegen_out.code_lines[:]
        in_binders = jit_object.codegen_out.in_binders
        outs = jit_object.codegen_out.outs
        num_consts = jit_object.program.num_consts
        with tempfile.NamedTemporaryFile(mode="w+") as f:
            for i in range(num_consts):
                const_pattern = f"%{in_binders[i].name}: {annotate_shape(in_binders[i].symval)}"
                code_lines[0] = code_lines[0].replace(f"{const_pattern}, ", "")
                code_lines[0] = code_lines[0].replace(f"{const_pattern})", ")")
            f.write("\n".join(code_lines[:2]) + "\n")
            for i in range(num_consts):
                f.write(
                    f"    %{in_binders[i].name} = stablehlo.constant "
                    f"""dense<"0x{in_binders[i].symval.numpy(memmap=True).tobytes().hex()}">"""
                    f": {annotate_shape(in_binders[i].symval)}\n"
                )
            f.writelines("\n".join(code_lines[2:]))
            f.flush()
            iree.compiler.compile_file(
                f.name, target_backends=target_backends, output_file=os.path.join(output_path, "model.vmfb")
            )

        input_arg_names = [ib.name for ib in in_binders[num_consts:]]
        input_arg_names_str = ", ".join(input_arg_names)
        outs_names = [out.name for out in outs]

        test_input_code = ""
        for i in range(num_consts, len(in_binders)):
            input_name = in_binders[i].name
            input_shape = in_binders[i].symval.shape
            dtype = in_binders[i].symval.dtype
            input_dtype = ("np." + dtype.numpy.__name__) if dtype is not dtypes.bool else "bool"
            test_input_code += f"""    {input_name} = np.ones({input_shape}, dtype={input_dtype})\n"""

        module_path = os.path.join(output_path, "__init__.py")
        module_code = f"""#!/usr/bin/env python
import os
import numpy as np
import iree.runtime

root_path = os.path.dirname(__file__)
model_path = os.path.join(root_path, "model.vmfb")
input_arg_names = {input_arg_names}
out_names = {outs_names}

instance = iree.runtime.VmInstance()
iree_device = iree.runtime.get_device("local-task")
hal_module = iree.runtime.create_hal_module(instance, iree_device)
with open(model_path, 'r+b') as f:
    m = iree.runtime.VmModule.from_flatbuffer(instance, f.read())
context = iree.runtime.VmContext(instance, modules=[hal_module, m])
f = m.lookup_function("main")
finv = iree.runtime.FunctionInvoker(context, iree_device, f, tracer=None)
if __name__ == "__main__":
{test_input_code}
    print("inputs:")
    for inp_name, inp in zip(input_arg_names, ({input_arg_names_str})):
        print(f"{{inp_name}} {{inp.shape}}{{inp.dtype}}\\n{{inp}}\\n")
    outs = finv({input_arg_names_str})
    if not isinstance(outs, tuple):
        outs = (outs,)
    print("outputs:")
    for out_name, out in zip(out_names, outs):
        print(f"{{out_name}} {{out.shape}}{{out.dtype}}\\n{{out.to_host()}}\\n")
"""
        with open(module_path, "w") as f:
            f.write(module_code)
            slope.dblog(module_code, enable=slope.LOG_JIT)


backend = IREEBackend(operator_set, procedure_set)


@backend.set_impl(backend.operator_set.jit_op)
def jit_op_impl(self, args, instruction, fn_defs, in_vals, out_vals):
    jit_program = instruction.params["program"]
    jit_name = f"{jit_program.name}"
    if jit_name not in fn_defs.keys():
        jit_codegen_out = self.codegen(
            jit_program,
            args,
            fn_name=jit_name,
            fn_defs=fn_defs,
        )
        fn_defs[jit_name] = jit_codegen_out.code_lines
        fn_defs = {**fn_defs, **jit_codegen_out.fn_defs}
    args_str = ", ".join(i.name for i in in_vals)
    sig = annotate_sig(tuple(i.symval for i in in_vals), out_vals[0].symval)
    impl_code = f"{', '.join(o.name for o in out_vals)} = func.call @{jit_name}({args_str}) {sig}"
    return impl_code, fn_defs


@backend.set_impl(backend.operator_set.cast)
def cast_impl(self, x, y, *, dtype):
    return f'%{y.name} = "stablehlo.convert"(%{x.name}) : {annotate_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.stop_gradient)
def stop_gradient_impl(self, x, y):
    return f'%{y.name} = "stablehlo.optimization_barrier"(%{x.name}): {annotate_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.sqrt)
def sqrt_impl(self, x, y):
    return f'%{y.name} = "stablehlo.sqrt"(%{x.name}) : {annotate_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.exp)
def exp_impl(self, x, y):
    return f'%{y.name} = "stablehlo.exponential"(%{x.name}) : {annotate_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.log)
def log_impl(self, x, y):
    return f'%{y.name} = "stablehlo.log"(%{x.name}) : {annotate_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.sin)
def sin_impl(self, x, y):
    return f'%{y.name} = "stablehlo.sine"(%{x.name}) : {annotate_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.invert)
def invert_impl(self, x, y):
    return f'%{y.name} = "stablehlo.not"(%{x.name}) : {annotate_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.add)
def add_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.add"(%{x.name}, %{w.name}) : {annotate_sig((x.symval, w.symval), y.symval)}'


@backend.set_impl(backend.operator_set.sub)
def sub_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.subtract"(%{x.name}, %{w.name}) : {annotate_sig((x.symval, w.symval), y.symval)}'


@backend.set_impl(backend.operator_set.mul)
def mul_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.multiply"(%{x.name}, %{w.name}) : {annotate_sig((x.symval, w.symval), y.symval)}'


@backend.set_impl(backend.operator_set.div)
def div_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.divide"(%{x.name}, %{w.name}) : {annotate_sig((x.symval, w.symval), y.symval)}'


@backend.set_impl(backend.operator_set.pow)
def pow_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.power"(%{x.name}, %{w.name}) : {annotate_sig((x.symval, w.symval), y.symval)}'


def get_compare_type(dtype):
    return (
        "FLOAT"
        if dtypes.is_float(dtype)
        else "SIGNED"
        if dtypes.is_int(dtype) and dtype is not dtypes.uint8
        else "UNSIGNED"
    )


@backend.set_impl(backend.operator_set.equal)
def equal_impl(self, x, w, y):
    return f"""%{y.name} = "stablehlo.compare"(%{x.name}, %{w.name}) {{
  comparison_direction = #stablehlo<comparison_direction EQ>,
  compare_type = #stablehlo<comparison_type {get_compare_type(x.symval.dtype)}>
}}  : {annotate_sig((x.symval, w.symval), y.symval)}
"""


@backend.set_impl(backend.operator_set.less)
def less_impl(self, x, w, y):
    return f"""%{y.name} = "stablehlo.compare"(%{x.name}, %{w.name}) {{
  comparison_direction = #stablehlo<comparison_direction LT>,
  compare_type = #stablehlo<comparison_type {get_compare_type(x.symval.dtype)}>
}}  : {annotate_sig((x.symval, w.symval), y.symval)}
"""


@backend.set_impl(backend.operator_set.greater)
def greater_impl(self, x, w, y):
    return f"""%{y.name} = "stablehlo.compare"(%{x.name}, %{w.name}) {{
  comparison_direction = #stablehlo<comparison_direction GT>,
  compare_type = #stablehlo<comparison_type {get_compare_type(x.symval.dtype)}>
}}  : {annotate_sig((x.symval, w.symval), y.symval)}
"""


@backend.set_impl(backend.operator_set.maximum)
def maximum_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.maximum"(%{x.name}, %{w.name}) : {annotate_sig((x.symval, w.symval), y.symval)}'


@backend.set_impl(backend.operator_set.matmul)
def matmul_impl(self, x, w, y):
    x_bdims = (
        []
        if (w.symval.ndim <= 2 or x.symval.ndim == 1)
        else list(range(x.symval.ndim - (2 if x.symval.ndim > 2 else 1)))
    )
    w_bdims = [] if (w.symval.ndim == 1 or x.symval.ndim <= 2) else list(range(w.symval.ndim - 2))
    x_cdim = 0 if x.symval.ndim == 1 else x.symval.ndim - 1
    w_cdim = 0 if w.symval.ndim == 1 else w.symval.ndim - 2
    return f"""%{y.name} = "stablehlo.dot_general"(%{x.name}, %{w.name}) {{
  dot_dimension_numbers = #stablehlo.dot<
    lhs_batching_dimensions = {x_bdims},
    rhs_batching_dimensions = {w_bdims},
    lhs_contracting_dimensions = [{x_cdim}],
    rhs_contracting_dimensions = [{w_cdim}]
  >,
  precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
    }}  : {annotate_sig((x.symval, w.symval), y.symval)}
"""


@backend.set_impl(backend.operator_set.sum)
def sum_impl(self, x, y, *, dim, keepdim):
    zero = "0." if dtypes.is_float(y.symval.dtype) else "0"
    y_init_type = SymbolicTensor((), y.symval.dtype, y.symval.device)
    y_mlir_type = annotate_shape(y_init_type)
    y_out_type = (
        y.symval
        if not keepdim
        else SymbolicTensor(
            tuple(d for i, d in enumerate(y.symval.shape) if i not in dim),
            y.symval.dtype,
            y.symval.device,
        )
    )
    return f"""
%{y.name}_init = stablehlo.constant dense<{zero}> : {annotate_shape(y_init_type)}
%{y.name}{'_' if keepdim else ''} = "stablehlo.reduce"(%{x.name}, %{y.name}_init) ({{
  ^bb0(%arg0: {y_mlir_type}, %arg1: {y_mlir_type}):
    %0 = "stablehlo.add"(%arg0, %arg1) : {annotate_sig((y_init_type, y_init_type), y_init_type)}
    "stablehlo.return"(%0) : ({y_mlir_type}) -> ()
}}) {{
  dimensions = dense<{repr(list(dim))}> : tensor<{len(dim)}xi64>
}} : {annotate_sig((x.symval, y_init_type), y_out_type)}
{f'%{y.name} = "stablehlo.reshape"(%{y.name}_) : {annotate_sig((y_out_type,), y.symval)}' if keepdim else ''}"""


@backend.set_impl(backend.operator_set.max)
def max_impl(self, x, y, *, dim, keepdim):
    min_val = {
        dtypes.float32: "1.E-38",
        dtypes.int8: "-128",
        dtypes.int32: "-65536",
    }[x.symval.dtype]
    y_init_type = SymbolicTensor((), y.symval.dtype, y.symval.device)
    y_mlir_type = annotate_shape(y_init_type)
    y_out_type = (
        y.symval
        if not keepdim
        else SymbolicTensor(
            tuple(d for i, d in enumerate(y.symval.shape) if i not in dim),
            y.symval.dtype,
            y.symval.device,
        )
    )
    return f"""
%{y.name}_init = stablehlo.constant dense<{min_val}> : {annotate_shape(y_init_type)}
%{y.name}{'_' if keepdim else ''} = "stablehlo.reduce"(%{x.name}, %{y.name}_init) ({{
  ^bb0(%arg0: {y_mlir_type}, %arg1: {y_mlir_type}):
    %0 = "stablehlo.maximum"(%arg0, %arg1) : {annotate_sig((y_init_type, y_init_type), y_init_type)}
    "stablehlo.return"(%0) : ({y_mlir_type}) -> ()
}}) {{
  dimensions = dense<{repr(list(dim))}> : tensor<{len(dim)}xi64>
}} : {annotate_sig((x.symval, y_init_type), y_out_type)}
{f'%{y.name} = "stablehlo.reshape"(%{y.name}_) : {annotate_sig((y_out_type,), y.symval)}' if keepdim else ''}
"""


@backend.set_impl(backend.operator_set.arange)
def arange_impl(self, y, *, start, stop, stride, dtype, device):
    if stride == 1 and start == 0:
        return f"""%{y.name} = "stablehlo.iota"() {{iota_dimension = 0 : i64}} : {annotate_sig((), y.symval)}"""
    one_symval = y.symval.like(shape=(1,))
    return f"""
%{y.name}_scale_ = stablehlo.constant dense<{stride}> : {annotate_shape(one_symval)}
%{y.name}_scale = "stablehlo.broadcast_in_dim"(%{y.name}_scale_) {{
        broadcast_dimensions = dense<{repr(list(range(y.symval.ndim)))}>: tensor<{y.symval.ndim}xi64>
        }} : {annotate_sig(( one_symval,), y.symval)}
%{y.name}_shift_ = stablehlo.constant dense<{start}> : {annotate_shape(y.symval.like(shape=(1,)))}
%{y.name}_shift = "stablehlo.broadcast_in_dim"(%{y.name}_shift_) {{
        broadcast_dimensions = dense<{repr(list(range(y.symval.ndim)))}>: tensor<{y.symval.ndim}xi64>
        }} : {annotate_sig(( one_symval,), y.symval)}
%{y.name}__ = "stablehlo.iota"() {{iota_dimension = 0 : i64}} : {annotate_sig((), y.symval)}
%{y.name}_ = "stablehlo.multiply"(%{y.name}__, %{y.name}_scale) : {annotate_sig((y.symval, y.symval), y.symval)}
%{y.name} = "stablehlo.add"(%{y.name}_, %{y.name}_shift) : {annotate_sig((y.symval, y.symval), y.symval)}
"""


@backend.set_impl(backend.operator_set.full)
def full_impl(self, y, *, shape, fill_value, dtype, device):
    fill_value = float(fill_value) if "f" in dtype.mlir else int(fill_value)
    fill_value = repr(fill_value)
    fill_value = fill_value.replace("e", "E") if "." in fill_value else fill_value.replace("e", ".E")
    return f'%{y.name} = "stablehlo.constant"() {{ value = dense<{fill_value}> : {annotate_shape(y.symval)} }} : {annotate_sig((), y.symval)}'


@backend.set_impl(backend.operator_set.random_uniform)
def random_uniform_impl(self, y, *, shape, dtype, device):
    zero = "0." if dtypes.is_float(y.symval.dtype) else "0"
    one = "1." if dtypes.is_float(y.symval.dtype) else "1"
    a_type = b_type = y.symval.like(shape=())
    is_scalar = shape == ()
    shape_val = f'dense<{repr(list(shape)) if not is_scalar else "[1]"}'
    shape_type = y.symval.like(shape=(1,) if is_scalar else (len(shape),), dtype=dtypes.int64)
    y_out_type = y.symval if not is_scalar else y.symval.like(shape=(1,))
    return f"""%{y.name}_a = stablehlo.constant dense<{zero}> : {annotate_shape(a_type)}
%{y.name}_b = stablehlo.constant dense<{one}> : {annotate_shape(b_type)}
%{y.name}_shape = stablehlo.constant {shape_val}> : {annotate_shape(shape_type)}
%{y.name}{'_' if is_scalar else ''} = "stablehlo.rng"(%{y.name}_a, %{y.name}_b,%{y.name}_shape) {{
        rng_distribution = #stablehlo<rng_distribution UNIFORM>
        }} : {annotate_sig((a_type, b_type, shape_type), y_out_type)}
{f'%{y.name} = "stablehlo.reshape"(%{y.name}_) : {annotate_sig((y_out_type,), y.symval)}' if is_scalar else ''}"""


@backend.set_impl(backend.operator_set.random_normal)
def random_normal_impl(self, y, *, shape, dtype, device):
    zero = "0." if dtypes.is_float(y.symval.dtype) else "0"
    one = "1." if dtypes.is_float(y.symval.dtype) else "1"
    a_type = b_type = SymbolicTensor((), dtype, device)
    is_scalar = shape == ()
    shape_val = f'dense<{repr(list(shape)) if not is_scalar else "[1]"}'
    shape_type = SymbolicTensor((1,) if is_scalar else (len(shape),), slope.dtypes.int64, device)
    y_out_type = y.symval if not is_scalar else SymbolicTensor((1,), y.symval.dtype, y.symval.device)
    return f"""%{y.name}_a = stablehlo.constant dense<{zero}> : {annotate_shape(a_type)}
%{y.name}_b = stablehlo.constant dense<{one}> : {annotate_shape(b_type)}
%{y.name}_shape = stablehlo.constant {shape_val}> : {annotate_shape(shape_type)}
%{y.name}{'_' if is_scalar else ''} = "stablehlo.rng"(%{y.name}_a, %{y.name}_b,%{y.name}_shape) {{
        rng_distribution = #stablehlo<rng_distribution NORMAL>}} : {annotate_sig((a_type, b_type, shape_type), y_out_type)}
{f'%{y.name} = "stablehlo.reshape"(%{y.name}_) : {annotate_sig((y_out_type,), y.symval)}' if is_scalar else ''}"""


# @backend.set_impl(backend.operator_set.rng_bits)
# def rng_bits_impl(self, x, y, *, shape, dtype, device):
#     return f"""%{y.name}_, %{y.name} =  "stablehlo.rng_bit_generator"(%{x.name}) {{
#   rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>
# }} : ({annotate_shape(x.symval)}) -> ({annotate_shape(x.symval)}, {annotate_shape(y.symval)})"""


@backend.set_impl(backend.operator_set.expand)
def expand_impl(self, x, y, *, shape):
    return f"""%{y.name} = "stablehlo.broadcast_in_dim"(%{x.name}) {{
        broadcast_dimensions = dense<{repr(list(range(len(shape))))}>: tensor<{len(shape)}xi64>
        }} : {annotate_sig(( x.symval,), y.symval)}
"""


@backend.set_impl(backend.operator_set.reshape)
def reshape_impl(self, x, y, *, shape):
    return f'%{y.name} = "stablehlo.reshape"(%{x.name}) : {annotate_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.pad)
def pad_impl(self, x, y, *, padding, mode, value):
    value = float(value) if "f" in x.symval.dtype.mlir else int(value)
    value_type = SymbolicTensor((), x.symval.dtype, x.symval.device)
    lo = padding[0::2][::-1]
    hi = padding[1::2][::-1]
    return f"""%{y.name}_value = stablehlo.constant dense<{value}> : {annotate_shape(value_type)}
%{y.name} = "stablehlo.pad"(%{x.name}, %{y.name}_value) {{
  edge_padding_low = dense<{repr(list(lo))}> : tensor<{len(lo)}xi64>,
  edge_padding_high = dense<{repr(list(hi))}> : tensor<{len(hi)}xi64>,
  interior_padding = dense<{repr([0]*len(lo))}> : tensor<{len(lo)}xi64>
}} : {annotate_sig((x.symval, value_type), y.symval)}
"""


@backend.set_impl(backend.operator_set.slice)
def slice_impl(self, x, y, *, starts, limits, strides):
    return f"""%{y.name} = "stablehlo.slice"(%{x.name}) {{
  start_indices = dense<{repr(list(starts))}> : tensor<{len(starts)}xi64>,
  limit_indices = dense<{repr(list(limits))}> : tensor<{len(limits)}xi64>,
  strides = dense<{repr(list(strides))}> : tensor<{len(strides)}xi64>
}} : {annotate_sig((x.symval,), y.symval)}
"""


@backend.set_impl(backend.operator_set.cat)
def cat_impl(self, *xs, dim):
    xs, y = xs[:-1], xs[-1]
    return f"""%{y.name} = "stablehlo.concatenate"({', '.join([f'%{x.name}' for x in xs])}) {{
 dimension = {dim} : i64
}} : {annotate_sig(([x.symval for x in xs]), y.symval)}"""


@backend.set_impl(backend.operator_set.permute)
def permute_impl(self, x, y, *, perm):
    return f"""%{y.name} = "stablehlo.transpose"(%{x.name}) {{
  permutation = dense<{repr(list(perm))}> : tensor<{len(perm)}xi64>
}} : {annotate_sig((x.symval,), y.symval)}"""


@backend.set_impl(backend.operator_set.flip)
def flip_impl(self, x, y, *, dim):
    return f"""%{y.name} = "stablehlo.reverse"(%{x.name}) {{
  dimensions = dense<{repr(list(dim))}> : tensor<{len(dim)}xi64>
}}  : {annotate_sig((x.symval,), y.symval)}
"""


@backend.set_impl(backend.operator_set.conv)
def conv_impl(self, x, w, y, *, groups, stride, dilation, padding):
    padding = [[s, e] for s, e in zip(list(padding[0::2]), list(padding[1::2]))]
    D = len(x.symval.shape[2:])
    nD = repr(list(range(D)))[1:-1]
    xdims = f"[b, f, {nD}]"
    wdims = f"[o, i, {nD}]"
    return f"""%{y.name} = "stablehlo.convolution"(%{x.name}, %{w.name}) {{
  window_strides = dense<{list(stride)}> : tensor<{len(stride)}xi64>,
  padding = dense<{padding}> : tensor<{D}x{D}xi64>,
  lhs_dilation = dense<1> : tensor<{D}xi64>,
  rhs_dilation = dense<{list(dilation)}> : tensor<{D}xi64>,
  window_reversal = dense<false> : tensor<{D}xi1>,
  dimension_numbers = #stablehlo.conv<{xdims}x{wdims}->{xdims}>,
  feature_group_count = {groups} : i64,
  batch_group_count = 1 : i64,
  precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
}}  : {annotate_sig((x.symval, w.symval), y.symval)}
"""


# @backend.set_impl(backend.operator_set.where)
# def where_impl(self, x, w, u, y):
    # return f"""%{y.name} = "stablehlo.select"(%{x.name}, %{w.name}, %{u.name}) : {annotate_sig((x.symval,w.symval,u.symval), y.symval)}"""


@backend.set_impl(backend.operator_set.gather_nd)
def gather_nd_impl(self, x, w, y, *, batch_dims):
    operand_shape = list(x.symval.shape)
    indices_shape = list(w.symval.shape)
    r = x.symval.ndim
    q = w.symval.ndim
    b = batch_dims
    offset_dims = list(range(1, q))
    index_vector_dim = q - 1
    y_reshape = None
    w_arange = None
    if b > 0:
        w_arange = w.symval

    if indices_shape[-1] == r:
        slice_sizes = [1] * r
        start_index_map = [i for i in range(q)]
        if b != 0:
            start_index_map = start_index_map[b:] + start_index_map[:b]
        collapsed_slice_dims = [
            i
            for i in range(1 - b, len(slice_sizes) - b)
            if slice_sizes[i] == 1 and operand_shape[i] == indices_shape[i]
        ]

        y_reshape = y.symval.like(shape=y.symval.shape + (1,))
    elif indices_shape[-1] < r:
        slice_sizes = [*[1] * (r - 1), *operand_shape[-1:]]
        start_index_map = [i for i, s in enumerate(slice_sizes) if s == 1 and i < q]

        collapsed_slice_dims = []
        for i in range(len(slice_sizes)):
            if slice_sizes[i] == 1 and len(offset_dims) + len(collapsed_slice_dims) != r:
                collapsed_slice_dims += [i]

        if len(collapsed_slice_dims) != len(start_index_map):
            y_reshape = y.symval.like(shape=y.symval.shape + (1,))

    else:
        raise ValueError
    w_fixed = (
        w.symval
        if w_arange is None
        else w.symval.like(shape=tuple(d + b if i == b else d for i, d in enumerate(w.symval.shape)),
                           dtype=dtypes.int32)
    )
    w_affix = "" if w_arange is None else "_"
    y_fixed = y.symval if y_reshape is None else y_reshape
    y_affix = "" if y_reshape is None else "_"
    return f"""{f'''%{w.name}_i = "stablehlo.iota"() {{ iota_dimension = 0 : i64}} : {annotate_sig((), w_arange)}
%{w.name}_ = "stablehlo.concatenate"(%{w.name}_i, %{w.name}) {{ dimension = {b} : i64}} : {annotate_sig((w_arange, w.symval), w_fixed)}'''
    if w_arange is not None else ''
}%{y.name}{y_affix} = "stablehlo.gather"(%{x.name}, %{w.name}{w_affix}) {{
dimension_numbers = #stablehlo.gather<
offset_dims = {offset_dims},
collapsed_slice_dims = {collapsed_slice_dims},
start_index_map = {start_index_map},
index_vector_dim = {index_vector_dim}>,
slice_sizes = dense<{slice_sizes}> : tensor<{len(slice_sizes)}xi64>,
indices_are_sorted = false
}} : {annotate_sig((x.symval, w_fixed), y_fixed)}
{f'%{y.name} = "stablehlo.reshape"(%{y.name}_) : {annotate_sig((y_fixed,), y.symval)}' 
if y_reshape is not None else ''}"""


@backend.set_impl(backend.operator_set.scatter_nd)
def scatter_nd_impl(self, x, w, u, y):
    y_init_type = SymbolicTensor((), y.symval.dtype, y.symval.device)
    y_mlir_type = annotate_shape(y_init_type)

    r = x.symval.ndim
    q = w.symval.ndim
    index_vector_dim = q - 1
    update_window_dims = list(range(1, u.symval.ndim))
    inserted_window_dims = [0]
    scatter_dims_to_operand_dims = [0]

    return f"""%{y.name} = "stablehlo.scatter"(%{x.name}, %{w.name}, %{u.name}) ({{
  ^bb0(%arg0: {y_mlir_type}, %arg1: {y_mlir_type}):
    %0 = "stablehlo.add"(%arg0, %arg1) : {annotate_sig((y_init_type, y_init_type), y_init_type)}
    "stablehlo.return"(%0) : ({y_mlir_type}) -> ()
}}) {{
  scatter_dimension_numbers = #stablehlo.scatter<
  update_window_dims = {update_window_dims},
  inserted_window_dims = {inserted_window_dims},
  scatter_dims_to_operand_dims = {scatter_dims_to_operand_dims},
  index_vector_dim = {index_vector_dim}>,
  indices_are_sorted = false,
  unique_indices = false
}} : {annotate_sig((x.symval, w.symval, u.symval), y.symval)}
"""
