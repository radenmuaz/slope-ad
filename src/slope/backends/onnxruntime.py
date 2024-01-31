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
import onnx
import onnxruntime
import tempfile
import random

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


class ONNXRuntimeBackend(Backend):
    dtype_for_indices = dtypes.int64
    dtype_map = {
        slope.core.dtypes.float32: "float",
        dtypes.uint8: "uint8",
        dtypes.int8: "int8",
        dtypes.bool: "bool",
        dtypes.int32: "int32",
        dtypes.int64: "int64",
        dtypes.float16: "float16",
    }
    # https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3
    # used for impl args
    onnx_dtype_enum_map = {
        slope.core.dtypes.float32: 1,
        dtypes.uint8: 2,
        dtypes.int8: 3,
        dtypes.int32: 6,
        dtypes.int64: 7,
        dtypes.bool: 9,
        dtypes.float16: 10,
    }

    device_map = {
        devices.cpu: "cpu",
        devices.cuda0: "cuda:0",
    }

    target_map = {
        devices.cpu: "CPUExecutionProvider",
        devices.cuda0: "CUDAExecutionProvider",
    }

    dtype_map_inv = {v: k for k, v in dtype_map.items()}
    device_map_inv = {v: k for k, v in device_map.items()}

    sess_options = onnxruntime.SessionOptions()
    # Disable this flags, easily get nan
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    # Other flags
    # sess_options.log_severity_level = 3
    # sess_options.use_deterministic_compute = True
    # sess_options.intra_op_num_threads = 4
    # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    

    def from_numpy(self, val, dtype=None, device=None):
        dtype = dtype or self.DEFAULT_DTYPE
        device = device or self.DEFAULT_DEVICE
        onnx_device = self.device_map[device]
        device_type, device_id = onnx_device.split(":") if ":" in onnx_device else (onnx_device, 0)
        np_val = np.array(val, dtype=dtype.numpy)
        val = onnxruntime.OrtValue.ortvalue_from_numpy(np_val, device_type=device_type, device_id=device_id)
        return Tensor(TensorBuffer(val))
    
    def numpy_of(self, tensor: Tensor, memmap=False):
        if not memmap:
            return tensor.buf.val.numpy()
        with tempfile.NamedTemporaryFile() as arr_file:
            arr = np.memmap(arr_file.name, dtype=tensor.dtype.numpy, shape=tensor.shape, mode="w+")
            arr[:] = tensor.buf.val.numpy()
            return arr

    def shape_of(self, tensor):
        return tuple(tensor.buf.val.shape())
    
    def dtype_of(self, tensor: Tensor):
        dtype_str = tensor.buf.val.data_type().replace("tensor(", "").replace(")", "")
        return self.dtype_map_inv[dtype_str]

    def device_of(self, tensor: Tensor):
        return self.device_map_inv[tensor.buf.val.device_name()]

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
        arg_type_strs = [
                f"{self.dtype_map[inb.symval.dtype]}[{repr(list(inb.symval.shape))[1:-1]}] {inb.name}" for inb in in_binders
            ] if fn_name == "main" else [f"{inb.name}" for inb in in_binders]
        fn_args_str = ", ".join(arg_type_strs)

        outs = list_map(lambda x: program.env[x], program.outs)
        out_type_strs = [
            f"{self.dtype_map[out.symval.dtype]}[{repr(list(out.symval.shape))[1:-1]}] {out.name}" for out in outs
        ] if fn_name == "main" else [f"{out.name}" for out in outs]
        out_type_str = ", ".join(out_type_strs)

        head_code_lines = []
        if fn_name == "main":
            head_code_lines += ['<ir_version: 7, opset_import: ["" : 18, "slope":1]>']
        head_code_lines += [f"{fn_name} ({fn_args_str}) => ({out_type_str})"]
        model_code_lines = head_code_lines + ["{"] + body_code_lines + ["}"]

        functions_head_def = '<domain: "slope",  opset_import: ["" : 18, "slope":1]>'
        functions_code_lines = []
        for fn_def_code_lines in fn_defs.values():
            functions_code_lines += [functions_head_def] + fn_def_code_lines
        code_lines = model_code_lines + functions_code_lines
        slope.dblog(
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
            op_codegen_out: CodegenOut = self.codegen(
                op_program,
                args,
                fn_name=name,
                fn_defs=fn_defs,
            )
            fn_defs = {**fn_defs, **op_codegen_out.fn_defs}
            fn_defs[name] = op_codegen_out.code_lines
        in_names = ", ".join(i.name for i in in_vals)
        out_names = ", ".join(o.name for o in out_vals)
        impl_code = f"{out_names} = slope.{name}({in_names})"
        return impl_code, fn_defs

    def compile(self, codegen_out):
        code_lines = codegen_out.code_lines
        code = "\n".join(code_lines)
        model = onnx.parser.parse_model(code)
        device = codegen_out.outs[0].device
        target = self.target_map[device]
        session = onnxruntime.InferenceSession(
            model.SerializeToString(),
            self.sess_options,
            providers=[target],
        )

        def fn(*args):
            io_binding = session.io_binding()
            for a, in_binder in zip(args, codegen_out.in_binders):
                io_binding.bind_input(
                    name=in_binder.name,
                    device_type=a.device_name(),
                    device_id=0,
                    element_type=self.dtype_map_inv[a.data_type().replace("tensor(", "").replace(")", "")].numpy,
                    shape=a.shape(),
                    buffer_ptr=a.data_ptr(),
                )
            for out in codegen_out.outs:
                io_binding.bind_output(out.name, self.device_map[device])
            run_options = onnxruntime.RunOptions()
            run_options.log_severity_level = 3
            session.run_with_iobinding(io_binding, run_options)
            outputs = tuple(io_binding.get_outputs())
            return outputs

        return fn, code

    def export(self, jit_object: slope.core.JitObject, output_path, *args, **kwargs):
        code = jit_object.code
        model = onnx.parser.parse_model(code)
        os.makedirs(output_path, exist_ok=True)
        in_binders = jit_object.codegen_out["in_binders"]
        outs = jit_object.codegen_out["outs"]
        num_consts = jit_object.program.num_consts
        for i in range(num_consts):
            const_array = in_binders[i]["type"].numpy()
            const_name = in_binders[i].name
            const = onnx.numpy_helper.from_array(const_array, name=const_name)
            model.graph.initializer.append(const)

        onnx.save(model.SerializeToString(), os.path.join(output_path, "model.onnx"))
        input_arg_names = [ib.name for ib in in_binders[num_consts:]]
        input_arg_names_str = ", ".join(input_arg_names)
        outs_names = [out.name for out in outs]

        test_input_code = ""
        for i in range(num_consts, len(in_binders)):
            input_name = in_binders[i].name
            input_shape = in_binders[i]["type"].shape
            dtype = in_binders[i]["type"].dtype
            input_dtype = ("np." + dtype.numpy.__name__) if dtype is not dtypes.bool else "bool"
            test_input_code += f"""    {input_name} = np.ones({input_shape}, dtype={input_dtype})\n"""

        module_path = os.path.join(output_path, "__init__.py")
        module_code = f"""import onnxruntime
import os
import numpy as np

root_path = os.path.dirname(__file__)
model_path = os.path.join(root_path, "model.onnx")
session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_arg_names = {input_arg_names}
out_names = {outs_names}

def run(*args, **kwargs):
    if len(args) > 0:
        for a_name, a in zip(input_arg_names, args):
            assert a_name not in kwargs.keys()
            kwargs[a_name] = a
    outputs = session.run(out_names, kwargs)
    return outputs
if __name__ == "__main__":
{test_input_code}
    print("inputs:")
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


backend = ONNXRuntimeBackend(operator_set, procedure_set)


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
    impl_code = f"{', '.join(o.name for o in out_vals)} = slope.{jit_name}({args_str})"
    return impl_code, fn_defs

### Operator Impls


@backend.set_impl(operator_set.cast)
def cast_impl(self, x, y, *, dtype):
    return f"{y.name} = Cast<to={self.onnx_dtype_enum_map[dtype]}>({x.name})"


@backend.set_impl(backend.operator_set.stop_gradient)
def stop_gradient_impl(self, x, y):
    return f"{y.name} = Identity({x.name})"


@backend.set_impl(backend.operator_set.sqrt)
def sqrt_impl(self, x, y):
    return f"{y.name} = Sqrt({x.name})"


@backend.set_impl(backend.operator_set.exp)
def exp_impl(self, x, y):
    return f"{y.name} = Exp({x.name})"


@backend.set_impl(backend.operator_set.log)
def log_impl(self, x, y):
    return f"{y.name} = Log({x.name})"


@backend.set_impl(backend.operator_set.sin)
def sin_impl(self, x, y):
    return f"{y.name} = Sin({x.name})"


@backend.set_impl(backend.operator_set.invert)
def invert_impl(self, x, y):
    return f"{y.name} = Not({x.name})"


@backend.set_impl(backend.operator_set.add)
def add_impl(self, x, w, y):
    return f"{y.name} = Add({x.name}, {w.name})"


@backend.set_impl(backend.operator_set.sub)
def sub_impl(self, x, w, y):
    return f"{y.name} = Sub({x.name}, {w.name})"


@backend.set_impl(backend.operator_set.mul)
def mul_impl(self, x, w, y):
    return f"{y.name} = Mul({x.name}, {w.name})"


@backend.set_impl(backend.operator_set.div)
def div_impl(self, x, w, y):
    return f"{y.name} = Div({x.name}, {w.name})"


@backend.set_impl(backend.operator_set.pow)
def pow_impl(self, x, w, y):
    return f"{y.name} = Pow({x.name}, {w.name})"


@backend.set_impl(backend.operator_set.equal)
def equal_impl(self, x, w, y):
    return f"{y.name} = Equal({x.name}, {w.name})"


@backend.set_impl(backend.operator_set.less)
def less_impl(self, x, w, y):
    return f"{y.name} = Less({x.name}, {w.name})"


@backend.set_impl(backend.operator_set.greater)
def greater_impl(self, x, w, y):
    return f"{y.name} = Greater({x.name}, {w.name})"


@backend.set_impl(backend.operator_set.maximum)
def maximum_impl(self, x, w, y):
    return f"{y.name} = Max({x.name}, {w.name})"


@backend.set_impl(backend.operator_set.matmul)
def matmul_impl(self, x, w, y):
    return f"{y.name} = MatMul({x.name}, {w.name})"


# @backend.set_impl(backend.operator_set.where)
# def where_impl(self, x, w, u, y):
#     return f"""{y.name} = "Where"(%{x.name}, %{w.name}, %{u.name})"""


@backend.set_impl(backend.operator_set.gather_nd)
def gather_nd_impl(self, x, w, y, *, batch_dims):
    return f"{y.name} = GatherND<batch_dims={batch_dims}>({x.name}, {w.name})"
#     return (f"{y.name} = GatherND<batch_dims={batch_dims}>({x.name}, {w.name})"
# if w.symval.dtype is dtypes.int64 else
# f"""{w.name}_ = Cast<to={self.onnx_dtype_enum_map[dtypes.int64]}>({w.name})
# {y.name} = GatherND<batch_dims={batch_dims}>({x.name}, {w.name}_)
# """
# )


@backend.set_impl(backend.operator_set.scatter_nd)
def scatter_nd_impl(
    self,
    x,
    w,
    u,
    y,
):
    return f"{y.name} = ScatterND({x.name}, {w.name}, {u.name})"
    
#     if w.symval.dtype is dtypes.int64:
#         return f"{y.name} = ScatterND({x.name}, {w.name}, {u.name})"
#     else:
#         name = f"{w.name}_{random.randrange(100)}"
#         return f"""{name} = Cast<to={self.onnx_dtype_enum_map[dtypes.int64]}>({w.name})
# {y.name} = ScatterND({x.name}, {name}_, {u.name})
# """
            



@backend.set_impl(operator_set.sum)
def sum_impl(self, x, y, *, dim, keepdim):
    return f"""{y.name}_dim = Constant <value = int64[{len(dim)}]  {{ {repr(dim)[1:(-1 if len(dim) > 1 else -2)]} }} >()
{y.name} = ReduceSum<keepdims={int(keepdim)}> ({x.name}, {y.name}_dim)
"""


@backend.set_impl(operator_set.max)
def max_impl(self, x, y, *, dim, keepdim):
    return f"""{y.name}_dim = Constant <value = int64[{len(dim)}]  {{ {repr(dim)[1:(-1 if len(dim) > 1 else -2)]} }} >()
{y.name} = ReduceMax<keepdims={int(keepdim)}> ({x.name}, {y.name}_dim)
"""


@backend.set_impl(operator_set.arange)
def arange_impl(self, y, *, start, stop, stride, dtype, device):
    return f"""
{y.name}_start = Constant <value_int = {start}> ()
{y.name}_limit = Constant <value_int = {stop}> ()
{y.name}_delta = Constant <value_int = {stride}> ()
{f'''
{y.name}_range = Range({y.name}_start, {y.name}_limit, {y.name}_delta)
{y.name} = Cast<to={self.onnx_dtype_enum_map[dtype]}>({y.name}_range)
''' if dtype is not dtypes.int64 else
f'''
{y.name} = Range({y.name}_start, {y.name}_limit, {y.name}_delta)
'''
}
"""


# {y.name}_range = Range({y.name}_start, {y.name}_limit, {y.name}_delta)
# {f'{y.name} = Cast<to={self.onnx_dtype_enum_map[dtype]}>({y.name}_range)'}
@backend.set_impl(operator_set.full)
def full_impl(self, y, *, shape, fill_value, dtype, device):
    if dtype is not dtypes.bool:
        if dtypes.is_float(dtype):
            fill_value = float(fill_value)
        elif dtypes.is_int(dtype):
            fill_value = int(fill_value)

        if len(shape) > 0:
            return f"""{y.name}_fill_value = Constant < value = {self.dtype_map[dtype]}[1] {{ {fill_value} }}>()
{y.name}_shape = Constant <value = int64[{len(shape)}] {{ {repr(list(shape))[1:-1]} }} >()
{y.name} = Expand ({y.name}_fill_value, {y.name}_shape)
"""
        else:  # scalar case
            return f"""{y.name}_fill_value = Constant < value = {self.dtype_map[dtype]}[1] {{ {fill_value} }}>()
{y.name}_squeeze_dim = Constant <value = int64[1] {{0}}> ()
{y.name} = Squeeze ({y.name}_fill_value, {y.name}_squeeze_dim)
"""
    else:
        if len(shape) > 0:
            return f"""{y.name}_fill_value = Constant < value = int64[1] {{ {int(fill_value)} }}>()
{y.name}_shape = Constant <value = int64[{len(shape)}] {{ {repr(list(shape))[1:-1]} }} >()
{y.name}_expand = Expand ({y.name}_fill_value, {y.name}_shape)
{y.name} = Cast<to={self.onnx_dtype_enum_map[dtype]}>({y.name}_expand)
"""
        else:  # scalar case
            return f"""{y.name}_fill_value = Constant < value = int64[1] {{ {int(fill_value)} }}>()
{y.name}_squeeze_dim = Constant <value = int64[1] {{0}}> ()
{y.name}_squeeze = Squeeze ({y.name}_fill_value, {y.name}_squeeze_dim)
{y.name} = Cast<to={self.onnx_dtype_enum_map[dtype]}>({y.name}_squeeze)
"""


@backend.set_impl(operator_set.random_uniform)
def random_uniform_impl(self, y, *, shape, dtype, device):
    if len(shape) > 0:
        return f"""
{y.name} = RandomUniform<dtype={self.onnx_dtype_enum_map[dtype]},shape={repr(list(shape))}>()
"""
    else:  # scalar case
        return f"""
{y.name}_rand = RandomUniform<dtype={self.onnx_dtype_enum_map[dtype]}, shape=[1]>()
{y.name}_squeeze_dim = Constant <value = int64[1] {{0}}> ()
{y.name} = Squeeze ({y.name}_rand, {y.name}_squeeze_dim)
"""


@backend.set_impl(operator_set.random_normal)
def random_normal_impl(self, y, *, shape, dtype, device):
    if len(shape) > 0:
        return f"""
{y.name} = RandomNormal<dtype={self.onnx_dtype_enum_map[dtype]}, shape={repr(list(shape))}>()
"""
    else:  # scalar case
        return f"""
{y.name}_randn = RandomNormal<dtype={self.onnx_dtype_enum_map[dtype]}, shape=[1]>()
{y.name}_squeeze_dim = Constant <value = int64[1] {{0}}> ()
{y.name} = Squeeze ({y.name}_randn, {y.name}_squeeze_dim)
"""


@backend.set_impl(operator_set.expand)
def expand_impl(self, x, y, *, shape):
    return f"""
{y.name}_shape = Constant <value = int64[{len(shape)}] {{ {repr(list(shape))[1:-1]} }} >()
{y.name} = Expand ({x.name}, {y.name}_shape)
"""


@backend.set_impl(operator_set.reshape)
def reshape_impl(self, x, y, *, shape):
    if len(shape) > 0:
        return f"""
{y.name}_shape = Constant <value = int64[{len(shape)}] {{ {repr(list(shape))[1:-1]} }} >()
{y.name} = Reshape({x.name}, {y.name}_shape)
"""
    else:  # scalar case
        f"""
        {y.name}_shape = Constant <value = int64[1] {1} >()
        {y.name}_reshape = Reshape({x.name}, {y.name}_shape)
        {y.name}_squeeze_dim = Constant <value = int64[1] {{0}}> ()
        {y.name} = Squeeze ({y.name}_reshape, {y.name}_squeeze_dim)"""


@backend.set_impl(operator_set.pad)
def pad_impl(self, x, y, *, padding, mode, value):
    padding = padding[::-1]
    padding = padding[0::2] + padding[1::2]
    #     return f"""
    # {y.name}_padding = Constant <value = int64[{len(padding)}]  {{ {repr(list(padding))[1:-1]} }}>()
    # {y.name}_constant_value =  Constant <value = {value} >()
    # {y.name} = Pad({x.name}, {y.name}_padding, {y.name}_constant_value)
    # """
    return f"""
{y.name}_padding = Constant <value = int64[{len(padding)}]  {{ {repr(list(padding))[1:-1]} }}>()
{y.name} = Pad({x.name}, {y.name}_padding)
"""


@backend.set_impl(operator_set.slice)
def slice_impl(self, x, y, *, starts, limits, strides):
    return f"""
{y.name}_starts = Constant <value = int64[{len(starts)}]  {{ {repr(list(starts))[1:-1]} }}>()
{y.name}_ends = Constant <value = int64[{len(limits)}]  {{ {repr(list(limits))[1:-1]} }}>()
{y.name}_dim = Constant <value = int64[{len(strides)}]  {{ {repr(list(range(len(starts))))[1:-1]} }}>()
{y.name}_steps = Constant <value = int64[{len(strides)}]  {{ {repr(list(strides))[1:-1]} }}>()
{y.name} = Slice({x.name}, {y.name}_starts, {y.name}_ends, {y.name}_dim, {y.name}_steps)
"""


@backend.set_impl(operator_set.cat)
def cat_impl(self, *xs, dim):
    xs, y = xs[:-1], xs[-1]
    return f"{y.name} = Concat< axis={dim}>({','.join([x.name for x in xs])})"


@backend.set_impl(operator_set.permute)
def permute_impl(self, x, y, *, perm):
    return f"{y.name} = Transpose<perm={repr(list(perm))}>({x.name})"


@backend.set_impl(operator_set.flip)
def flip_impl(self, x, y, *, dim):
    padding = [0, 0] * x.symval.ndim
    for d in dim:
        padding[d] = 1
    return f"""{x.name}_padding = Constant <value = int64[{len(padding)}]  {{ {repr(list(padding))[1:-1]} }}>()
{x.name}_ = Pad({x.name}, {x.name}_padding)
{y.name}_starts = Constant <value = int64[{len(dim)}]  {{ {", ".join([str(x.symval.shape[d]) for d in dim])} }}>()
{y.name}_ends = Constant <value = int64[{len(dim)}] {{ {", ".join(["0"] * len(dim))} }}>()
{y.name}_dim = Constant <value = int64[{len(dim)}]  {{ {repr(list(dim))[1:-1]} }}>()
{y.name}_steps = Constant <value = int64[{len(dim)}] {{ {", ".join(["-1"] * len(dim))} }}>()
{y.name} = Slice({x.name}_, {y.name}_starts, {y.name}_ends, {y.name}_dim, {y.name}_steps)
"""




@backend.set_impl(operator_set.conv)
def conv_impl(self, x, w, y, *, groups, stride, dilation, padding):
    padding = padding[0::2] + padding[1::2]
    dilations_attr = f"dilations=[{repr(list(dilation))[1:-1]}]"
    pads_attr = f"pads=[{repr(list(padding))[1:-1]}]"
    strides_attr = f"strides=[{repr(list(stride))[1:-1]}]"
    group_attr = f"group={groups}"
    return f"""{y.name} = Conv<{dilations_attr}, {pads_attr}, {strides_attr}, {group_attr}>({x.name}, {w.name})"""


# ---- conv_shape__lp_100_cm_64_cm_15_cm_15_rp__dtype_float32_shape__lp_32_cm_64_cm_3_cm_3_rp__dtype_float32_groups_1_stride__lp_1_cm_1_rp__dilation__lp_1_cm_1_rp__padding__lp_1_cm_2_cm_1_cm_2_rp__ codegen:

# func.func @main (%x0: tensor<100x64x15x15xf32>, %x1: tensor<32x64x3x3xf32>) -> (tensor<100x32x16x16xf32>)
# {
#     %y0 = "stablehlo.convolution"(%x0, %x1) {
#       window_strides = dense<[1, 1]> : tensor<2xi64>,
#       padding = dense<[[1, 2], [1, 2]]> : tensor<2x2xi64>,
#       lhs_dilation = dense<1> : tensor<2xi64>,
#       rhs_dilation = dense<[1, 1]> : tensor<2xi64>,
#       window_reversal = dense<false> : tensor<2xi1>,
#       dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
#       feature_group_count = 1 : i64,
#       batch_group_count = 1 : i64,
#       precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
#     }  : (tensor<100x64x15x15xf32>,tensor<32x64x3x3xf32>) -> (tensor<100x32x16x16xf32>)
    
#     "func.return"(%y0): (tensor<100x32x16x16xf32>) -> ()
# }

# ===========

# (100, 64, 15, 15) (32, 64, 3, 3) (100, 32, 16, 16)

# ---- conv_shape__lp_100_cm_64_cm_15_cm_15_rp__dtype_float32_shape__lp_32_cm_64_cm_3_cm_3_rp__dtype_float32_groups_1_stride__lp_1_cm_1_rp__dilation__lp_1_cm_1_rp__padding__lp_1_cm_2_cm_1_cm_2_rp__ codegen:

# <ir_version: 7, opset_import: ["" : 18, "slope":1]>
# main (float[100, 64, 15, 15] x0, float[32, 64, 3, 3] x1) => (float[100, 32, 16, 16] y0)
# {
#     y0 = Conv<dilations=[1, 1], pads=[1, 2, 1, 2], strides=[1, 1], group=1>(x0, x1)
# }