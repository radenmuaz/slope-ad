import slope
import slope.core
from slope.core import (
    Backend,
    Operator,
    MetaOperator,
    OperatorSet,
    ProcedureSet,
    Tensor,
    TensorBuffer,
    SymbolicTensor,
    list_zip,
    list_map,
    dtypes,
    devices,
)

import math
import numpy as np
from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Optional,
    Sequence,
    Union,
    Iterator,
    NamedTuple,
)
from collections import defaultdict
import iree.compiler
import iree.runtime
import os

from slope.operators import operator_set
from slope.procedures import procedure_set

max_py = max
abs_py = abs
slice_py = slice

compile_py = compile


def as_mlir_shape(symval):
    xdtype = symval.dtype.mlir
    if len(symval.shape) > 0:
        xshape = f"{'x'.join((repr(i) for i in symval.shape))}"
        return f"tensor<{xshape}x{xdtype}>"
    else:
        return f"tensor<{xdtype}>"


def as_mlir_sig(in_symvals, out_symval):
    typing_code = (
        f" : ({','.join(as_mlir_shape(t) for t in in_symvals)}) -> {as_mlir_shape(out_symval)}"
    )
    return typing_code


class IREEBackend(Backend):
    def from_numpy(self, val, dtype=None, device=None):
        dtype = dtype or self.DEFAULT_DTYPE
        device = device or self.DEFAULT_DEVICE
        np_val = np.array(val, dtype=dtype.numpy)
        iree_device = iree.runtime.get_device(self.device_map[device])
        val = iree.runtime.asdevicearray(iree_device, np_val)
        return Tensor(TensorBuffer(val))

    def numpy_of(self, tensor):
        return tensor.buf.val.to_host()

    def device_of(self, tensor):
        return self.device_map_inv[str(tensor.buf.val._device)]

    def shape_of(self, tensor):
        return tuple(tensor.buf.val.shape)

    def dtype_of(self, tensor):
        return self.dtype_map_inv[tensor.buf.val.dtype]

    def export(self, jit_object, output_path, *args, **kwargs):
        code = jit_object.code
        model = onnx.parser.parse_model(code)
        os.makedirs(output_path, exist_ok=True)
        in_binders = jit_object.codegen_out["in_binders"]
        outs = jit_object.codegen_out["outs"]
        num_consts = jit_object.program.num_consts
        for i in range(num_consts):
            const_array = in_binders[i].symval.numpy()
            const_name = in_binders[i].name
            const = onnx.numpy_helper.from_array(const_array, name=const_name)
            model.graph.initializer.append(const)
            # TODO: try if need these
            # const_tensor = next(t for t in model.graph.input if t.name == const_name)
            # const_tensor.type.tensor_type.shape.dim[0].dim_param = const_name
            # const_tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

        onnx.save(model.SerializeToString(), os.path.join(output_path, "model.onnx"))
        input_arg_names = [ib.name for ib in in_binders[num_consts:]]
        input_arg_names_str = ", ".join(input_arg_names)
        outs_names = [out.name for out in outs]

        test_input_code = ""
        for i in range(num_consts, len(in_binders)):
            input_name = in_binders[i].name
            input_shape = in_binders[i].symval.shape
            dtype = in_binders[i].symval.dtype
            input_dtype = ("np." + dtype.numpy.__name__) if dtype is not Tensor.bool else "bool"
            test_input_code += (
                f"""    {input_name} = np.ones({input_shape}, dtype={input_dtype})\n"""
            )

        module_path = os.path.join(output_path, "__init__.py")
        module_code = f"""import iree_runtime
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

    def compile(self, codegen_out):
        code_lines = codegen_out["code_lines"]
        code = "\n".join(code_lines)
        instance = iree.runtime.VmInstance()
        iree_device = iree.runtime.get_device("local-task")
        hal_module = iree.runtime.create_hal_module(instance, iree_device)
        # iree.compiler.core.DEFAULT_TESTING_BACKENDS
        binary = iree.compiler.compile_str(
            code,
            target_backends=("llvm-cpu",),
        )
        m = iree.runtime.VmModule.from_flatbuffer(instance, binary)
        context = iree.runtime.VmContext(instance, modules=[hal_module, m])
        f = m.lookup_function("main")
        finv = iree.runtime.FunctionInvoker(context, iree_device, f, tracer=None)
        return finv, code

    def codegen(self, program, args, *, fn_name: str = "main", fn_defs=dict()) -> List[Any]:
        if fn_name == "main":
            assert not hasattr(self, "fn_count")
            self.fn_count = 0

        def indent(code, amount):
            spaces = " " * (len(code) - len(code.lstrip()))
            spaces += " " * amount
            return "\n".join([spaces + line for line in code.strip().split("\n")])

        # codegen is recursive if jit-of-jit happens
        il1 = 4  # indent length
        body_code_lines = []

        for instruction in program.instructions:
            if len(instruction.out_binders) == 0:  # skip codegen for function returns nothing
                continue
            in_vals = list_map(lambda x: program.env[x], instruction.inputs)
            out_vals = list_map(lambda z: program.env[z], instruction.out_binders)
            if isinstance(instruction.op, MetaOperator):
                impl_code, fn_defs = self.impls[instruction.op](
                    args, instruction, fn_defs, in_vals, out_vals
                )
            else:
                if instruction.op in self.impls.keys():
                    impl_code = self.impls[instruction.op](
                        *in_vals, *out_vals, **instruction.params
                    )
                else:
                    # No impl is defined, fallback to procedure
                    op = instruction.op
                    op_name = {v: k for k, v in vars(self.operator_set.items())}[op]
                    op_procedure = getattr(self.procedure_set, op_name)
                    symvals_in = tuple(inp.symval for inp in instruction.inputs)
                    params = instruction.params
                    op_program, consts, _ = slope.core.make_program(
                        op_procedure,
                        *symvals_in,
                        static_args=tuple(params.items()),
                        name=op.name,
                    )
                    name = op.get_jit_name(tuple(symvals_in), params)
                    if name not in fn_defs.keys():
                        op_codegen_out = self.codegen(
                            op_program,
                            args,
                            fn_name=name,
                            fn_defs=fn_defs,
                        )
                        fn_defs = {**fn_defs, **op_codegen_out["fn_defs"]}
                        fn_defs[name] = op_codegen_out["code_lines"]
                    in_names = ", ".join(i.name for i in in_vals)
                    out_names = ", ".join(o.name for o in out_vals)
                    sig = as_mlir_sig(tuple(i.symval for i in in_vals), out_vals[0].symval)
                    impl_code = f"{out_names} = func.call @{name}({in_names}) {sig}"
            for impl_code_line in impl_code.split("\n"):  # handle multi-line code
                body_code_lines += [indent(impl_code_line, il1)]

        # inb_consts = [v for v in env.values() if "c" in v.name]
        # const_type_strs = [f"{self.dtype_map[c['type'].dtype]}[{repr(c['type'].shape)[1:-1]}] {c['name']}" for c in inb_consts]

        in_binders = list_map(lambda x: program.env[x], program.in_binders)
        fn_args_str = ", ".join([f"%{i.name}: {as_mlir_shape(i.symval)}" for i in in_binders])

        outs = list_map(
            lambda x: program.env[x], program.outs
        )  # TODO: input that is output should has identity op
        out_str = ", ".join([f"%{o.name}" for o in outs])
        out_type_str = ", ".join([f"{as_mlir_shape(o.symval)}" for o in outs])

        head_code_line = [f"func.func @{fn_name} ({fn_args_str}) -> ({out_type_str})"]
        tail_code_line = [indent(f'"func.return"({out_str}): ({out_type_str}) -> ()', il1)]
        model_code_lines = head_code_line + ["{"] + body_code_lines + tail_code_line + ["}"]

        functions_code_lines = []
        for op, fn_def_code_lines in fn_defs.items():
            functions_code_lines += fn_def_code_lines
        code_lines = model_code_lines + functions_code_lines
        slope.core.dblog(
            f"\n---- {program.name} codegen:\n\n" + "\n".join(code_lines) + "\n\n===============\n",
            enable=slope.LOG_JIT,
        )

        if fn_name == "main":
            del self.fn_count
        assert len(outs) == len(program.outs)
        return dict(
            code_lines=code_lines,
            fn_defs=fn_defs,
            in_binders=in_binders,
            outs=outs,
        )


backend = IREEBackend(
    operator_set,
    procedure_set,
    {
        dtypes.float32: np.dtypes.Float32DType(),
        dtypes.uint8: np.dtypes.UInt8DType(),
        dtypes.int8: np.dtypes.Int8DType(),
        dtypes.bool: np.dtypes.BoolDType(),
        dtypes.int32: np.dtypes.Int32DType(),
        dtypes.int64: np.dtypes.Int64DType(),
        dtypes.float16: np.dtypes.Float16DType(),
    },
    {
        devices.cpu: "local-task",
        devices.cuda0: "cuda:0",
    },
)


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
        fn_defs[jit_name] = jit_codegen_out["code_lines"]
        fn_defs = {**fn_defs, **jit_codegen_out["fn_defs"]}
    args_str = ", ".join(i.name for i in in_vals)
    sig = as_mlir_sig(tuple(i.symval for i in in_vals), out_vals[0].symval)
    ret = f"{', '.join(o['name'] for o in out_vals)} = func.call @{jit_name}({args_str}) {sig}"
    return ret, fn_defs


@backend.set_impl(backend.operator_set.cast)
def cast_impl(self, x, y, *, dtype):
    return f'%{y.name} = "stablehlo.convert"(%{x.name}) {as_mlir_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.stop_gradient)
def stop_gradient_impl(self, x, y):
    return f'%{y.name} = "stablehlo.optimization_barrier"(%{x.name}){as_mlir_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.sqrt)
def sqrt_impl(self, x, y):
    return f'%{y.name} = "stablehlo.sqrt"(%{x.name}) {as_mlir_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.exp)
def exp_impl(self, x, y):
    return f'%{y.name} = "stablehlo.exponential"(%{x.name}) {as_mlir_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.log)
def log_impl(self, x, y):
    return f'%{y.name} = "stablehlo.log"(%{x.name}) {as_mlir_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.sin)
def sin_impl(self, x, y):
    return f'%{y.name} = "stablehlo.sine"(%{x.name}) {as_mlir_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.invert)
def invert_impl(self, x, y):
    return f'%{y.name} = "stablehlo.not"(%{x.name}) {as_mlir_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.add)
def add_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.add"(%{x.name}, %{w.name}) {as_mlir_sig((x.symval, w.symval), y.symval)}'


@backend.set_impl(backend.operator_set.sub)
def sub_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.subtract"(%{x.name}, %{w.name}) {as_mlir_sig((x.symval, w.symval), y.symval)}'


@backend.set_impl(backend.operator_set.mul)
def mul_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.multiply"(%{x.name}, %{w.name}) {as_mlir_sig((x.symval, w.symval), y.symval)}'


@backend.set_impl(backend.operator_set.div)
def div_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.divide"(%{x.name}, %{w.name}) {as_mlir_sig((x.symval, w.symval), y.symval)}'


@backend.set_impl(backend.operator_set.pow)
def pow_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.power"(%{x.name}, %{w.name}) {as_mlir_sig((x.symval, w.symval), y.symval)}'


@backend.set_impl(backend.operator_set.equal)
def equal_impl(self, x, w, y):
    return f"""%{y.name} = "stablehlo.compare"(%{x.name}, %{w.name}) {{
  comparison_direction = #stablehlo<comparison_direction EQ>,
  compare_type = #stablehlo<comparison_type FLOAT>
}}  {as_mlir_sig((x.symval, w.symval), y.symval)}
"""


@backend.set_impl(backend.operator_set.maximum)
def maximum_impl(self, x, w, y):
    return f'%{y.name} = "stablehlo.maximum"(%{x.name}, %{w.name}) {as_mlir_sig((x.symval, w.symval), y.symval)}'


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
    }}  {as_mlir_sig((x.symval, w.symval), y.symval)}
"""


@backend.set_impl(backend.operator_set.sum)
def sum_impl(self, x, y, *, dim, keepdim):
    zero = "0." if "f" in y.symval.dtype.mlir else "0"
    y_init_type = SymbolicTensor((), y.symval.dtype, y.symval.device)
    y_mlir_type = as_mlir_shape(y_init_type)
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
%{y.name}_init = stablehlo.constant dense<{zero}> : {as_mlir_shape(y_init_type)}
%{y.name}{'_' if keepdim else ''} = "stablehlo.reduce"(%{x.name}, %{y.name}_init) ({{
  ^bb0(%arg0: {y_mlir_type}, %arg1: {y_mlir_type}):
    %0 = "stablehlo.add"(%arg0, %arg1) {as_mlir_sig((y_init_type, y_init_type), y_init_type)}
    "stablehlo.return"(%0) : ({y_mlir_type}) -> ()
}}) {{
  dimensions = dense<{repr(list(dim))}> : tensor<{len(dim)}xi64>
}} {as_mlir_sig((x.symval, y_init_type), y_out_type)}
{f'%{y.name} = "stablehlo.reshape"(%{y.name}_) {as_mlir_sig((y_out_type,), y.symval)}' if keepdim else ''}"""


@backend.set_impl(backend.operator_set.max)
def max_impl(self, x, y, *, dim, keepdim):
    min_val = {
        dtypes.float32: "1.E-38",
        dtypes.int8: "-128",
        dtypes.int32: "-65536",
    }[x.symval.dtype]
    y_init_type = SymbolicTensor((), y.symval.dtype, y.symval.device)
    y_mlir_type = as_mlir_shape(y_init_type)
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
%{y.name}_init = stablehlo.constant dense<{min_val}> : {as_mlir_shape(y_init_type)}
%{y.name}{'_' if keepdim else ''} = "stablehlo.reduce"(%{x.name}, %{y.name}_init) ({{
  ^bb0(%arg0: {y_mlir_type}, %arg1: {y_mlir_type}):
    %0 = "stablehlo.maximum"(%arg0, %arg1) {as_mlir_sig((y_init_type, y_init_type), y_init_type)}
    "stablehlo.return"(%0) : ({y_mlir_type}) -> ()
}}) {{
  dimensions = dense<{repr(list(dim))}> : tensor<{len(dim)}xi64>
}} {as_mlir_sig((x.symval, y_init_type), y_out_type)}
{f'%{y.name} = "stablehlo.reshape"(%{y.name}_) {as_mlir_sig((y_out_type,), y.symval)}' if keepdim else ''}
"""


@backend.set_impl(backend.operator_set.arange)
def arange_impl(self, y, *, start, stop, stride, dtype, device):
    ret = ""
    if stride == 1 and start == 0:
        ret += f"""%{y.name} = "stablehlo.iota"() {{iota_dimension = 0 : i64}} {as_mlir_sig((), y.symval)}"""
    else:
        ret += f"""%{y.name}_ = "stablehlo.iota"() {{iota_dimension = 0 : i64}} {as_mlir_sig((), y.symval)}"""
    return ret


@backend.set_impl(backend.operator_set.full)
def full_impl(self, y, *, shape, fill_value, dtype, device):
    fill_value = float(fill_value) if "f" in dtype.mlir else int(fill_value)
    fill_value = repr(fill_value)
    fill_value = (
        fill_value.replace("e", "E") if "." in fill_value else fill_value.replace("e", ".E")
    )
    return f'%{y.name} = "stablehlo.constant"() {{ value = dense<{fill_value}> : {as_mlir_shape(y.symval)} }} {as_mlir_sig((), y.symval)}'


@backend.set_impl(backend.operator_set.random_uniform)
def random_uniform_impl(self, y, *, shape, dtype, device):
    zero = "0." if "f" in y.symval.dtype.mlir else "0"
    one = "1." if "f" in y.symval.dtype.mlir else "1"
    a_type = b_type = SymbolicTensor((), dtype)
    is_scalar = shape == ()
    shape_val = f'dense<{repr(list(shape)) if not is_scalar else "[1]"}'
    shape_type = SymbolicTensor((1,) if is_scalar else (len(shape),), Tensor.int64)
    y_out_type = y.symval if not is_scalar else SymbolicTensor((1,), y.symval.dtype)
    return f"""%{y.name}_a = stablehlo.constant dense<{zero}> : {as_mlir_shape(a_type)}
%{y.name}_b = stablehlo.constant dense<{one}> : {as_mlir_shape(b_type)}
%{y.name}_shape = stablehlo.constant {shape_val}> : {as_mlir_shape(shape_type)}
%{y.name}{'_' if is_scalar else ''} = "stablehlo.rng"(%{y.name}_a, %{y.name}_b,%{y.name}_shape) {{
        rng_distribution = #stablehlo<rng_distribution UNIFORM>}} {as_mlir_sig((a_type, b_type, shape_type), y_out_type)}
{f'%{y.name} = "stablehlo.reshape"(%{y.name}_) {as_mlir_sig((y_out_type,), y.symval)}' if is_scalar else ''}"""


@backend.set_impl(backend.operator_set.random_normal)
def random_normal_impl(self, y, *, shape, dtype, device):
    zero = "0." if "f" in y.symval.dtype.mlir else "0"
    one = "1." if "f" in y.symval.dtype.mlir else "1"
    a_type = b_type = SymbolicTensor((), dtype, device)
    is_scalar = shape == ()
    shape_val = f'dense<{repr(list(shape)) if not is_scalar else "[1]"}'
    shape_type = SymbolicTensor((1,) if is_scalar else (len(shape),), slope.dtypes.int64, device)
    y_out_type = (
        y.symval if not is_scalar else SymbolicTensor((1,), y.symval.dtype, y.symval.device)
    )
    return f"""%{y.name}_a = stablehlo.constant dense<{zero}> : {as_mlir_shape(a_type)}
%{y.name}_b = stablehlo.constant dense<{one}> : {as_mlir_shape(b_type)}
%{y.name}_shape = stablehlo.constant {shape_val}> : {as_mlir_shape(shape_type)}
%{y.name}{'_' if is_scalar else ''} = "stablehlo.rng"(%{y.name}_a, %{y.name}_b,%{y.name}_shape) {{
        rng_distribution = #stablehlo<rng_distribution NORMAL>}} {as_mlir_sig((a_type, b_type, shape_type), y_out_type)}
{f'%{y.name} = "stablehlo.reshape"(%{y.name}_) {as_mlir_sig((y_out_type,), y.symval)}' if is_scalar else ''}"""


@backend.set_impl(backend.operator_set.expand)
def expand_impl(self, x, y, *, shape):
    return f"""%{y.name} = "stablehlo.broadcast_in_dim"(%{x.name}) {{
        broadcast_dimensions = dense<{repr(list(range(len(shape))))}>: tensor<{len(shape)}xi64>
        }} {as_mlir_sig(( x.symval,), y.symval)}
"""


@backend.set_impl(backend.operator_set.reshape)
def reshape_impl(self, x, y, *, shape):
    return f'%{y.name} = "stablehlo.reshape"(%{x.name}) {as_mlir_sig((x.symval,), y.symval)}'


@backend.set_impl(backend.operator_set.pad)
def pad_impl(self, x, y, *, padding, mode, value):
    value = float(value) if "f" in x.symval.dtype.mlir else int(value)
    value_type = SymbolicTensor((), x.symval.dtype, x.symval.device)
    lo = padding[0::2][::-1]
    hi = padding[1::2][::-1]
    return f"""%{y.name}_value = stablehlo.constant dense<{value}> : {as_mlir_shape(value_type)}
%{y.name} = "stablehlo.pad"(%{x.name}, %{y.name}_value) {{
  edge_padding_low = dense<{repr(list(lo))}> : tensor<{len(lo)}xi64>,
  edge_padding_high = dense<{repr(list(hi))}> : tensor<{len(hi)}xi64>,
  interior_padding = dense<{repr([0]*len(lo))}> : tensor<{len(lo)}xi64>
}} {as_mlir_sig((x.symval, value_type), y.symval)}
"""


@backend.set_impl(backend.operator_set.slice)
def slice_impl(self, x, y, *, starts, limits, strides):
    return f"""%{y.name} = "stablehlo.slice"(%{x.name}) {{
  start_indices = dense<{repr(list(starts))}> : tensor<{len(starts)}xi64>,
  limit_indices = dense<{repr(list(limits))}> : tensor<{len(limits)}xi64>,
  strides = dense<{repr(list(strides))}> : tensor<{len(strides)}xi64>
}} {as_mlir_sig((x.symval,), y.symval)}
"""


@backend.set_impl(backend.operator_set.cat)
def cat_impl(self, *xs, dim):
    xs, y = xs[:-1], xs[-1]
    return f"""%{y.name} = "stablehlo.concatenate"({', '.join([f'%{x.name}' for x in xs])}) {{
 dimension = {dim} : i64
}} {as_mlir_sig(([x.symval for x in xs]), y.symval)}"""


@backend.set_impl(backend.operator_set.permute)
def permute_impl(self, x, y, *, perm):
    return f"""%{y.name} = "stablehlo.transpose"(%{x.name}) {{
  permutation = dense<{repr(list(perm))}> : tensor<{len(perm)}xi64>
}} {as_mlir_sig((x.symval,), y.symval)}"""


@backend.set_impl(backend.operator_set.flip)
def flip_impl(self, x, y, *, dim):
    return f"""%{y.name} = "stablehlo.reverse"(%{x.name}) {{
  dimensions = dense<{repr(list(dim))}> : tensor<{len(dim)}xi64>
}}  {as_mlir_sig((x.symval,), y.symval)}
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
}}  {as_mlir_sig((x.symval, w.symval), y.symval)}
"""

def translate_to_stablehlo_gather(data, indices, batch_dims):
    # Step 1: Extract necessary information from inputs
    rank_data = len(data.shape)
    rank_indices = len(indices.shape)
    batch_dim_sizes = indices.shape[:batch_dims]
    offset_dim_sizes = indices.shape[batch_dims:-1]
    offset_dims = list(range(rank_data - len(offset_dim_sizes), rank_data))
    collapsed_slice_dims = [batch_dims]
    index_vector_dim = rank_indices - 1

    # Step 2: Create slice_sizes tensor
    slice_sizes = np.ones(rank_data, dtype=np.int64)
    slice_sizes[offset_dims] = offset_dim_sizes

    # Step 3: Create start_index_map tensor
    start_index_map = np.arange(index_vector_dim, dtype=np.int64)

    # Step 4: Create offset_dims tensor
    offset_dims_tensor = np.array(offset_dims, dtype=np.int64)

    # Step 5: Gather operation using stablehlo.gather
    result = np.einsum("ijk->ijkl", np.take(data, indices, axis=batch_dims))

    return result, batch_dim_sizes, offset_dim_sizes, offset_dims_tensor, collapsed_slice_dims, start_index_map, index_vector_dim, slice_sizes


@backend.set_impl(backend.operator_set.gather_nd)
def gather_nd_impl(self, x, w, y, *, batch_dims):
    offset_dims = list(range(batch_dims + 1, w.symval.ndim))
    squeeze_after = w.symval.shape[-1] == x.symval.ndim - batch_dims
    if not squeeze_after:
        # collapsed_slice_dims = list(range(len(w.symval.shape[:-1])))
        # start_index_map = list(range(w.symval.shape[-1]))
        start_index_map = collapsed_slice_dims = list(
            range(batch_dims, w.symval.shape[-1] + batch_dims)
        )
        slice_sizes = [1] * len(x.symval.shape[:-1]) + [x.symval.shape[-1]]

        return f"""%{y.name} = "stablehlo.gather"(%{x.name}, %{w.name}) {{
  dimension_numbers = #stablehlo.gather<
  offset_dims = {offset_dims},
  collapsed_slice_dims = {collapsed_slice_dims},
  start_index_map = {start_index_map},
  index_vector_dim = {batch_dims+1}>,
  slice_sizes = dense<{slice_sizes}> : tensor<{len(slice_sizes)}xi64>,
  indices_are_sorted = false
}} {as_mlir_sig((x.symval, w.symval), y.symval)}
"""
    else:
        start_index_map = list(range(batch_dims, w.symval.shape[-1]))
        collapsed_slice_dims = list(range(x.symval.ndim - 1))
        slice_sizes = [1] * x.symval.shape[-1]
        y_symval = SymbolicTensor(y.symval.shape + (1,), y.symval.dtype, y.symval.device)
        return f"""%{y.name}_ = "stablehlo.gather"(%{x.name}, %{w.name}) {{
  dimension_numbers = #stablehlo.gather<
  offset_dims = {offset_dims},
  collapsed_slice_dims = {collapsed_slice_dims},
  start_index_map = {start_index_map},
  index_vector_dim = {batch_dims+1}>,
  slice_sizes = dense<{slice_sizes}> : tensor<{len(slice_sizes)}xi64>,
  indices_are_sorted = false
}} {as_mlir_sig((x.symval, w.symval), y_symval)}
%{y.name} = "stablehlo.reshape"(%{y.name}_) {as_mlir_sig((y_symval,), y.symval)}
"""
        

    # OK
    # func.func @main (%x0: tensor<2x2x2xf32>, %x1: tensor<2x2xi32>) -> (tensor<2x2xf32>)
    # {
    #     %y0 = "stablehlo.gather"(%x0, %x1) {
    #       dimension_numbers = #stablehlo.gather<
    #       offset_dims = [1],
    #       collapsed_slice_dims = [0, 1],
    #       start_index_map = [0, 1],
    #       index_vector_dim = 1>,
    #       slice_sizes = dense<[1, 1, 2]> : tensor<3xi64>,
    #       indices_are_sorted = false
    #     }  : (tensor<2x2x2xf32>,tensor<2x2xi32>) -> tensor<2x2xf32>

    #     "func.return"(%y0): (tensor<2x2xf32>) -> ()
    # }

    offset_dims = [1]
    start_index_map = [0, 1]
    collapsed_slice_dims = [0, 1]
    slice_sizes = [1, 1, 2]
    return f"""%{y.name} = "stablehlo.gather"(%{x.name}, %{w.name}) {{
  dimension_numbers = #stablehlo.gather<
  offset_dims = {offset_dims},
  collapsed_slice_dims = {collapsed_slice_dims},
  start_index_map = {start_index_map},
  index_vector_dim = {batch_dims+1}>,
  slice_sizes = dense<{slice_sizes}> : tensor<{len(slice_sizes)}xi64>,
  indices_are_sorted = false
}} {as_mlir_sig((x.symval, w.symval), y.symval)}
"""


@backend.set_impl(backend.operator_set.scatter_nd)
def scatter_nd_impl(self, x, w, u, y, *, batch_dims):
    y_init_type = SymbolicTensor((), y.symval.dtype, y.symval.device)
    y_mlir_type = as_mlir_shape(y_init_type)
    lim = (
        batch_dims + (len(w.symval.shape[batch_dims + 1 :])) - len(x.symval.shape[: batch_dims + 1])
    )
    lim = None if lim == 0 else lim
    update_window_dims = list(x.symval.shape[(batch_dims + 1) : lim])
    inserted_window_dims = [0]
    scatter_dims_to_operand_dims = [0]
    one = 1.0 if "f" in x.symval.dtype.mlir else 1
    # TODO: Find cheaper way to copy if exists
    return f"""%{x.name}_1 = "stablehlo.constant"(){{ value = dense<{one}> : {as_mlir_shape(x.symval)} }} {as_mlir_sig((), x.symval)}
%{x.name}_ = "stablehlo.multiply"(%{x.name}, %{x.name}_1) {as_mlir_sig((x.symval,  x.symval), x.symval)}
%{y.name} = "stablehlo.scatter"(%{x.name}_, %{w.name}, %{u.name}) ({{
  ^bb0(%arg0: {y_mlir_type}, %arg1: {y_mlir_type}):
    %0 = "stablehlo.add"(%arg0, %arg1) {as_mlir_sig((y_init_type, y_init_type), y_init_type)}
    "stablehlo.return"(%0) : ({y_mlir_type}) -> ()
}}) {{
  scatter_dimension_numbers = #stablehlo.scatter<
  update_window_dims = {update_window_dims},
  inserted_window_dims = {inserted_window_dims},
  scatter_dims_to_operand_dims = {scatter_dims_to_operand_dims},
  index_vector_dim = {batch_dims+1}>,
  indices_are_sorted = false,
  unique_indices = false
}} {as_mlir_sig((x.symval, w.symval, u.symval), y.symval)}
"""
