import slope
import slope.core
from slope.core import (
    Compiler,
    Backend,
    Operator,
    OperatorSet,
    ProcedureSet,
    Tensor,
    TensorBuffer,
    Typecheckor,
    PrimalProxy,
    list_zip,
    list_map,
    jit_op,
)

import math
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Sequence, Union, Iterator, NamedTuple
from collections import defaultdict
import iree.compiler
import iree.runtime
import os

from slope.operators import operator_set
from slope.procedures import procedure_set

sum_py = sum
max_py = max
abs_py = abs
slice_py = slice

compile_py = compile


def type_mlir(typecheckor):
    xdtype = typecheckor.dtype.short_name
    if len(typecheckor.shape) > 0:
        xshape = f"{'x'.join((repr(i) for i in typecheckor.shape))}"
        return f"tensor<{xshape}x{xdtype}>"
    else:
        return f"tensor<{xdtype}>"


def type_mlir_sig(in_avals, out_aval):
    typing_code = f" : ({','.join(type_mlir(t) for t in in_avals)}) -> {type_mlir(out_aval)}"
    return typing_code


class IREECompiler(Compiler):
    name = "iree"
    dtype_map = {
        Tensor.float32: np.dtypes.Float32DType(),
        Tensor.uint8: np.dtypes.UInt8DType(),
        Tensor.int8: np.dtypes.Int8DType(),
        Tensor.bool: np.dtypes.BoolDType(),
        Tensor.int32: np.dtypes.Int32DType(),
        Tensor.int64: np.dtypes.Float64DType(),
        Tensor.float16: np.dtypes.Float16DType(),
    }

    def from_numpy(self, val, dtype=Backend.DEFAULT_DTYPE, device=Backend.DEFAULT_DEVICE):
        # device_type, device_id = device.split(":") if ":" in device else (device, 0)
        np_val = np.array(val, dtype=dtype.numpy)
        iree_device = iree.runtime.get_device("local-task")
        val = iree.runtime.asdevicearray(iree_device, np_val)
        return Tensor(TensorBuffer(val))

    def numpy_of(self, tensor):
        return tensor.buf.val.to_host()

    def device_of(self, tensor):
        return tensor.buf.val._device

    def shape_of(self, tensor):
        return tuple(tensor.buf.val.shape)

    def dtype_of(self, tensor):
        return self.dtype_map_inv[tensor.buf.val.dtype]

    def export(self, jit_object, output_path, *args, **kwargs):
        raise NotImplementedError
        code = jit_object.code
        model = onnx.parser.parse_model(code)
        os.makedirs(output_path, exist_ok=True)
        in_binders = jit_object.codegen_out["in_binders"]
        outs = jit_object.codegen_out["outs"]
        num_consts = jit_object.program.num_consts
        for i in range(num_consts):
            const_array = in_binders[i]["type"].numpy()
            const_name = in_binders[i]["name"]
            const = onnx.numpy_helper.from_array(const_array, name=const_name)
            model.graph.initializer.append(const)
            # TODO: try if need these
            # const_tensor = next(t for t in model.graph.input if t.name == const_name)
            # const_tensor.type.tensor_type.shape.dim[0].dim_param = const_name
            # const_tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

        onnx.save(model.SerializeToString(), os.path.join(output_path, "model.onnx"))
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
        env: Dict[slope.Var, Any] = {}
        il1 = 4  # indent length
        body_code_lines = []

        for inb in program.in_binders:
            prefix = "%x" if type(inb.aval) is Typecheckor else "%c"
            idx = sum_py([1 if v["name"][0:2] == prefix else 0 for v in env.values()])
            env[inb] = dict(name=f"{prefix}{idx}", type=inb.aval)

        for instruction in program.instructions:
            if len(instruction.out_binders) == 0:  # skip codegen for function returns nothing
                continue
            in_vals = list_map(lambda x: env[x], instruction.inputs)
            for outb in instruction.out_binders:
                prefix = "%y" if outb in program.outs else "%z"
                idx = sum_py([1 if v["name"][0:2] == prefix else 0 for v in env.values()])
                env[outb] = dict(name=f"{prefix}{idx}", type=outb.aval)

            out_vals = list_map(lambda z: env[z], instruction.out_binders)
            if instruction.op.op_type is slope.core.OperatorType.Meta:
                impl_code, fn_defs = self.impls[instruction.op](args, instruction, in_vals, fn_defs)
            else:
                if instruction.op not in backend.compiler.impls.keys():
                    op = instruction.op
                    avals_in = tuple(inp.aval for inp in instruction.inputs)
                    params = instruction.params
                    op_program, consts, _ = slope.core.make_program(
                        getattr(backend.procedure_set, op.name),
                        *avals_in,
                        static_args=tuple(params.items()),
                        name=op.name,
                    )
                    name = op.get_jit_name(tuple(avals_in), params)
                    if name not in fn_defs.keys():
                        op_codegen_out = self.codegen(
                            op_program,
                            args,
                            fn_name=name,
                            fn_defs=fn_defs,
                        )
                        fn_defs = {**fn_defs, **op_codegen_out["fn_defs"]}
                        fn_defs[name] = op_codegen_out["code_lines"]
                    in_names = ", ".join(i["name"] for i in in_vals)
                    out_names = ", ".join(o["name"] for o in out_vals)
                    sig = type_mlir_sig(tuple(i["type"] for i in in_vals), out_vals[0]["type"])
                    impl_code = f"{out_names} = func.call @{name}({in_names}) {sig}"
                else:
                    impl_code = self.impls[instruction.op](*in_vals, *out_vals, **instruction.params)
            for impl_code_line in impl_code.split("\n"):  # handle multi-line code
                body_code_lines += [indent(impl_code_line, il1)]

        # inb_consts = [v for v in env.values() if "c" in v["name"]]
        # const_type_strs = [f"{self.dtype_map[c['type'].dtype]}[{repr(c['type'].shape)[1:-1]}] {c['name']}" for c in inb_consts]

        in_binders = list_map(lambda x: env[x], program.in_binders)
        fn_args_str = ", ".join([f"{i['name']}: {type_mlir(i['type'])}" for i in in_binders])

        outs = list_map(lambda x: env[x], program.outs)  # TODO: input that is output should has identity op
        out_str = ", ".join([f"{o['name']}" for o in outs])
        out_type_str = ", ".join([f"{type_mlir(o['type'])}" for o in outs])

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
        return dict(code_lines=code_lines, fn_defs=fn_defs, in_binders=in_binders, outs=outs)


compiler = IREECompiler()


@compiler.set_impl(operator_set.cast)
def cast_impl(self, x, y, *, dtype):
    return f'{y["name"]} = "stablehlo.convert"({x["name"]}) {type_mlir_sig((x["type"],), y["type"])}'


@compiler.set_impl(operator_set.stop_gradient)
def stop_gradient_impl(self, x, y):
    return f'{y["name"]} = "stablehlo.convert"({x["name"]}){type_mlir_sig((x["type"],), y["type"])}'


@compiler.set_impl(operator_set.sqrt)
def sqrt_impl(self, x, y):
    return f'{y["name"]} = "stablehlo.sqrt"({x["name"]}) {type_mlir_sig((x["type"],), y["type"])}'


@compiler.set_impl(operator_set.exp)
def exp_impl(self, x, y):
    return f'{y["name"]} = "stablehlo.exponential"({x["name"]}) {type_mlir_sig((x["type"],), y["type"])}'


@compiler.set_impl(operator_set.log)
def log_impl(self, x, y):
    return f'{y["name"]} = "stablehlo.log"({x["name"]}) {type_mlir_sig((x["type"],), y["type"])}'


@compiler.set_impl(operator_set.sin)
def sin_impl(self, x, y):
    return f'{y["name"]} = "stablehlo.sin"({x["name"]}) {type_mlir_sig((x["type"],), y["type"])}'


@compiler.set_impl(operator_set.invert)
def invert_impl(self, x, y):
    return f'{y["name"]} = "stablehlo.not"({x["name"]}) {type_mlir_sig((x["type"],), y["type"])}'


@compiler.set_impl(operator_set.add)
def add_impl(self, x, w, y):
    return f'{y["name"]} = "stablehlo.add"({x["name"]}, {w["name"]}) {type_mlir_sig((x["type"], w["type"]), y["type"])}'


@compiler.set_impl(operator_set.sub)
def sub_impl(self, x, w, y):
    return f'{y["name"]} = "stablehlo.subtract"({x["name"]}, {w["name"]}) {type_mlir_sig((x["type"], w["type"]), y["type"])}'


@compiler.set_impl(operator_set.mul)
def mul_impl(self, x, w, y):
    return f'{y["name"]} = "stablehlo.multiply"({x["name"]}, {w["name"]}) {type_mlir_sig((x["type"], w["type"]), y["type"])}'


@compiler.set_impl(operator_set.div)
def div_impl(self, x, w, y):
    return (
        f'{y["name"]} = "stablehlo.divide"({x["name"]}, {w["name"]}) {type_mlir_sig((x["type"], w["type"]), y["type"])}'
    )


@compiler.set_impl(operator_set.pow)
def pow_impl(self, x, w, y):
    return (
        f'{y["name"]} = "stablehlo.power"({x["name"]}, {w["name"]}) {type_mlir_sig((x["type"], w["type"]), y["type"])}'
    )


@compiler.set_impl(operator_set.equal)
def equal_impl(self, x, w, y):
    return f"""{y["name"]} = "stablehlo.compare"({x["name"]}, {w["name"]}) {{
  comparison_direction = #stablehlo<comparison_direction EQ>,
  compare_type = #stablehlo<comparison_type FLOAT>
}}  {type_mlir_sig((x["type"], w["type"]), y["type"])}
"""


@compiler.set_impl(operator_set.maximum)
def maximum_impl(self, x, w, y):
    return f'{y["name"]} = "stablehlo.maximum"({x["name"]}, {w["name"]}) {type_mlir_sig((x["type"], w["type"]), y["type"])}'


@compiler.set_impl(operator_set.matmul)
def matmul_impl(self, x, w, y):
    return f'{y["name"]} = "stablehlo.dot"({x["name"]}, {w["name"]}) {type_mlir_sig((x["type"], w["type"]), y["type"])}'


@compiler.set_impl(operator_set.sum)
def sum_impl(self, x, y, *, dim, keepdim):
    zero = "0." if "f" in y["type"].dtype.short_name else "0"
    y_init_type = Typecheckor((), y["type"].dtype)
    y_mlir_type = type_mlir(y_init_type)
    y_out_type = (
        y["type"]
        if not keepdim
        else Typecheckor(tuple(d for i, d in enumerate(y["type"].shape) if i not in dim), y["type"].dtype)
    )
    return f"""
{y["name"]}_init = stablehlo.constant dense<{zero}> : {type_mlir(y_init_type)}
{y["name"]}{'_' if keepdim else ''} = "stablehlo.reduce"({x["name"]}, {y["name"]}_init) ({{
  ^bb0(%arg0: {y_mlir_type}, %arg1: {y_mlir_type}):
    %0 = "stablehlo.add"(%arg0, %arg1) {type_mlir_sig((y_init_type, y_init_type), y_init_type)}
    "stablehlo.return"(%0) : ({y_mlir_type}) -> ()
}}) {{
  dimensions = dense<{repr(list(dim))}> : tensor<{len(dim)}xi64>
}} {type_mlir_sig((x["type"], y_init_type), y_out_type)}
{f'{y["name"]} = "stablehlo.reshape"({y["name"]}_) {type_mlir_sig((y_out_type,), y["type"])}' if keepdim else ''}"""


@compiler.set_impl(operator_set.max)
def max_impl(self, x, y, *, dim, keepdim):
    min_val = {Tensor.float32: "1.E-38", Tensor.int8: "-128", Tensor.int32: "-65536"}[x["type"].dtype]
    y_init_type = Typecheckor((), y["type"].dtype)
    y_mlir_type = type_mlir(y_init_type)
    y_out_type = (
        y["type"]
        if not keepdim
        else Typecheckor(tuple(d for i, d in enumerate(y["type"].shape) if i not in dim), y["type"].dtype)
    )
    return f"""
{y["name"]}_init = stablehlo.constant dense<{min_val}> : {type_mlir(y_init_type)}
{y["name"]}{'_' if keepdim else ''} = "stablehlo.reduce"({x["name"]}, {y["name"]}_init) ({{
  ^bb0(%arg0: {y_mlir_type}, %arg1: {y_mlir_type}):
    %0 = "stablehlo.maximum"(%arg0, %arg1) {type_mlir_sig((y_init_type, y_init_type), y_init_type)}
    "stablehlo.return"(%0) : ({y_mlir_type}) -> ()
}}) {{
  dimensions = dense<{repr(list(dim))}> : tensor<{len(dim)}xi64>
}} {type_mlir_sig((x["type"], y_init_type), y_out_type)}
{f'{y["name"]} = "stablehlo.reshape"({y["name"]}_) {type_mlir_sig((y_out_type,), y["type"])}' if keepdim else ''}
"""


@compiler.set_impl(operator_set.arange)
def arange_impl(self, y, *, start, stop, stride, dtype):
    return f'{y["name"]} = "stablehlo.iota"() {{iota_dimension = 0 : i64}} {type_mlir_sig((), y["type"])}'


@compiler.set_impl(operator_set.full)
def full_impl(self, y, *, shape, fill_value, dtype):
    fill_value = float(fill_value) if "f" in dtype.short_name else int(fill_value)
    fill_value = repr(fill_value)
    fill_value = fill_value.replace("e", "E") if "." in fill_value else fill_value.replace("e", ".E")
    return f'{y["name"]} = "stablehlo.constant"() {{ value = dense<{fill_value}> : {type_mlir(y["type"])} }} {type_mlir_sig((), y["type"])}'


@compiler.set_impl(operator_set.random_uniform)
def random_uniform_impl(self, y, *, shape, dtype):
    zero = "0." if "f" in y["type"].dtype.short_name else "0"
    one = "1." if "f" in y["type"].dtype.short_name else "1"
    a_type = b_type = Typecheckor((), dtype)
    is_scalar = shape == ()
    shape_val = f'dense<{repr(list(shape)) if not is_scalar else "[1]"}'
    shape_type = Typecheckor((1,) if is_scalar else (len(shape),), Tensor.int64)
    y_out_type = y["type"] if not is_scalar else Typecheckor((1,), y["type"].dtype)
    return f"""{y["name"]}_a = stablehlo.constant dense<{zero}> : {type_mlir(a_type)}
{y["name"]}_b = stablehlo.constant dense<{one}> : {type_mlir(b_type)}
{y["name"]}_shape = stablehlo.constant {shape_val}> : {type_mlir(shape_type)}
{y["name"]}{'_' if is_scalar else ''} = "stablehlo.rng"({y["name"]}_a, {y["name"]}_b,{y["name"]}_shape) {{
        rng_distribution = #stablehlo<rng_distribution UNIFORM>}} {type_mlir_sig((a_type, b_type, shape_type), y_out_type)}
{f'{y["name"]} = "stablehlo.reshape"({y["name"]}_) {type_mlir_sig((y_out_type,), y["type"])}' if is_scalar else ''}"""


@compiler.set_impl(operator_set.random_normal)
def random_normal_impl(self, y, *, shape, dtype):
    zero = "0." if "f" in y["type"].dtype.short_name else "0"
    one = "1." if "f" in y["type"].dtype.short_name else "1"
    a_type = b_type = Typecheckor((), dtype)
    is_scalar = shape == ()
    shape_val = f'dense<{repr(list(shape)) if not is_scalar else "[1]"}'
    shape_type = Typecheckor((1,) if is_scalar else (len(shape),), Tensor.int64)
    y_out_type = y["type"] if not is_scalar else Typecheckor((1,), y["type"].dtype)
    return f"""{y["name"]}_a = stablehlo.constant dense<{zero}> : {type_mlir(a_type)}
{y["name"]}_b = stablehlo.constant dense<{one}> : {type_mlir(b_type)}
{y["name"]}_shape = stablehlo.constant {shape_val}> : {type_mlir(shape_type)}
{y["name"]}{'_' if is_scalar else ''} = "stablehlo.rng"({y["name"]}_a, {y["name"]}_b,{y["name"]}_shape) {{
        rng_distribution = #stablehlo<rng_distribution NORMAL>}} {type_mlir_sig((a_type, b_type, shape_type), y_out_type)}
{f'{y["name"]} = "stablehlo.reshape"({y["name"]}_) {type_mlir_sig((y_out_type,), y["type"])}' if is_scalar else ''}"""


@compiler.set_impl(operator_set.expand)
def expand_impl(self, x, y, *, shape):
    return f"""{y["name"]} = "stablehlo.broadcast_in_dim"({x["name"]}) {{
        broadcast_dimensions = dense<{repr(list(range(len(shape))))}>: tensor<{len(shape)}xi64>
        }} {type_mlir_sig(( x["type"],), y["type"])}
"""


@compiler.set_impl(operator_set.reshape)
def reshape_impl(self, x, y, *, shape):
    return f'{y["name"]} = "stablehlo.reshape"({x["name"]}) {type_mlir_sig((x["type"],), y["type"])}'


@compiler.set_impl(operator_set.pad)
def pad_impl(self, x, y, *, padding, mode, value):
    value = float(value) if "f" in x["type"].dtype.short_name else int(value)
    value_type = Typecheckor((), x["type"].dtype)
    lo = padding[0::2][::-1]
    hi = padding[1::2][::-1]
    return f"""{y["name"]}_value = stablehlo.constant dense<{value}> : {type_mlir(value_type)}
{y["name"]} = "stablehlo.pad"({x["name"]}, {y["name"]}_value) {{
  edge_padding_low = dense<{repr(list(lo))}> : tensor<{len(lo)}xi64>,
  edge_padding_high = dense<{repr(list(hi))}> : tensor<{len(hi)}xi64>,
  interior_padding = dense<{repr([0]*len(lo))}> : tensor<{len(lo)}xi64>
}} {type_mlir_sig((x["type"], value_type), y["type"])}
"""


@compiler.set_impl(operator_set.slice)
def slice_impl(self, x, y, *, starts, limits, strides):
    return f"""{y["name"]} = "stablehlo.slice"({x["name"]}) {{
  start_indices = dense<{repr(list(starts))}> : tensor<{len(starts)}xi64>,
  limit_indices = dense<{repr(list(limits))}> : tensor<{len(limits)}xi64>,
  strides = dense<{repr(list(strides))}> : tensor<{len(strides)}xi64>
}} {type_mlir_sig((x["type"],), y["type"])}
"""


@compiler.set_impl(operator_set.cat)
def cat_impl(self, *xs, dim):
    xs, y = xs[:-1], xs[-1]
    return f"""{y["name"]} = "stablehlo.concatenate"({','.join([x["name"] for x in xs])}) {{
 dimension = {dim} : i64
}} {type_mlir_sig(([x["type"] for x in xs]), y["type"])}"""


@compiler.set_impl(operator_set.permute)
def permute_impl(self, x, y, *, perm):
    return f"""{y["name"]} = "stablehlo.transpose"({x["name"]}) {{
  permutation = dense<{repr(list(perm))}> : tensor<{len(perm)}xi64>
}} {type_mlir_sig((x["type"],), y["type"])}"""


@compiler.set_impl(operator_set.flip)
def flip_impl(self, x, y, *, dim):
    return f"""{y["name"]} = "stablehlo.reverse"({x["name"]}) {{
  dimensions = dense<{repr(list(dim))}> : tensor<{len(dim)}xi64>
}}  {type_mlir_sig((x["type"],), y["type"])}
"""


@compiler.set_impl(operator_set.conv)
def conv_impl(self, x, w, y, *, groups, stride, dilation, padding):
    padding = [[s, e] for s, e in zip(list(padding[0::2]), list(padding[1::2]))]
    HW = len(x["type"].shape[2:])
    # return f"""ret = Conv<{dilations_attr}, {pads_attr}, {strides_attr}, {group_attr}>({x}, {w})"""
    return f"""{y["name"]} = "stablehlo.convolution"({x["name"]}, {w["name"]}) {{
  window_strides = dense<{list(stride)}> : tensor<{len(stride)}xi64>,
  padding = dense<{padding}> : tensor<{HW}x{HW}xi64>,
  lhs_dilation = dense<1> : tensor<{HW}xi64>,
  rhs_dilation = dense<{list(dilation)}> : tensor<{HW}xi64>,
  window_reversal = dense<false> : tensor<{HW}xi1>,
  dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>,
  feature_group_count = {groups} : i64,
  batch_group_count = 1 : i64,
  precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
}}  {type_mlir_sig((x["type"], w["type"]), y["type"])}
"""


@compiler.set_impl(jit_op)
def jit_op_impl(self, args, instruction, fn_defs, in_vals, out_vals):
    jit_program = instruction.params["program"]
    jit_name = f"{jit_program.name}"
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
    ret = f"{out_vals} = slope.{jit_name}({args_str})"
    return ret, fn_defs


backend = Backend(operator_set, procedure_set, compiler)
