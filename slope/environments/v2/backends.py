import slope
from slope.environments.v1.operators import operator_set
from slope.core import Backend, Tensor, Typecheckor, list_zip, list_map
import numpy as np
from typing import (
    List,
    Dict,
    Any,
)
import onnxruntime
import inspect
from functools import partial
import onnx
from onnx import helper, numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto

onnxruntime_backend = Backend(name="numpy", default_dtype=Tensor.float32, deps=("numpy as np", "math"))
onnxruntime_dtype_map = {
    Tensor.float32: TensorProto.FLOAT,
    Tensor.float16: TensorProto.FLOAT16,
    Tensor.int8: TensorProto.INT8,
    Tensor.uint8: TensorProto.UINT8,
    Tensor.int32: TensorProto.INT32,
    Tensor.int64: TensorProto.INT64,
    Tensor.bool: TensorProto.BOOL,
}
onnxruntime_backend.set_dtype_map(onnxruntime_dtype_map)

default_dtype_backend = onnxruntime_backend.default_dtype_value


@onnxruntime_backend.set_method
def numpy_of(self, tensor):
    return tensor.buf.val


@onnxruntime_backend.set_method
def set_device_of(self, tensor):
    return "cpu"


def create_session(model: str) -> onnxruntime.InferenceSession:
    providers = ["CPUExecutionProvider"]
    providers.insert(0, "CUDAExecutionProvider")
    return onnxruntime.InferenceSession(model, providers=providers)


# Run the model on CPU consuming and producing numpy arrays
def run(x: np.array, y: np.array) -> np.array:
    session = create_session(MODEL_FILE)

    z = session.run(["z"], {"x": x, "y": y})

    return z[0]


# Run the model on device consuming and producing ORTValues
def run_with_data_on_device(x: np.array, y: np.array) -> onnxruntime.OrtValue:
    session = create_session(MODEL_FILE)

    x_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(x, DEVICE_NAME, DEVICE_INDEX)
    y_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(y, DEVICE_NAME, DEVICE_INDEX)

    io_binding = session.io_binding()
    io_binding.bind_input(
        name="x",
        device_type=x_ortvalue.device_name(),
        device_id=0,
        element_type=x.dtype,
        shape=x_ortvalue.shape(),
        buffer_ptr=x_ortvalue.data_ptr(),
    )
    io_binding.bind_input(
        name="y",
        device_type=y_ortvalue.device_name(),
        device_id=0,
        element_type=y.dtype,
        shape=y_ortvalue.shape(),
        buffer_ptr=y_ortvalue.data_ptr(),
    )
    io_binding.bind_output(
        name="z",
        device_type=DEVICE_NAME,
        device_id=DEVICE_INDEX,
        element_type=x.dtype,
        shape=x_ortvalue.shape(),
    )
    session.run_with_iobinding(io_binding)

    z = io_binding.get_outputs()

    return z[0]


@onnxruntime_backend.set_method
def compile(self, codegen_out):
    code_lines = codegen_out["code_lines"]
    exec_locals = {}
    code = "\n".join(code_lines)
    # X is numpy array on cpu

    model = onnx.parser.parse_model(code_lines)
    # text = onnx.printer.to_text(model)
    onnx.checker.check_model(model)

    def run(*args):
        # X is numpy array on cpu
        X_ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X, "cuda", 0)
        # Y_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type([3, 2], np.float32, 'cuda', 0)
        session = onnxruntime.InferenceSession(
            "model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        io_binding = session.io_binding()
        io_binding.bind_input(
            name="input",
            device_type=X_ortvalue.device_name(),
            device_id=0,
            element_type=np.float32,
            shape=X_ortvalue.shape(),
            buffer_ptr=X_ortvalue.data_ptr(),
        )
        # Request ONNX Runtime to bind and allocate memory on CUDA for 'output'
        io_binding.bind_output("output", "cuda")
        # io_binding.bind_ortvalue_input('input', X_ortvalue)
        # io_binding.bind_ortvalue_output('output', Y_ortvalue)
        session.run_with_iobinding(io_binding)
        # The following call returns an OrtValue which has data allocated by ONNX Runtime on CUDA
        ort_output = io_binding.get_outputs()[0]
        return ort_output

    return fn, code


@onnxruntime_backend.set_method
def codegen(self, program, args, *, fn_name: str = "main", fn_defs=dict()) -> List[Any]:
    if fn_name == "main":
        assert not hasattr(self, "fn_count")
        self.fn_count = 0
    slope.dblog(f"\n-- Codegen program {program.name} as {fn_name}\n", program, "\n ==", level=1)

    def indent(code_line, amount):
        spaces = " " * (len(code_line) - len(code_line.lstrip()))
        spaces += " " * amount
        return "\n".join([spaces + line for line in code_line.strip().split("\n")])

    # codegen is recursive if jit-of-jit happens
    environment: Dict[slope.Var, Any] = {}
    il1 = 4
    ncs = 0
    nxs = 0
    nzs = 0
    inb_args = []
    inb_consts = []

    input = """
<
    ir_version: 7,
    opset_import: ["" : 10]
>
name (float[N, 128] X, float[128, 10] W, float[10] B) => (float[N, 10] C)
{
    T = MatMul(X, W)
    S = Add(T, B)
    C = Softmax(S)
}
"""
    model = onnx.parser.parse_model(input)

    for inb in program.in_binders:
        if type(inb.aval) is not Typecheckor:
            environment[inb] = f"c{ncs}"
            inb_consts += [environment[inb]]
            ncs += 1
        else:
            environment[inb] = f"x{nxs}"
            inb_args += [environment[inb]]
            nxs += 1

    code_lines = []
    fn_args_strs = f""
    if inb_consts:
        fn_args_strs += f"{', '.join(inb_consts)}, "
    fn_args_strs += f"{', '.join(inb_args)}"
    code_lines += [f"def {fn_name}({fn_args_strs}):"]
    for instruction in program.instructions:
        in_vals = list_map(lambda x: environment[x], instruction.inputs)
        in_avals = [x.aval for x in instruction.inputs]
        for outb in instruction.out_binders:
            environment[outb] = f"z{nzs}"
            nzs += 1
        out_vals = list_map(lambda z: environment[z], instruction.out_binders)

        if instruction.op.op_type is slope.core.OperatorType.Meta:
            if instruction.op is slope.core.procedure_op:
                proc_program = instruction.params["program"]
                self.fn_count += 1
                proc_name = f"{proc_program.name}_{self.fn_count}"
                proc_codegen_out = self.codegen(
                    proc_program,
                    args,
                    fn_name=proc_name,
                    fn_defs=fn_defs,
                )
                fn_defs[proc_name] = proc_codegen_out["code_lines"]
                fn_defs = {**fn_defs, **proc_codegen_out["fn_defs"]}
                # if proc_key not in fn_defs.keys():
                #     proc_name = f"{proc_program.name}_{len(fn_defs)}"
                #     proc_codegen_out = self.codegen(
                #         proc_program,
                #         args,
                #         fn_name=proc_name,
                #         fn_defs=fn_defs,
                #     )
                #     fn_defs = {**fn_defs, **proc_codegen_out["fn_defs"]}
                #     fn_defs[proc_key] = proc_codegen_out["code_lines"]
                # else:
                #     proc_code_lines = fn_defs[proc_key]
                #     proc_name = proc_code_lines[0].split()[1].split("(")[0]

                # if proc_name == "zeros_partial2_T_5":
                #     breakpoint()

                args_str = ", ".join(in_vals)
                lhs = f"{out_vals[0]+',' if len(out_vals) == 1 else ', '.join([o for o in out_vals])}"
                rhs = f"{proc_name}({args_str})"
                if len(lhs) == 0:  # return None functions
                    continue

            elif instruction.op is slope.core.jit_op:
                jit_program = instruction.params["program"]
                jit_name = f"{program.name}_{len(fn_defs)}"
                jit_codegen_out = self.codegen(
                    jit_program,
                    args,
                    fn_name=jit_name,
                    fn_defs=fn_defs,
                )
                fn_defs = {**fn_defs, **jit_codegen_out["fn_defs"]}
                fn_defs[jit_name] = jit_codegen_out["code_lines"]

                args_str = ", ".join(in_vals)
                lhs = f"{out_vals[0]+',' if len(out_vals) == 1 else ', '.join([o for o in out_vals])}"
                rhs = f"{jit_name}({args_str})"
            else:
                raise
        else:
            impl = slope.M().backend.impls[instruction.op]
            args_str = ", ".join(in_vals)
            lhs = f"{out_vals[0] if len(out_vals) == 1 else ', '.join([o for o in out_vals])}"

            impl_lines = inspect.getsourcelines(impl)[0]
            if impl_lines[0][0] == "@":  # delete decorator line
                impl_lines = impl_lines[1:]

            if len(impl_lines) > 2:
                params = {
                    k: v if not isinstance(v, slope.core.DType) else self.dtype_map[v]
                    for k, v in instruction.params.items()
                }
                kwargs_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                rhs = f"{instruction.op.name}({args_str}, {kwargs_str})"

                if instruction.op.name not in fn_defs.keys():
                    def_str = impl_lines[0]
                    impl_lines[0] = f"def {instruction.op.name}{def_str[def_str.find('('):]}"
                    impl_lines[0] = impl_lines[0].replace("self, ", "")
                    fn_defs[instruction.op.name] = impl_lines
            else:
                sig = inspect.signature(impl)
                args_strs = [
                    k
                    for k, v in sig.parameters.items()
                    if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and k != "self"
                ]
                rhs = impl_lines[1].replace("return", "").strip()

                for argname, arg in list_zip(args_strs, in_vals):
                    mark = "," if argname != args_strs[-1] or len(instruction.params) > 0 else ")"
                    rhs = rhs.replace(f"{argname}{mark}", f"{arg}{mark}")
                for kwargname, kwarg in instruction.params.items():
                    if isinstance(kwarg, slope.core.DType):
                        kwarg = self.dtype_map[kwarg]
                    rhs = rhs.replace(f"={kwargname}", f"={kwarg}")
        code_line = f"{lhs} = {rhs}"
        code_lines += [indent(code_line, il1)]
    outs = list_map(lambda x: environment[x], program.outs)
    ret_str = f"{', '.join(outs)}{',' if len(outs)==1 else ''}"
    code_lines += [indent(f"return {ret_str}", il1)]
    # if fn_name == "cos_jvp_partial2_T_jvp_partial2_T_5":
    #     breakpoint()
    if fn_name == "main":
        if len(fn_defs) > 0:
            code_lines = (
                code_lines[0:1]
                + [indent(line, il1) for impl_lines in fn_defs.values() for line in impl_lines]
                + code_lines[1:]
            )
        code_lines = code_lines[0:1] + [indent(f"float32 = np.float32", il1)] + code_lines[1:]

    slope.dblog("\n-- Code:\n\n" + "\n".join(code_lines) + "\n\n==\n", level=1)
    if fn_name == "main":
        del self.fn_count

    return dict(code_lines=code_lines, fn_defs=fn_defs)


### Operator Impls


@onnxruntime_backend.set_impl(operator_set.convert)
def f(self, x, *, dtype):
    ret = x
    return ret.astype(dtype=dtype)


@onnxruntime_backend.set_impl(operator_set.stop_gradient)
def f(self, x, *, dtype):
    return x


@onnxruntime_backend.set_impl(operator_set.neg)
def f(self, x):
    return np.negative(x)


@onnxruntime_backend.set_impl(operator_set.sqrt)
def f(self, x):
    return np.sqrt(x)


@onnxruntime_backend.set_impl(operator_set.exp)
def f(self, x):
    return np.exp(x)


@onnxruntime_backend.set_impl(operator_set.log)
def f(self, x):
    return np.log(x)


@onnxruntime_backend.set_impl(operator_set.sin)
def f(self, x):
    return np.sin(x)


@onnxruntime_backend.set_impl(operator_set.add)
def f(self, x, y):
    return np.add(x, y)


@onnxruntime_backend.set_impl(operator_set.sub)
def f(self, x, y):
    return np.subtract(x, y)


@onnxruntime_backend.set_impl(operator_set.mul)
def f(self, x, y):
    return np.multiply(x, y)


@onnxruntime_backend.set_impl(operator_set.div)
def f(self, x, y):
    return np.divide(x, y)


@onnxruntime_backend.set_impl(operator_set.equal)
def f(self, x, y):
    return np.equal(x, y)


@onnxruntime_backend.set_impl(operator_set.not_equal)
def f(self, x, y):
    return np.not_equal(x, y)


@onnxruntime_backend.set_impl(operator_set.maximum)
def f(self, x, y):
    return np.maximum(x, y)


@onnxruntime_backend.set_impl(operator_set.sum)
def f(self, x, *, axes=None, keepdims=False):
    return np.sum(x, axis=axes, keepdims=keepdims)


@onnxruntime_backend.set_impl(operator_set.max)
def f(self, x, *, axes=None, keepdims=False):
    return np.max(x, axis=axes, keepdims=keepdims)


@onnxruntime_backend.set_impl(operator_set.constant)
def f(self, val, *, dtype=default_dtype_backend):
    return np.array(val, dtype=dtype)


@onnxruntime_backend.set_impl(operator_set.arange)
def f(self, *, start, stop, stride, dtype=default_dtype_backend):
    return np.arange(start=start, stop=stop, stride=stride, dtype=dtype)


@onnxruntime_backend.set_impl(operator_set.full)
def f(self, *, shape, fill_value, dtype=default_dtype_backend):
    return np.full(shape=shape, fill_value=fill_value, dtype=dtype)


@onnxruntime_backend.set_impl(operator_set.random_uniform)
def f(self, *, shape, dtype=default_dtype_backend):
    return np.random.uniform(size=shape).astype(dtype=dtype)


@onnxruntime_backend.set_impl(operator_set.random_normal)
def f(self, *, shape, dtype=default_dtype_backend):
    return np.random.normal(loc=np.zeros(shape=shape)).astype(dtype=dtype)


@onnxruntime_backend.set_impl(operator_set.broadcast_in_dim)
def f(self, x, *, shape, axes=()):
    ret = x
    for a in sorted(axes):
        ret = np.expand_dims(ret, a)
    ret = np.broadcast_to(ret, shape)
    return ret


@onnxruntime_backend.set_impl(operator_set.reshape)
def f(self, x, *, shape):
    return np.reshape(x, newshape=shape)


@onnxruntime_backend.set_impl(operator_set.pad_hlo)
def f(self, x, *, lo, hi, interior, value):
    # TODO: implement interior pad
    return np.pad(x, list(zip(lo, hi)), constant_values=value)


@onnxruntime_backend.set_impl(operator_set.slice_hlo)
def f(self, x, *, starts, limits, strides):
    slices = tuple(slice(s, l, st) for s, l, st in zip(starts, limits, strides))
    return x[slices]


@onnxruntime_backend.set_impl(operator_set.concatenate)
def f(self, *xs, axis):
    assert len(xs) > 1
    return np.concatenate(xs, axis)


@onnxruntime_backend.set_impl(operator_set.transpose)
def f(self, x, *, perm):  # NOTE: np.transpose is like torch.permute
    return np.transpose(x, axes=perm)


@onnxruntime_backend.set_impl(operator_set.flip)
def f(self, x, *, axes):
    return np.flip(x, axis=axes)


# def inline():
# if instruction.op.op_dtype is slope.core.OperatorType.Meta:
#     # TODO: generalize interface to other than jit_op
#     raise NotImplementedError
#     breakpoint()
#     op_out = impl(in_vals, in_avals, params=instruction.params)
#     co = op_out["codegen_out"]
#     outs = co["outs"]
#     rhs = f"{outs[0] if len(outs) == 1 else ', '.join([o for o in outs])}"
#     op_code_lines = co["code_lines"]
#     input_lhs = ", ".join((co["inb_args"] + co["inb_consts"]))
#     input_code_line = f"{input_lhs} = {args_str}"
#     output_code_line = f"{lhs} = {rhs}"
#     code_lines += [input_code_line] + op_code_lines + [output_code_line]
#     continue
