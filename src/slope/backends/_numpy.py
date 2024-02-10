import slope
from slope.core import (
    Compiler,
    Backend,
    Operator,
    OperatorSet,
    ProcedureSet,
    Tensor,
    TensorBuffer,
    SymbolicTensor,
    UndefPrimal,
    list_zip,
    list_map,
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
    Callable,
)
from collections import defaultdict
import importlib
import os
import functools

# --------------
# Compiler
# --------------

compiler = Compiler(name="numpy", default_dtype=slope.SLOPE_DTYPE)
compiler.set_dtype_map(
    {
        slope.core.dtypes.float32: np.dtype("float32"),
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
    return tuple(int(i) for i in tensor.buf.val.shape)


@compiler.set_method
def dtype_of(self, tensor):
    return self.dtype_map_inv[tensor.buf.val.dtype]


@compiler.set_method
def export(self, jit_output: slope.core.JitOutput, output_path, *args, **kwargs):
    code = jit_output.code
    os.makedirs(output_path, exist_ok=True)
    consts_dir_path = os.path.join(output_path, "consts")
    os.makedirs(consts_dir_path, exist_ok=True)
    in_binders = jit_output.codegen_output["in_binders"]
    outs = jit_output.codegen_output["outs"]
    num_consts = jit_output.program.num_consts
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
def compile(self, codegen_output):
    deps_dict = dict()
    deps_dict["numpy"] = importlib.import_module("numpy")
    deps_dict["np"] = deps_dict["numpy"]
    deps_dict["math"] = importlib.import_module("math")
    code_lines = codegen_output["code_lines"]
    exec_locals = dict()
    code = "\n".join(code_lines)
    exec(compile(code, "<string>", "exec"), deps_dict, exec_locals)
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
        prefix = "x" if type(inb.symtensor) is SymbolicTensor else "c"
        idx = sum_py([1 if v["name"][0] == prefix else 0 for v in backend.values()])
        backend[inb] = dict(name=f"{prefix}{idx}", type=inb.symtensor)

    for instruction in program.instructions:
        if len(instruction.out_binders) == 0:  # skip codegen for function returns nothing
            continue
        in_vals = list_map(lambda x: backend[x]["name"], instruction.inputs)
        for outb in instruction.out_binders:
            prefix = "y" if outb in program.outs else "z"
            idx = sum_py([1 if v["name"][0] == prefix else 0 for v in backend.values()])
            backend[outb] = dict(name=f"{prefix}{idx}", type=outb.symtensor)

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
                np_dtype.name,
                "bool" if np_dtype is np.dtype("bool") else f"np.{np_dtype.name}",
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
    slope.dblog(
        f"\n-- {program.name} codegen:\n\n" + "\n".join(code_lines) + "\n\n==\n",
        enable=slope.LOG_JIT,
    )

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
compiler.set_impl(operator_set.sum)(lambda self, x, *, dim, keepdim: f"ret = np.sum({x}, axis={dim}, keepdims={keepdim})")
compiler.set_impl(operator_set.max)(lambda self, x, *, dim, keepdim: f"ret = np.max({x}, axis={dim}, keepdims={keepdim})")
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

compiler.set_impl(operator_set.cat)(lambda self, *xs, dim: f"ret = np.concatenate(({','.join(xs)}), axis={dim})")
compiler.set_impl(operator_set.permute)(lambda self, x, *, perm: f"ret = np.transpose({x}, axes={perm})")
compiler.set_impl(operator_set.flip)(lambda self, x, *, dim: f"ret = np.flip({x}, axis={dim})")


@compiler.set_impl(slope.core.jit_op)
def jit_op_impl(self, program, args, instruction, in_vals, fn_defs):
    jit_program = instruction.params["program"]
    jit_name = f"{program.name}"
    jit_codegen_output = self.codegen(
        jit_program,
        args,
        fn_name=jit_name,
        fn_defs=fn_defs,
    )
    assert jit_name not in fn_defs.keys()
    fn_defs[jit_name] = jit_codegen_output["code_lines"]
    fn_defs = {**fn_defs, **jit_codegen_output["fn_defs"]}
    args_str = ", ".join(in_vals)
    rhs = f"{jit_name}({args_str})"
    return rhs, fn_defs


procedure_set = ProcedureSet()


@procedure_set.register()
def zeros(*args, **kwargs):
    dtype = kwargs.get("dtype", slope.SLOPE_DTYPE)
    if kwargs.get("shape", None) is None:
        shape = args[0] if isinstance(args[0], (tuple, list)) else args
        assert all(i >= 0 for i in shape)
    return slope.full(shape, 0.0, dtype)


@procedure_set.register()
def ones(*args, **kwargs):
    dtype = kwargs.get("dtype", slope.SLOPE_DTYPE)
    if kwargs.get("shape", None) is None:
        shape = args[0] if isinstance(args[0], (tuple, list)) else args
        assert all(i >= 0 for i in shape)
    return slope.full(shape=shape, fill_value=1.0, dtype=dtype)


@procedure_set.register(static_argnames="fill_value")
def full_like(y, fill_value):
    return slope.full(shape=y.shape, fill_value=fill_value, dtype=y.dtype)


@procedure_set.register()
def zeros_like(y):
    return full_like(y, 0.0)


@procedure_set.register()
def ones_like(y):
    return full_like(y, 1.0)


@procedure_set.register()
def where(x, trueval, falseval):
    cond = x != 0.0
    if not isinstance(trueval, Tensor):
        trueval = slope.full((), trueval)
    if not isinstance(falseval, Tensor):
        falseval = slope.full((), falseval)
    cond = cond.cast(trueval.dtype)
    return cond * trueval + (1.0 - cond) * falseval


@procedure_set.register(static_argnames="dim keepdim")
def mean(x, dim=None, keepdim=False):
    out = x.sum(dim=dim, keepdim=keepdim)
    return out * (math.prod(out.shape) / math.prod(x.shape))


@procedure_set.register()
def rsqrt(x):
    return (slope.ones_like(x) / x).sqrt()


@procedure_set.register()
def cos(x):
    return ((math.pi / 2) - x).sin()


@procedure_set.register()
def tan(x):
    return x.sin() / x.cos()


@procedure_set.register()
def not_equal(x, w):
    return ~(x.equal(w))


@procedure_set.register()
def greater_equal(x, w):
    return x.maximum(w).equal(w)


@procedure_set.register()
def less_equal(x, w):
    return x.minimum(w).equal(w)


@procedure_set.register()
def greater(x, w):
    return 1.0 - (x <= w)


@procedure_set.register()
def less(x, w):
    return 1.0 - (x >= w)


@procedure_set.register()
def minimum(x, w):
    return -x.maximum(-x, -w)


@procedure_set.register(static_argnames="dim keepdim")
def min(x, dim=None, keepdim=False):
    return -((-x).max(x, dim, keepdim))


@procedure_set.register(static_argnames="dim keepdim")
def argmax(x, dim=None, keepdim=False):
    if dim is None:
        idx = (x == x.max(dim)) * slope.arange(
            math.prod(x.shape) - 1,
            -1,
            -1,
            dtype=slope.int32,
        ).reshape(x.shape)
        return math.prod(x.shape) - idx.max() - 1
    dim = dim + len(x.shape) if dim < 0 else dim
    m = (x == x.max(dim=dim, keepdim=True)).cast(slope.int32)
    idx = m * slope.arange(x.shape[dim] - 1, -1, -1, dtype=slope.int32).reshape((x.shape[dim], *[1] * (x.ndim - dim - 1)))
    ret = x.shape[dim] - idx.max(dim=dim, keepdim=keepdim) - 1
    return ret


@procedure_set.register(static_argnames="dim keepdim")
def argmin(x, dim=None, keepdim=False):
    return (-x).argmax(dim=dim, keepdim=keepdim)


@procedure_set.register()
def log2(x):
    return x.log() / math.log(2)


@procedure_set.register()
@staticmethod
def _tri(r: int, c: int, k: int = 0, **kwargs) -> Tensor:
    return slope.arange(r, **kwargs).unsqueeze(1).expand(r, c) <= Tensor.arange(-k, c - k, **kwargs).unsqueeze(0).expand(r, c)


@procedure_set.register()
def triu(self, k: int = 0) -> Tensor:
    return slope._tri(
        self.shape[-2],
        self.shape[-1],
        k=k,
        dtype=self.dtype,
        device=self.device,
    ).where(self, slope.zeros_like(self))


@procedure_set.register()
def tril(self, k: int = 0) -> Tensor:
    return slope._tri(
        self.shape[-2],
        self.shape[-1],
        k=k + 1,
        dtype=self.dtype,
        device=self.device,
    ).where(slope.zeros_like(self), self)


@procedure_set.register()
def trunc(self: Tensor) -> Tensor:
    return self.cast(slope.int32).cast(self.dtype)


@procedure_set.register()
def ceil(self: Tensor) -> Tensor:
    return (self > (b := self.trunc())).where(b + 1, b)


@procedure_set.register()
def floor(self: Tensor) -> Tensor:
    return (self < (b := self.trunc())).where(b - 1, b)


@procedure_set.register()
def square(self):
    return self * self


@procedure_set.register()
def clip(self, min_, max_):
    return self.maximum(min_).minimum(max_)


@procedure_set.register()
def abs(self):
    return self.relu() + (-self).relu()


@procedure_set.register()
def sign(self):
    return self / (self.abs() + 1e-10)


@procedure_set.register()
def reciprocal(self):
    return 1.0 / self


@procedure_set.register()
def matmul(x, w):
    x = x.reshape((*x.shape[0:-1], 1, x.shape[-1]))
    w = w.reshape((*w.shape[0:-2], 1, w.shape[-2], w.shape[-1])).T()
    return (x * w).sum(-1).reshape((*x.shape[0:-2], -1))


@procedure_set.register()
def T(x):
    perm = list(range(x.ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return x.permute(tuple(perm))


@procedure_set.register()
def getitem(self, val):
    # Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
    def normalize_int(e, i, dim_sz):
        if -dim_sz <= e < dim_sz:
            return e if e != -1 else dim_sz - 1
        raise IndexError(f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}")

    orig_slices = list(val) if isinstance(val, tuple) else [val]
    count = defaultdict(list)
    for i, v in enumerate(orig_slices):
        count[type(v) if not isinstance(v, slope.core.Tensor) else "tensor"] += [i]

    if (num_slices := len(count[int]) + len(count[slice_py]) + len(count["tensor"])) > len(self.shape):
        raise IndexError(f"too many indices for tensor of dimension {len(self.shape)}")
    if len(ellipsis_found := count[type(Ellipsis)]) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
    orig_slices[ellipsis_idx : ellipsis_idx + 1] = [slice_py(None)] * (len(self.shape) - num_slices)

    valid_slices = [v for v in orig_slices if v is not None]
    valid_slices = [
        v if isinstance(v, slice_py) else slice_py(y_ := normalize_int(v, i, dim_sz), y_ + 1) if isinstance(v, int) else slice_py(None)
        for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))
    ]

    start, stop, strides = zip(*y) if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)]) else ((), (), ())
    new_slice = tuple((s, e) if st > 0 else (e + 1, s + 1) for s, e, st in zip(start, stop, strides))
    sliced_tensor = self.padslice(new_slice).flip(dim=tuple([i for i, s in enumerate(strides) if s < 0]))
    new_shape = sliced_tensor.shape
    if any(abs_py(s) != 1 for s in strides):
        strides = tuple(abs_py(s) for s in strides)
        # Pad: add pad at the end: [dim_sz] -> [dim_sz_padded]
        padded_tensor = sliced_tensor.pad(
            tuple((0, s - (dim_sz % s) if dim_sz % s != 0 else 0) for s, dim_sz in zip(strides, sliced_tensor.shape))
        )
        # Reshape: [dim_sz_padded] -> [dim_sz_padded // s, s]
        reshaped_tensor = padded_tensor.reshape(flatten([sh // s, s] for sh, s in zip(padded_tensor.shape, strides)))
        new_shape = reshaped_tensor.shape[::2]
        # Shrink: do [:, 0]
        sliced_tensor = reshaped_tensor.padslice(tuple(flatten(((0, sh), (0, 1)) for sh in new_shape)))

    final_shape, it_shape, dim, tensors, dim_collapsed = (
        [],
        iter(new_shape),
        [],
        [],
        0,
    )
    for i, s in enumerate(orig_slices):
        if s is None:
            final_shape.append(1)
        else:  # s is int or slice or Tensor
            dim_shape = next(it_shape)
            if isinstance(s, int):
                dim_collapsed += 1
            else:
                final_shape.append(dim_shape)
                if isinstance(s, slope.core.Tensor):
                    tensors.append(s)
                    dim.append(i - dim_collapsed)
    ret = sliced_tensor.reshape(tuple(final_shape))

    if tensors:  # Fancy/tensor indexing
        # normalize idx
        idx = [t.sign().neg().relu() * ret.shape[d] + t for d, t in zip(dim, tensors)]
        max_dim = max(i.ndim for i in idx)
        # compute sum_dim, arange, and idx
        sum_dim = [d if n == 0 else d + max_dim - n for n, d in enumerate(dim)]
        slice_arange = [
            slope.arange(
                ret.shape[d],
                dtype=slope.int32,
                requires_grad=False,
                device=self.device,
            ).reshape(
                *[1] * sd,
                ret.shape[d],
                *[1] * (ret.ndim + max_dim - n - sd - 1),
            )
            for n, (sd, d) in enumerate(zip(sum_dim, dim))
        ]
        first_idx = [
            idx[0].reshape(
                *[1] * dim[0],
                *[1] * (1 + max_dim - idx[0].ndim),
                *idx[0].shape,
                *[1] * (ret.ndim - dim[0] - 1),
            )
        ]
        rest_idx = [
            i.reshape(
                *[1] * dim[0],
                *[1] * (max_dim - i.ndim),
                *i.shape,
                *[1] * (ret.ndim - dim[0] - n),
            )
            for n, i in enumerate(idx[1:], 1)
        ]
        idx = first_idx + rest_idx
        ret = ret.reshape(
            *ret.shape[: sum_dim[0] + 1],
            *[1] * max_dim,
            *ret.shape[sum_dim[0] + 1 :],
        )
        # iteratively fancy index
        for a, i, sd in zip(slice_arange, idx, sum_dim):
            ret = (a == i).mul(ret).sum(sd)
        # special permute case
        if dim[0] != 0 and len(dim) != 1 and dim != list(range(dim[0], dim[-1] + 1)):
            ret_dims = list(range(ret.ndim))
            ret = ret.permute(ret_dims[dim[0] : dim[0] + max_dim] + ret_dims[: dim[0]] + ret_dims[dim[0] + max_dim :])
    return ret


@procedure_set.register(static_argnames=("arg", "value"))
def padslice(x, arg: Sequence[Optional[Tuple[int, int]]], value: float = 0):
    def flatten_seq(l: Iterator):
        return [item for sublist in l for item in sublist]

    # some dim are pad, some are sliced
    arg_ = tuple([a if a is not None else (0, s) for s, a in zip(x.shape, arg)])
    padding = tuple([(max_py(0, -p[0]), max_py(0, p[1] - x.shape[i])) for i, p in enumerate(arg_)])
    x = x.pad(flatten_seq(padding), value=value)  # flatten
    starts, limits, strides = tuple(zip(*[(p[0] + padding[i][0], p[1] + padding[i][0], 1) for i, p in enumerate(arg_)]))
    x = x.slice(starts, limits, strides)
    return x


@procedure_set.register(static_argnames="padding value")
def pad2d(x, padding: Union[List[int], Tuple[int, ...]], value: float = 0):
    # (padding_left, padding_right, padding_top, padding_bottom)
    slc = [(-p0, s + p1) for p0, p1, s in zip(padding[::2], padding[1::2], x.shape[::-1])][::-1]
    return x.padslice([(0, s) for s in x.shape[: -(len(padding) // 2)]] + slc, value=value)


@procedure_set.register(static_argnames="dim")
def gather(x, idx, dim: int):
    assert idx.ndim == x.ndim, "x.ndim must equal idx.ndim"
    assert all(s >= i for s, i in zip(x.shape, idx.shape)), "all dim of idx.shape must be smaller than x.shape"
    if dim < 0:
        dim += x.ndim
    idx = idx.transpose(ax=dim, aw=0).expand_dims(-1)
    permarg = list(range(x.ndim))
    permarg = permarg[1:dim] + [permarg[0]] + permarg[dim + 1 :] + [permarg[dim]] if dim != 0 else permarg[1:] + [permarg[0]]
    return (
        (
            (
                idx
                == slope.arange(
                    x.shape[dim],
                    dtype=slope.int32,
                    requires_grad=False,
                    device=x.device,
                )
            )
            * x.permute(*permarg).padslice(tuple([*[(0, sh) for sh in idx.shape[1:-1]], (0, x.shape[dim])])).expand_dims(0)
        )
        .sum(-1)
        .transpose(ax=0, aw=dim)
    )


@procedure_set.register(static_argnames="dim")
@staticmethod
def stack(tensors, dim=0):
    first = tensors[0].expand_dims(dim)
    expand_dimsd_tensors = [tensor.expand_dims(dim) for tensor in tensors[1:]]
    return first.cat(*expand_dimsd_tensors, dim=dim)


@procedure_set.register(static_argnames="repeats")
def repeat(x, repeats):
    base_shape = (1,) * (len(repeats) - x.ndim) + x.shape
    new_shape = [x for b in base_shape for x in [1, b]]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r * s for r, s in zip(repeats, base_shape)]
    return x.reshape(new_shape).broadcast(expand_shape).reshape(final_shape)


@procedure_set.register(static_argnames="dim")
def split(x, num: int, dim: int):
    dim, step = dim + x.ndim if dim < 0 else dim, math.ceil(x.shape[dim] / num)
    slice_params = [[slice(None)] * dim + [slice(k, k + step)] for k in range(0, x.shape[dim], step)]
    return tuple(x[tuple(sl)] for sl in slice_params)


@procedure_set.register(static_argnames="dim")
def squeeze(x, dim=None):
    if dim is None:
        return x if 1 not in x.shape else x.reshape(*[size for size in x.shape if size != 1])
    if dim <= 0 and x.ndim == 0:
        return x  # This is to match PyTorch behavior
    if not -x.ndim <= dim < x.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-x.ndim if x.ndim > 0 else x.ndim-1}, {x.ndim-1 if x.ndim > 0 else x.ndim}], but got {dim})"
        )
    if dim < 0:
        dim += x.ndim
    return x if x.shape[dim] != 1 else x.reshape(*[size for idx, size in enumerate(x.shape) if idx != dim])


@procedure_set.register(static_argnames="dim")
def expand_dims(x, dim):
    if dim < 0:
        dim = len(x.shape) + dim + 1
    return x.reshape(x.shape[:dim] + (1,) + x.shape[dim:])


@procedure_set.register(static_argnames="ax aw")
def transpose(x, ax=1, aw=0):
    order = list(range(len(x.shape)))
    order[ax], order[aw] = order[aw], order[ax]
    return x.permute(tuple(order))


@procedure_set.register(static_argnames="start_dim")
def flatten(x, start_dim=0):
    return x.reshape(shape=x.shape[:start_dim] + (-1,))


@procedure_set.register(static_argnames="dim")
def cumsum(x, dim: int = 0):
    return x.transpose(dim, -1).pad((x.shape[dim] - 1, 0)).pool((x.shape[dim],)).sum(-1).transpose(dim, -1)


@staticmethod
@procedure_set.register(static_argnames="start stop step")
def arange_with_cumsum(start, stop=None, step=1):
    if stop is None:
        stop, start = start, 0
    return slope.full((math.ceil((stop - start) / step),), step).cumsum() + (start - step)


@procedure_set.register(static_argnames="dtype")
def one_hot(x, k, dtype=Tensor.int32):
    return (x[:, None].cast(dtype) == slope.arange(k, dtype=dtype)).cast(dtype)


numpy_backend = Backend(operator_set, procedure_set, compiler)
