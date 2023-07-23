import math
from contextlib import contextmanager
import numpy as np
import itertools
from typing import Any, Optional, Union, Tuple, List, Dict
import numpy as np
import functools
import slope
from slope import utils
from slope.array import Array, ArrayBuffer
from slope.array_shape import ArrayShape
import numpy as np

from slope.base_backend import Backend, JitFn
from slope import ops
import pickle
import inspect
numpy_backend = Backend("numpy")
for typ in [bool, int, float, np.ndarray, np.float64, np.float32]:
    numpy_backend.set_input_handler(typ, np.asarray)

@numpy_backend.set_run_op_impl
def numpy_run_op_impl(self, op_name, *args, **kwargs):
    return Array(ArrayBuffer(self.op_impls[op_name](*args, **kwargs)))


@numpy_backend.set_compile
def fn(self, prog, consts, in_avals, name) -> List[Any]:
    safe_builtins = {"math": math, "np": np, "pickle": pickle}

    exec_locals = {}
    env: Dict[slope.ad.Var, Any] = {}
    ncs = 0
    nxs = 0
    nzs = 0
    inb_args = []
    inb_consts = []
    
    for inb in prog.in_binders:
        if type(inb.aval) is not ArrayShape:
            env[inb] = f"c{ncs}"
            inb_consts += [env[inb]]
            ncs += 1
        else:
            env[inb] = f"x{nxs}"
            inb_args += [env[inb]]
            nxs += 1
    code_lines = []
    code_lines += [f"def {name}({', '.join(inb_args)}):"]
    for inb_const, const in zip(inb_consts, consts):
        code_lines += [f"    {inb_const} = pickle.loads({pickle.dumps(const.val)})"]
    multiline_op_impl_set = set()
    multiline_op_impl_defs = []
    for eqn in prog.instrs:
        in_vals = utils.list_map(lambda x: env[x], eqn.inputs)
        for outb in eqn.out_binders:
            env[outb] = f"z{nzs}"
            nzs += 1
        out_vals = utils.list_map(lambda z: env[z], eqn.out_binders)
        assert not len(out_vals) > 1, "Op with >1 output not supported"
        # op_ir = self.op_impls[eqn.op.name].ir(*in_vals, **eqn.params, ret=out_vals[0])
        op_impl_code_lines = inspect.getsourcelines(self.op_impls[eqn.op.name])
        args_str = ', '.join(in_vals)
        kwargs_str = ', '.join([f"{k}={v}" for k,v in eqn.params.items()])
        if len(op_impl_code_lines) > 2:
            if eqn.op.name not in multiline_op_impl_set:
                multiline_op_impl_set.add(eqn.op.name)
                multiline_op_impl_defs += [op_impl_code_lines]
            code_line += f"{out_vals[0]} = {eqn.op.name}({args_str}, {kwargs_str})"
        else:
            argspec = inspect.getargspec(self.op_impls[eqn.op.name])
            op_str = op_impl_code_lines[1].replace("return", "").strip()
            for argname, arg in zip(argspec.args, in_vals):
                op_str.replace(argname, arg)
            for kwargname, kwarg in eqn.params.items():
                op_str.replace(kwargname, kwarg)
            code_line = f"{out_vals[0]} = {op_str}"


        code_line = "\n".join(["    " + line for line in code_line.strip().split("\n")])
        code_lines += [code_line]
        # out_vals = eqn.op.jit(in_avals, in_vals, **eqn.params)

    outs = utils.list_map(lambda y: env[y], prog.outs)
    # ops_code += [f"    outs[0]}"]
    code_lines += [f"    return {', '.join(outs)}{',' if len(outs)==1 else ''}"]
    code_lines = multiline_op_impl_defs + code_lines
    code = "\n".join(code_lines)
    exec(compile(code, '<string>', 'exec'), safe_builtins, exec_locals)
    fn = exec_locals[name]
    # exec('\n'.join(ops_code), safe_builtins, exec_locals)
    return JitFn(code, fn)



_rng: np.random.Generator = np.random.default_rng()

@classmethod
def manual_seed(cls, seed=None):
    cls._rng = np.random.default_rng(seed=seed)

# # control flow
# choose = select = lambda arr, *vals, idx: Array(np.choose(idx, *vals))
# where = lambda arr, trueval, falseval: Array(np.where(arr, trueval, falseval))

### Op Impls

@numpy_backend.set_op_impl("convert")
def fn(x, *, dtype,):
    return np.astype(x, dtype)

@numpy_backend.set_op_impl("stop_gradient")
def fn(x, dtype):
    return x
    
@numpy_backend.set_op_impl("neg")
def fn(x,):
    return np.neg(x)

@numpy_backend.set_op_impl("sqrt")
def fn(x):
    return np.sqrt(x)


@numpy_backend.set_op_impl("exp")
def fn(x):
    return np.exp(x)


@numpy_backend.set_op_impl("log")
def fn(x):
    return np.log(x)


@numpy_backend.set_op_impl("add")
def fn(x, y):
    return np.add(x, y)


@numpy_backend.set_op_impl("sub")
def fn(x, y):
    return np.subtract(x, y)



@numpy_backend.set_op_impl("mul")
def fn(x, y):
    return np.multiply(x, y)

@numpy_backend.set_op_impl("div")
def fn(x, y):
    return np.divide(x, y)


@numpy_backend.set_op_impl("constant")
def fn(*, val, dtype):
    return np.array(val, dtype=dtype)


@numpy_backend.set_op_impl("arange")
def fn(*, start, stop, stride, dtype):
    return np.arange(start, stop, stride, dtype=dtype)

@numpy_backend.set_op_impl("full")
def fn(*, fill_value, dtype):
    return np.full(fill_value=fill_value, dtype=dtype)

@numpy_backend.set_op_impl("random_uniform")
def fn(*, shape, dtype):
    return np.random.uniform(size=shape, dtype=dtype)


@numpy_backend.set_op_impl("random_normal")
def fn(*, shape, dtype):
    return np.random.normal(size=shape, dtype=dtype)

@numpy_backend.set_op_impl("broadcast")
def fn(x, *, shape, axes):
    ret = x
    if not axes is None:
        for a in sorted(axes):
            ret = np.expand_dims(ret, a)
    ret = np.broadcast_to(ret, shape)
    return ret

@numpy_backend.set_op_impl("reshape")
def fn(x, *, shape):
    return np.reshape(x, shape)

@numpy_backend.set_op_impl("pad")
def fn(x, *, lo, hi, interior, value):
    return np.pad({x}, list(zip(lo,hi)), constant_values={value})

@numpy_backend.set_op_impl("slice")
def fn(x, *, starts, limits, strides):
        return x[[slice(s, l, st) for s, l, t in zip(starts, limits, strides)]]

@numpy_backend.set_op_impl("concatenate")
def fn(xs, *, axes, ret: str):
    return np.concatenate(xs, axes)
    
@numpy_backend.set_op_impl("transpose")
def fn(x, *, axes, ret):
    return np.transpose(x, axes)


@numpy_backend.set_op_impl("flip")
def fn(x, *, axes, ret):
    return np.flip(x, axes)

@numpy_backend.set_op_impl("scatter")
def fn(inputs, scatter_indices, updates,
            *, update_window_dims, inserted_window_dims,
                scatter_dims_to_operand_dims, index_vector_dim: int,
                slice_sizes, result_type, 
            ret):
        return """
# SmallVector<Tensor> evalScatterOp(
#     ArrayRef<Tensor> inputs, const Tensor &scatterIndices,
#     ArrayRef<Tensor> updates, const Axes &updateWindowDims,
#     const Axes &insertedWindowDims, const Axes &scatterDimsToOperandDims,
#     Axis indexVectorDim, Region &updateComputation, Scope &scope,
#     ArrayRef<ShapedType> resultTypes) {
#   SmallVector<Tensor> results;
#   for (auto input : inputs) results.push_back(input);

#   Axes updateScatterDims;
#   for (auto d : updates[0].getAxes())
#     if (!llvm::is_contained(updateWindowDims, d))
#       updateScatterDims.push_back(d);

#   for (auto updateIndexIt = updates[0].index_begin();
#        updateIndexIt != updates[0].index_end(); ++updateIndexIt) {
#     auto updateIndex = *updateIndexIt;
#     Index updateScatterIndex;
#     for (auto d : updateScatterDims)
#       updateScatterIndex.push_back(updateIndex[d]);

#     auto startIndicesIndex = updateScatterIndex;
#     if (indexVectorDim < scatterIndices.getRank())
#       startIndicesIndex.insert(startIndicesIndex.begin() + indexVectorDim,
#                                kColon);
#     auto startIndex = evalIndex(evalSliceOp(scatterIndices, startIndicesIndex));

#     Index fullStartIndex(inputs[0].getRank(), 0);
#     for (auto dInput : inputs[0].getAxes()) {
#       auto dStartIt = llvm::find(scatterDimsToOperandDims, dInput);
#       if (dStartIt == scatterDimsToOperandDims.end()) continue;
#       auto dStart = dStartIt - scatterDimsToOperandDims.begin();
#       fullStartIndex[dInput] = startIndex[dStart];
#     }

#     Index updateWindowIndex;
#     for (auto d : updateWindowDims) updateWindowIndex.push_back(updateIndex[d]);

#     Index fullWindowIndex(updateWindowIndex.size() + insertedWindowDims.size(),
#                           0);
#     for (size_t i = 0, wi = 0; i < fullWindowIndex.size(); ++i) {
#       if (llvm::is_contained(insertedWindowDims, i)) continue;
#       fullWindowIndex[i] = updateWindowIndex[wi++];
#     }

#     auto resultIndex = fullStartIndex + fullWindowIndex;
#     if (!resultIndex.inBounds(results[0].getShape())) continue;

#     SmallVector<InterpreterValue> updateComputationArgs;
#     for (auto result : results)
#       updateComputationArgs.push_back(
#           Tensor(RankedTensorType::get({}, result.getElementType()),
#                  result.get(resultIndex)));
#     for (auto update : updates)
#       updateComputationArgs.push_back(
#           Tensor(RankedTensorType::get({}, update.getElementType()),
#                  update.get(updateIndex)));

#     auto updatedValues = eval(updateComputation, updateComputationArgs, &scope);
#     for (auto [result, updatedValue] : llvm::zip(results, updatedValues))
#       result.set(resultIndex, updatedValue.getTensor().get({}));
#   }

#   return results;
# }
"""

@numpy_backend.set_op_impl("gather")
def fn(operand, start_indices, 
            *, collapsed_slice_dims, start_index_map,
                offset_dims,  index_vector_dim: int,
                slice_sizes):
    expanded_indices_shape = list(start_indices.shape)
    if len(expanded_indices_shape) == index_vector_dim:
        expanded_indices_shape.append(1)

    output_shape_rank = len(offset_dims) + start_indices.ndim - 1

    expanded_indices_shape.pop(index_vector_dim)
    indices_shape = iter(expanded_indices_shape)

    slice_sizes = (s for i, s in enumerate(slice_sizes)
            if i not in collapsed_slice_dims)
    res_size= tuple(next(slice_sizes) if i in offset_dims
        else next(indices_shape) for i in range(output_shape_rank))

    res = np.zeros(res_size)
    batch_dims = [d for d in list(range(res.ndim)) if d in offset_dims]

    for res_idx, _ in np.ndenumerate(res):
        batch_idx = [res_idx[d] for d in batch_dims]

    start_indices_idx = batch_idx[:]
    if index_vector_dim < start_indices.ndim:
        start_indices_idx.insert(index_vector_dim, -1)
    start_idx = start_indices[start_indices_idx]

    full_start_idx = [None]*operand.ndim
    for d in range(operand.ndim):
        dStartIt = start_index_map[d]
        if (dStartIt == start_index_map[-1]):
            continue
        dStart = dStartIt - start_index_map[0]
        full_start_idx[d] = np.clip(start_idx[d], operand.shape[d] - slice_sizes[d])

    offset_idx = [res_idx[d] for d in offset_dims]
    full_offset_idx = [None]*(len(offset_dims) + len(collapsed_slice_dims))
    oi = 0
    for i in range(len(full_offset_idx)):
        if i in collapsed_slice_dims:
            continue
        full_offset_idx[i] = offset_idx[oi]
        oi += 1
    operand_idx = full_start_idx + full_offset_idx
    res[res_idx] = operand[operand_idx]
    return res
