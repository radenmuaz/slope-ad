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

from slope.base_backend import BaseBackend
from slope import ops
import pickle


class NumpyBackend(BaseBackend):
    default_dtype = np.float32

    class NumpyOpImpl(BaseBackend.BaseOpImpl):
        ir_args = ()
        ir_kwargs = ()

        @classmethod
        def do(cls, *args, **kwargs):
            exec_locals = {
                **{ir_a: a.val for ir_a, a in zip(cls.ir_args, args)},
                **kwargs,
            }
            safe_builtins = {"math": math, "np": np}
            code = cls.ir(
                *cls.ir_args, **{**{kwa: kwa for kwa in cls.ir_kwargs}, "ret": "ret"}
            )
            exec(compile(code, '<string>', 'exec'), safe_builtins, exec_locals)
            return Array(ArrayBuffer(exec_locals["ret"]))
    class ConvertImpl(NumpyOpImpl):
        ir_args = ("x",)
        ir_kwargs = ("dtype",)

        @classmethod
        def ir(cls, x: str, *, dtype: str, ret: str):
            return f"{ret} = np.astype({x}, {dtype})"
    
    class StopGradientImpl(NumpyOpImpl):
        ir_args = ("x",)

        @classmethod
        def ir(cls, x: str, *, dtype: str, ret: str):
            return f"{ret} = {x}"
        
    class NegImpl(NumpyOpImpl):
        ir_args = ("x",)

        @classmethod
        def ir(cls, x: str, y: str, *, ret: str):
            return f"{ret} = np.neg({x})"
    
    class SqrtImpl(NumpyOpImpl):
        ir_args = ("x",)

        @classmethod
        def ir(cls, x: str, y: str, *, ret: str):
            return f"{ret} = np.sqrt({x})"
    
    
    class ExpImpl(NumpyOpImpl):
        ir_args = ("x",)

        @classmethod
        def ir(cls, x: str, *, ret: str):
            return f"{ret} = np.exp({x})"

    class LogImpl(NumpyOpImpl):
        ir_args = ("x",)

        @classmethod
        def ir(cls, x: str, *, ret: str):
            return f"{ret} = np.log({x})"

    class AddImpl(NumpyOpImpl):
        ir_args = ("x", "y")

        @classmethod
        def ir(cls, x: str, y: str, *, ret: str):
            return f"{ret} = np.add({x}, {y})"
        
    class SubImpl(NumpyOpImpl):
        ir_args = ("x", "y")

        @classmethod
        def ir(cls, x: str, y: str, *, ret: str):
            return f"{ret} = np.sub({x}, {y})"

    class MulImpl(NumpyOpImpl):
        ir_args = ("x", "y")

        @classmethod
        def ir(cls, x: str, y: str, *, ret: str):
            return f"{ret} = np.mul({x}, {y})"

    class DivImpl(NumpyOpImpl):
        ir_args = ("x", "y")

        @classmethod
        def ir(cls, x: str, y: str, *, ret: str):
            return f"{ret} = np.div({x}, {y})"
    
    class ConstantImpl(NumpyOpImpl):
        ir_args = ()
        ir_kwargs = ("val", "dtype")

        @classmethod
        def ir(cls, *, val: str, dtype: str, ret: str):
            return f"{ret} = np.array(val, dtype={dtype})"
    
    class ArangeImpl(NumpyOpImpl):
        ir_args = ()
        ir_kwargs = ("start", "stop", "stride", "dtype")

        @classmethod
        def ir(cls, *, start: str, stop: str , stride: str, dtype: str, ret: str):
            return f"{ret} = np.array({start}, {stop}, {stride}, dtype={dtype})"

    class FullImpl(NumpyOpImpl):
        ir_kwargs = ("fill_value", "shape")

        @classmethod
        def ir(cls,  *, fill_value: str, shape: str, dtype: str,ret: str):
            return f"{ret} = np.full({fill_value}, {shape}, dtype={dtype})"
    
    class RandomUniformImpl(NumpyOpImpl):
        ir_kwargs = ("shape")

        @classmethod
        def ir(cls, *, shape: str, dtype: str, ret: str):
            return f"{ret} = np.random.uniform(size={shape}, dtype={dtype})"
    
    class RandomNormalImpl(NumpyOpImpl):
        ir_kwargs = ("shape")

        @classmethod
        def ir(cls, *, shape: str, dtype: str, ret: str):
            return f"{ret} = np.random.normal(size={shape}, dtype={dtype})"

    class BroadcastImpl(NumpyOpImpl):
        ir_args = ("x",)
        ir_kwargs = ("shape", "axes")

        @classmethod
        def ir(cls, x: str, *, shape, axes, ret):
            return f"""
{ret}_shape = {shape} 
{ret}_axes = {axes} 
{ret} = {x}
if not {ret}_axes is None:
    for a in sorted({ret}_axes):
        {ret} = np.expand_dims({ret},a)
{ret} = np.broadcast_to({ret}, {ret}_shape)
"""
    class ReshapeImpl(NumpyOpImpl):
        ir_args = ("x",)
        ir_kwargs = ("shape",)

        @classmethod
        def ir(cls, x: str, *, shape: str, ret: str):
            return f"{ret} = np.reshape({x}, {shape})"
    
    class PadImpl(NumpyOpImpl):
        ir_args = ("x",)
        ir_kwargs = ("lo", "hi", "interior", "value")

        @classmethod
        def ir(cls, x: str, *, lo, hi, interior, value, ret: str):
            return f"""
assert {interior} is None, "Not supported for numpy backend"
args = {lo}, {hi}, {interior}, {value}
{ret} = np.pad({x}, list(zip(lo,hi)), constant_values={value})
"""
    
    class SliceImpl(NumpyOpImpl):
        ir_args = ("x",)
        ir_kwargs = ("starts", "limits", "strides")

        @classmethod
        def ir(cls, x: str, *, starts, limits, strides, ret: str):
            return f"""
{ret} = x[[slice(s, l, st] for ])
            """
    
    class ConcatenateImpl(NumpyOpImpl):
        ir_args = ("xs",)
        ir_kwargs = ("axes")

        @classmethod
        def ir(cls, xs: str, *, axes, ret: str):
            return f"""
{ret} = np.concatenate({xs}, {axes})
"""
        
    class TransposeImpl(NumpyOpImpl):
        ir_args = ("x",)
        ir_kwargs = ("axes",)

        @classmethod
        def ir(cls, x: str, *, axes: str, ret: str):
            return f"{ret} = np.transpose({x}, {axes})"
    
    class FlipImpl(NumpyOpImpl):
        ir_args = ("x",)
        ir_kwargs = ("axes",)

        @classmethod
        def ir(cls, x: str, *, axes, ret):
            return f"{ret} = np.flip({x}, {axes})"
    
    class ScatterImpl(NumpyOpImpl):
        ir_args = ("x",)
        ir_kwargs = ("axes",)

        @classmethod
        def ir(cls,
               inputs, scatter_indices, updates,
                *, update_window_dims, inserted_window_dims,
                    scatter_dims_to_operand_dims, index_vector_dim: int,
                    slice_sizes, result_type, 
               ret: str):
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
    
    class GatherImpl(NumpyOpImpl):
        ir_args = ("x",)
        ir_kwargs = ("axes",)

        @classmethod
        def ir(cls, operand, start_indices, 
                *, collapsed_slice_dim, start_index_map,
                    offset_dims,  index_vector_dim: int,
                    slice_sizes, result_type, 
               ret: str):
            return f"""
expanded_indices_shape = list(start_indices.shape)
if len(expanded_indices_shape) == index_vector_dim:
    expanded_indices_shape.append(1)

output_offset_dim_count = len(offset_dims)
output_shape_rank = len(offset_dims) + _rank(indices) - 1

for i in range(output_offset_dim_count):
    offset_dim = offset_dims[i]

    for i in range(len(start_index_map)):
        operand_dim_for_start_index_i = start_index_map[i]

for i in range(len(slice_sizes)):
    slice_size = slice_sizes[i]
    corresponding_input_size = operand.shape[i]


for i in range(len(collapsed_slice_dims)):
    bound = slice_sizes[collapsed_slice_dims[i]]

expanded_indices_shape.pop(index_vector_dim)
indices_shape = iter(expanded_indices_shape)

slice_sizes = (s for i, s in enumerate(slice_sizes)
            if i not in collapsed_slice_dims)
res_size= tuple(next(slice_sizes) if i in offset_dims
        else next(indices_shape) for i in range(output_shape_rank))

res = np.zeros_like(res_size)
batch_dims = [d for d in list(range(res.ndim)) if d in offset_dims]

for res_idx, _ in np.ndenumerate(res):
    batch_idx = [res_idx[d] for d in batch_dims]

    start_indices_idx = batch_idx[:]
    if index_vector_dim < start_indices.ndim:
        start_indices_idx.insert(index_vector_dim, -1)
    start_idx = start_indices[start_indices_idx]

    full_start_idx = [None]*operand.ndim
    for d in range(operand.ndim):
        dStartIt = start _index_map[d]
        if (dStartIt == start_index_map[-1]):
            continue
        dStart = dStartIt - start_index_map[0]
        full_start_idx[d] = np.clip(start_idx[d], operand.shape[d] - slice_sizes[d])
    
    offset_idx = [res_idx[d] for d in offset_dims]
    full_offset_idx = [None]*(len(offset_dims) + len(collapsed_slice_dim))
    oi = 0
    for i in range(len(full_offset_idx)):
        if i in collapsed_slice_dim:
            continue
        full_offset_idx[i] = offset_idx[oi]
        oi += 1
    operand_idx = full_start_index + full_offset_index
    result[result_index] = operand[operandIndex]
    return result
"""

    
    
    

    input_handlers = {
        ty: np.asarray for ty in [bool, int, float, np.ndarray, np.float64, np.float32]
    }

    class JitFn(BaseBackend.BaseJitFn):
        def __init__(self, code, fn):
            self.code = code
            self.fn = fn

        def __call__(self, *args, **kwargs):
            args = [a.val if isinstance(a, Array) else a for a in args]
            outs = self.fn(*args, **kwargs)
            return [Array(o) if isinstance(o, np.ndarray) else o for o in outs]

    @classmethod
    def compile(cls, prog, consts, in_avals, name) -> List[Any]:
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
        ops_code = []
        ops_code += [f"def {name}({', '.join(inb_args)}):"]
        for inb_const, const in zip(inb_consts, consts):
            ops_code += [f"    {inb_const} = pickle.loads({pickle.dumps(const.val)})"]
        for eqn in prog.instrs:
            in_vals = utils.list_map(lambda x: env[x], eqn.inputs)
            for outb in eqn.out_binders:
                env[outb] = f"z{nzs}"
                nzs += 1
            out_vals = utils.list_map(lambda z: env[z], eqn.out_binders)
            assert not len(out_vals) > 1, "Op with >1 output not supported"
            op_ir = eqn.op.get_impl().ir(*in_vals, **eqn.params, ret=out_vals[0])
            op_ir = "\n".join(["    " + line for line in op_ir.strip().split("\n")])
            ops_code += [op_ir]
            # out_vals = eqn.op.jit(in_avals, in_vals, **eqn.params)

        outs = utils.list_map(lambda y: env[y], prog.outs)
        # ops_code += [f"    outs[0]}"]
        ops_code += [f"    return {', '.join(outs)}{',' if len(outs)==1 else ''}"]
        fn_code = "\n".join(ops_code)
        exec(compile(fn_code, '<string>', 'exec'), safe_builtins, exec_locals)
        fn = exec_locals[name]
        # exec('\n'.join(ops_code), safe_builtins, exec_locals)
        return cls.JitFn(fn_code, fn)

    @classmethod
    def execute_compiled(cls, compiled, out_avals, *args):
        input_bufs = [cls.input_handlers[type(x)](x) for x in args]
        out_bufs = compiled.execute(input_bufs)
        return [cls.handle_result(aval, buf) for aval, buf in zip(out_avals, out_bufs)]

    @classmethod
    def handle_result(cls, aval: ArrayShape, buf):
        del aval
        return np.asarray(buf)
    
    _rng: np.random.Generator = np.random.default_rng()

    @classmethod
    def manual_seed(cls, seed=None):
        cls._rng = np.random.default_rng(seed=seed)
   
    # control flow
    choose = select = lambda arr, *vals, idx: Array(np.choose(idx, *vals))
    where = lambda arr, trueval, falseval: Array(np.where(arr, trueval, falseval))


