import math
from contextlib import contextmanager
import numpy as np
import itertools
from typing import Any, Optional, Union, Tuple, List, Dict
import numpy as np
import functools
import slope
from slope import utils
from slope.array import Array
from slope.array_shape import ArrayShape
from functools import lru_cache, partial
import numpy as np
from dataclasses import dataclass

from slope.base_backend import BaseBackend
from slope import ops
import inspect
import pickle


class NumpyBackend(BaseBackend):
    default_dtype = np.float32

    class NumpyOpImpl(BaseBackend.BaseOpImpl):
        ir_args = ()
        ir_kwargs = {}

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
            exec(code, safe_builtins, exec_locals)
            return Array(exec_locals["ret"])

    class ExpImpl(NumpyOpImpl):
        ir_args = "x"

        @classmethod
        def ir(cls, x: str, y: str, *, ret: str):
            return f"{ret} = np.exp({x})"

    class LogImpl(NumpyOpImpl):
        ir_args = "x"

        @classmethod
        def ir(cls, x: str, y: str, *, ret: str):
            return f"{ret} = np.log({x})"

    class SubImpl(NumpyOpImpl):
        ir_args = ("x", "y")

        @classmethod
        def ir(cls, x: str, y: str, *, ret: str):
            return f"{ret} = np.add({x}, {y})"

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

    class FullImpl(NumpyOpImpl):
        ir_kwargs = ("fill_value", "shape")

        @classmethod
        def ir(cls, fill_value: str, shape: str, *, ret: str):
            return f"{ret} = np.full({fill_value}, {shape})"

    class AddImpl(NumpyOpImpl):
        ir_args = ("x", "y")

        @classmethod
        def ir(cls, x: str, y: str, *, ret: str):
            return f"{ret} = np.add({x}, {y})"

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
            ops_code += [f'    {inb_const} = pickle.loads({pickle.dumps(const.val)})']
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
        fn_code = '\n'.join(ops_code)
        exec(fn_code, safe_builtins, exec_locals)
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

    @staticmethod
    def eye(dim, **kwargs):
        return Array(np.eye(dim), **kwargs)

    @staticmethod
    def arange(stop, start=0, step=1, **kwargs):
        return Array(
            np.arange(start=start, stop=stop, step=step, dtype=np.float32), **kwargs
        )

    _rng: np.random.Generator = np.random.default_rng()

    @classmethod
    def manual_seed(cls, seed=None):
        cls._rng = np.random.default_rng(seed=seed)

    @classmethod
    def rand(cls, *shape, **kwargs):
        return Array(
            np.array(
                cls._rng.random(
                    size=shape, dtype=kwargs.get("dtype", cls.default_dtype)
                ),
            ),
            **kwargs,
        )

    @classmethod
    def randn(cls, *shape, **kwargs):
        return Array(
            np.array(
                cls._rng.standard_normal(
                    size=shape, dtype=kwargs.get("dtype", cls.default_dtype)
                ),
            ),
            **kwargs,
        )

    @staticmethod
    def uniform(*shape, **kwargs):
        return Array.rand(*shape, **kwargs) * 2 - 1

    @staticmethod
    def scaled_uniform(*shape, **kwargs):
        return Array.uniform(*shape, **kwargs).mul(math.prod(shape) ** -0.5)

    @staticmethod
    def glorot_uniform(*shape, **kwargs):
        return Array.uniform(*shape, **kwargs).mul(
            (6 / (shape[0] + math.prod(shape[1:]))) ** 0.5
        )

    @staticmethod
    def stop_gradient(cls, arr):
        return cls.zeros_like(arr)

    def max(arr, axes=None, keepdims=False):
        return Array(np.max(arr.val, axis=axes, keepdims=keepdims))

    def sum(arr, axes=None, keepdims=False):
        return Array(np.sum(arr.val, axis=axes, keepdims=keepdims))

    @staticmethod
    def pad(arr, lo, hi, interior=None, value=0):
        if interior is None:
            interior = [1] * len(lo)
        new_shape, slices = [], []
        for s, l, h, r in zip(arr.shape, lo, hi, interior):
            stride = r + 1
            new_shape += [s * stride + l + h]
            slices += [slice(l, s * stride + l, stride)]
        padded = np.full(new_shape, value, dtype=arr.dtype)
        padded[tuple(slices)] = arr.val
        return Array(padded)

    @staticmethod
    def slice(arr, starts, limits, strides):
        return Array(
            arr.val[tuple(slice(s, l, r) for s, l, r in zip(starts, limits, strides))]
        )

    @staticmethod
    def gather(
        operand,
        startIndices,
        offsetDims,
        collapsedSliceDims,
        startIndexMap,
        indexVectorDim,
        sliceSizes,
        indicesAreSorted,
        resultType,
    ):
        result = np.empty(resultType.shape, dtype=resultType.dtype)
        batchDims = [d for d in resultType.shape if d not in offsetDims]
        for resultIndex in np.ndindex(*resultType.shape):
            resultIndex = np.array(resultIndex)

            batchIndex = np.array([resultIndex[d] for d in batchDims])

            startIndicesIndex = batchIndex.copy()
            if indexVectorDim < startIndices.ndim:
                startIndicesIndex = np.insert(
                    startIndicesIndex, indexVectorDim, slice(None)
                )
            # startIndex = evalIndex(evalSliceOp(startIndices, startIndicesIndex))
            startIndex = startIndices.slice(startIndicesIndex)

            fullStartIndex = np.zeros(operand.ndim, dtype=np.int64)
            for dOperand in operand.shape:
                dStartIt = np.where(startIndexMap == dOperand)[0]
                if len(dStartIt) == 0:
                    continue
                dStart = dStartIt[0]
                fullStartIndex[dOperand] = startIndex[dStart]

            offsetIndex = np.array([resultIndex[d] for d in offsetDims])

            fullOffsetIndex = np.zeros(
                offsetIndex.size + len(collapsedSliceDims), dtype=np.int64
            )
            oi = 0
            for i in range(fullOffsetIndex.size):
                if i in collapsedSliceDims:
                    continue
                fullOffsetIndex[i] = offsetIndex[oi]
                oi += 1

            operandIndex = fullStartIndex + fullOffsetIndex
            if np.all(np.less(operandIndex, operand.shape)):
                result[tuple(resultIndex)] = operand[tuple(operandIndex)]
        return result

    @staticmethod
    def scatter(
        inputs,
        scatterIndices,
        updates,
        updateWindowDims,
        insertedWindowDims,
        scatterDimsToOperandDims,
        indexVectorDim,
        updateComputation,
        scope,
        resultTypes,
    ):
        results = []
        for input in inputs:
            results.append(input)

        updateScatterDims = []
        for d in updates[0].getAxes():
            if d not in updateWindowDims:
                updateScatterDims.append(d)

        for updateIndexIt in np.ndindex(updates[0].shape):
            updateIndex = np.array(updateIndexIt)
            updateScatterIndex = updateIndex[updateScatterDims]

            startIndicesIndex = updateScatterIndex.copy()
            if indexVectorDim < scatterIndices.ndim:
                startIndicesIndex = np.insert(
                    startIndicesIndex, indexVectorDim, slice(None)
                )

            startIndex = scatterIndices.slice(startIndicesIndex)

            fullStartIndex = np.zeros_like(inputs[0].shape)
            for dInput in inputs[0].getAxes():
                dStart = np.where(scatterDimsToOperandDims == dInput)[0]
                if len(dStart) == 0:
                    continue
                dStart = dStart[0]
                fullStartIndex[dInput] = startIndex[dStart]

            updateWindowIndex = updateIndex[updateWindowDims]

            fullWindowIndex = np.zeros(updateWindowIndex.size + len(insertedWindowDims))
            wi = 0
            for i in range(fullWindowIndex.size):
                if i in insertedWindowDims:
                    continue
                fullWindowIndex[i] = updateWindowIndex[wi]
                wi += 1

            resultIndex = fullStartIndex + fullWindowIndex
            if not np.all(np.less(resultIndex, results[0].shape)):
                continue

            updateComputationArgs = []
            for result in results:
                resultType = np.dtype(result.getElementType())
                resultValue = result.get(resultIndex)
                updateComputationArgs.append(np.array(resultValue, dtype=resultType))

            for update in updates:
                updateType = np.dtype(update.getElementType())
                updateValue = update[tuple(updateIndex)]
                updateComputationArgs.append(np.array(updateValue, dtype=updateType))

            updatedValues = eval(updateComputation, updateComputationArgs, scope)

            for result, updatedValue in zip(results, updatedValues):
                result[resultIndex] = np.array(updatedValue)

        return results

    # control flow
    choose = select = lambda arr, *vals, idx: Array(np.choose(idx, *vals))
    where = lambda arr, trueval, falseval: Array(np.where(arr, trueval, falseval))

    # slice = lambda arr, start, end, step: Array(arr.val.__getitem__(slice(start, end, step)))
    # def broadcast_to(arr, shape):
    #     return Array(np.broadcast_to(arr.val, shape))
