import numpy as np
from typing import (
    Sequence,
    Callable,
    Union,
    Callable,
)
import math
import slope as sp


def compute_fans(shape: Sequence, in_axis=-2, out_axis=-1, batch_axis=()):
    if isinstance(in_axis, int):
        in_size = shape[in_axis]
    else:
        in_size = int(np.prod([shape[i] for i in in_axis]))
    if isinstance(out_axis, int):
        out_size = shape[out_axis]
    else:
        out_size = int(np.prod([shape[i] for i in out_axis]))
    if isinstance(batch_axis, int):
        batch_size = shape[batch_axis]
    else:
        batch_size = int(np.prod([shape[i] for i in batch_axis]))
    receptive_field_size = math.prod(shape) / in_size / out_size / batch_size
    fan_in = in_size * receptive_field_size
    fan_out = out_size * receptive_field_size
    return fan_in, fan_out


def normal(dtype=np.float32) -> Callable:
    def init(shape, dtype=dtype):
        return sp.rt.ops.randn(shape)

    return init


def variance_scaling(
    scale,
    mode: str,
    distribution: str,
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype=np.float32,
) -> Callable:
    def init(shape, dtype=dtype):
        fan_in, fan_out = compute_fans(shape, in_axis, out_axis, batch_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")
        variance = sp.rt.array(scale / denominator, dtype=dtype)
        if distribution == "normal":
            return sp.rt.ops.randn(shape) * variance.sqrt()
        elif distribution == "uniform":
            return sp.rt.array(
                sp.rt.ops.rand(size=shape.astype(dtype)) * (3 * variance).sqrt()
            )
        else:
            raise ValueError(
                f"invalid distribution for variance scaling initializer: {distribution}"
            )

    return init


def glorot_normal(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype=sp.core.BaseArray.float32,
) -> Callable:
    return variance_scaling(
        1.0,
        "fan_avg",
        "normal",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )


def glorot_uniform(
    in_axis: Union[int, Sequence[int]] = -2,
    out_axis: Union[int, Sequence[int]] = -1,
    batch_axis: Sequence[int] = (),
    dtype=sp.core.BaseArray.float32,
) -> Callable:
    return variance_scaling(
        1.0,
        "fan_avg",
        "uniform",
        in_axis=in_axis,
        out_axis=out_axis,
        batch_axis=batch_axis,
        dtype=dtype,
    )
