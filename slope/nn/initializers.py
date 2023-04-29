import numpy as np

def _compute_fans(shape: core.NamedShape, in_axis=-2, out_axis=-1,
                  batch_axis=()):
  """
  Compute effective input and output sizes for a linear or convolutional layer.

  Axes not in in_axis, out_axis, or batch_axis are assumed to constitute the
  "receptive field" of a convolution (kernel spatial dimensions).
  """
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
  receptive_field_size = shape.total / in_size / out_size / batch_size
  fan_in = in_size * receptive_field_size
  fan_out = out_size * receptive_field_size
  return fan_in, fan_out

def normal(stddev=1e-2, dtype: DType = jnp.float_) -> Callable:
  """Builds an initializer that returns real normally-distributed random arrays.

  Args:
    stddev: optional; the standard deviation of the distribution.
    dtype: optional; the initializer's default dtype.

  Returns:
    An initializer that returns arrays whose values are normally distributed
    with mean ``0`` and standard deviation ``stddev``.

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.normal(5.0)
  >>> initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)  # doctest: +SKIP
  DeviceArray([[ 3.0613258 ,  5.6129413 ,  5.6866574 ],
               [-4.063663  , -4.4520254 ,  0.63115686]], dtype=float32)
   """
  def init(key, shape, dtype=dtype):
    return np.random.normal(key, shape, dtype) * stddev
  return init


def variance_scaling(scale, mode: str, distribution: str,
                     in_axis: Union[int, Sequence[int]] = -2,
                     out_axis: Union[int, Sequence[int]] = -1,
                     batch_axis: Sequence[int] = (),
                     dtype: DType = jnp.float_) -> Callable:
  r"""
  Initializer that adapts its scale to the shape of the weights tensor.

  With ``distribution="truncated_normal"`` or ``distribution="normal"``, samples
  are drawn from a (truncated) normal distribution with a mean of zero
  and a standard deviation (after truncation, if applicable) of
  :math:`\sqrt{\frac{scale}{n}}`, where `n` is:

  * the number of input units in the weights tensor, if ``mode="fan_in"``,
  * the number of output units, if ``mode="fan_out"``, or
  * the average of the numbers of input and output units, if ``mode="fan_avg"``.

  This initializer can be configured with ``in_axis``, ``out_axis``, and
  ``batch_axis`` to work with general convolutional or dense layers; axes that
  are not in any of those arguments are assumed to be the "receptive field"
  (convolution kernel spatial axes).

  With ``distribution="truncated_normal"``, the absolute values of the samples
  are truncated at 2 standard deviations before scaling.

  With ``distribution="uniform"``, samples are drawn from:

  * a uniform interval, if `dtype` is real, or
  * a uniform disk, if `dtype` is complex,

  with a mean of zero and a standard deviation of ``stddev``.

  Args:
    scale: scaling factor (positive float).
    mode: one of ``"fan_in"``, ``"fan_out"``, and ``"fan_avg"``.
    distribution: random distribution to use. One of ``"truncated_normal"``,
      ``"normal"`` and ``"uniform"``.
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.
  """

  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    shape = core.as_named_shape(shape)
    fan_in, fan_out = _compute_fans(shape, in_axis, out_axis, batch_axis)
    if mode == "fan_in": denominator = fan_in
    elif mode == "fan_out": denominator = fan_out
    elif mode == "fan_avg": denominator = (fan_in + fan_out) / 2
    else:
      raise ValueError(
        f"invalid mode for variance scaling initializer: {mode}")
    variance = jnp.array(scale / denominator, dtype=dtype)

    if distribution == "truncated_normal":
      if jnp.issubdtype(dtype, jnp.floating):
        # constant is stddev of standard normal truncated to (-2, 2)
        stddev = jnp.sqrt(variance) / jnp.array(.87962566103423978, dtype)
        return random.truncated_normal(key, -2, 2, shape, dtype) * stddev
      else:
        # constant is stddev of complex standard normal truncated to 2
        stddev = jnp.sqrt(variance) / jnp.array(.95311164380491208, dtype)
        return _complex_truncated_normal(key, 2, shape, dtype) * stddev
    elif distribution == "normal":
      return random.normal(key, shape, dtype) * jnp.sqrt(variance)
    elif distribution == "uniform":
      if jnp.issubdtype(dtype, jnp.floating):
        return random.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)
      else:
        return _complex_uniform(key, shape, dtype) * jnp.sqrt(variance)
    else:
      raise ValueError(f"invalid distribution for variance scaling initializer: {distribution}")

  return init


def glorot_normal(in_axis: Union[int, Sequence[int]] = -2,
                  out_axis: Union[int, Sequence[int]] = -1,
                  batch_axis: Sequence[int] = (),
                  dtype: DType = jnp.float_) -> Callable:
  """Builds a Glorot normal initializer (aka Xavier normal initializer).

  A `Glorot normal initializer`_ is a specialization of
  :func:`jax.nn.initializers.variance_scaling` where ``scale = 1.0``,
  ``mode="fan_avg"``, and ``distribution="truncated_normal"``.

  Args:
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.

  Returns:
    An initializer.

  Example:

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.glorot_normal()
  >>> initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)  # doctest: +SKIP
  DeviceArray([[ 0.41770416,  0.75262755,  0.7619329 ],
               [-0.5516644 , -0.6028657 ,  0.08661086]], dtype=float32)

  .. _Glorot normal initializer: http://proceedings.mlr.press/v9/glorot10a.html
  """
  return variance_scaling(1.0, "fan_avg", "truncated_normal", in_axis=in_axis,
                          out_axis=out_axis, batch_axis=batch_axis, dtype=dtype)

xavier_normal = glorot_normal
