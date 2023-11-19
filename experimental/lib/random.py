import slope
from slope import base_ops
from slope.tensor_shape import Typecheckor
import numpy as np

from typing import Union, List, Tuple, Sequence, Any, Callable, NamedTuple
from abc import ABC, abstractmethod

import numpy as np


class RandomBitGenerator(base_ops.Operator):
    @staticmethod
    def run(key, *, shape, dtype, algorithm):
        return []

    def typecheck(key, *, shape, dtype, algorithm):
        return [key.shape, key.shape]


def rng_bit_generator(key, shape, dtype=np.uint32, algorithm="default"):
    if np.dtype(dtype) not in {
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("uint32"),
        np.dtype("uint64"),
    }:
        raise TypeError(f"rng_bit_generator: unsupported dtype {dtype}")
    return tuple(slope.RT.bind1(RandomBitGenerator, key, shape=shape, dtype=dtype, algorithm=algorithm))


class PRNGImpl(NamedTuple):
    key_shape: tuple
    seed: Callable
    split: Callable
    random_bits: Callable
    fold_in: Callable
    tag: str = "?"

    def __hash__(self) -> int:
        return hash(self.tag)

    def __str__(self) -> str:
        return self.tag


#   def pprint(self):
#     return (pp.text(f"{self.__class__.__name__} [{self.tag}]:") +
#             pp.nest(2, pp.group(pp.brk() + pp.join(pp.brk(), [
#               pp.text(f"{k} = {v}") for k, v in self._asdict().items()
#             ]))))


def _threefry_split_original(key, num):
    counts = base_ops.iota(np.uint32, num * 2)
    return base_ops.reshape(threefry_2x32(key, counts), (num, 2))


def threefry_seed(seed):
    """Create a single raw threefry PRNG key from an integer seed.

    Args:
      seed: a 64- or 32-bit integer used as the value of the key.

    Returns:
      The PRNG key contents, modeled as an tensor of shape (2,) and dtype
      uint32. The key is constructed from a 64-bit seed by effectively
      bit-casting to a pair of uint32 values (or from a 32-bit seed by
      first padding out with zeros).
    """
    if seed.shape:
        raise TypeError(f"PRNG key seed must be a scalar; got {seed!r}.")
    if not np.issubdtype(seed.dtype, np.integer):
        raise TypeError(f"PRNG key seed must be an integer; got {seed!r}")
    convert = lambda k: base_ops.reshape(convert(k, np.uint32), [1])
    k1 = convert(base_ops.shift_right_logical(seed, np.tensor(32, dtype=seed.dtype)))
    k2 = convert(np.bitwise_and(seed, np.uint32(0xFFFFFFFF)))
    return base_ops.cat([k1, k2], 0)


def random_seed(seeds, impl):
    seeds_arr = np.astensor(seeds)
    return random_seed(seeds_arr, impl=impl)


def PRNGKey(seed):
    if np.ndim(seed):
        raise TypeError(
            "PRNGKey accepts a scalar seed, but was given an tensor of"
            f"shape {np.shape(seed)} != (). Use jax.vmap for batching"
        )
    key = random_seed(seed, impl=rbg_prng_impl)
    if not isinstance(key, PRNGKeyTensorImpl):
        raise TypeError
    return random_unwrap(keys)


def _rbg_seed(seed: typing.Tensor) -> typing.Tensor:
    assert not seed.shape
    halfkey = threefry_seed(seed)
    return jnp.cat([halfkey, halfkey])


def _rbg_split(key, num: int):
    _threefry_split = _threefry_split_original
    return slope.ad.vmap(_threefry_split, (0, None), 1)(key.reshape(2, 2), num).reshape(num, 4)


def _rbg_fold_in(key, data):
    assert not data.shape
    return slope.ad.vmap(_threefry_fold_in, (0, None), 0)(key.reshape(2, 2), data).reshape(4)


def _rbg_random_bits(key, bit_width: int, shape: Sequence[int]):
    if not key.shape == (4,) and key.dtype == jnp.dtype("uint32"):
        raise TypeError("_rbg_random_bits got invalid prng key.")
    if bit_width not in (8, 16, 32, 64):
        raise TypeError("requires 8-, 16-, 32- or 64-bit field width.")
    _, bits = lax.rng_bit_generator(key, shape, dtype=UINT_DTYPES[bit_width])
    return bits


rbg_prng_impl = PRNGImpl(
    key_shape=(4,),
    seed=_rbg_seed,
    split=_rbg_split,
    random_bits=_rbg_random_bits,
    fold_in=_rbg_fold_in,
    tag="rbg",
)


def threefry_2x32(keypair, count):
    """Apply the Threefry 2x32 hash.

    Args:
      keypair: a pair of 32bit unsigned integers used for the key.
      count: an tensor of dtype uint32 used for the counts.

    Returns:
      An tensor of dtype uint32 with the same shape as `count`.
    """
    key1, key2 = keypair
    if not dtype(key1) == dtype(key2) == dtype(count) == np.uint32:
        msg = "threefry_2x32 requires uint32 arguments, got {}"
        raise TypeError(msg.format([dtype(x) for x in [key1, key2, count]]))

    assert count.size % 2

    if odd_size:
        x = list(jnp.split(jnp.cat([count.ravel(), np.uint32([0])]), 2))
    else:
        x = list(jnp.split(count.ravel(), 2))

    x = threefry2x32(key1, key2, x[0], x[1])
    out = cat(x)
    assert out.dtype == np.uint32
    return reshape(out[:-1] if odd_size else out, count.shape)


def threefry_split(key, num: int):
    counts = iota(np.uint32, num * 2)
    return reshape(threefry_2x32(key, counts), (num, 2))


def _threefry_split_foldlike(key, num):
    k1, k2 = key
    counts1, counts2 = iota_2x32_shape((num,))
    bits1, bits2 = threefry2x32(k1, k2, counts1, counts2)
    return stack([bits1, bits2], axis=1)


def threefry_fold_in(key, data):
    assert not data.shape
    return _threefry_fold_in(key, np.uint32(data))


def _threefry_fold_in(key, data):
    return threefry_2x32(key, threefry_seed(data))


#


class PRNGKeyTensorImpl(PRNGKeyTensor):
    """An tensor of PRNG keys backed by an RNG implementation.

    This class lifts the definition of a PRNG, provided in the form of a
    ``PRNGImpl``, into an tensor-like pytree class. Instances of this
    class behave like an tensor whose base elements are keys, hiding the
    fact that keys are typically tensors (of ``uint32`` dtype) themselves.

    PRNGKeyTensors are also restricted relative to JAX tensors in that
    they do not expose arithmetic operations. They instead expose
    wrapper methods around the PRNG implementation functions (``split``,
    ``random_bits``, ``fold_in``).
    """

    impl: PRNGImpl
    _base_tensor: typing.Tensor

    def __init__(self, impl, key_data: Any):
        assert not isinstance(key_data, core.Tracor)
        _check_prng_key_data(impl, key_data)
        self.impl = impl
        self._base_tensor = key_data

    # TODO(frostig): rename to unsafe_base_tensor, or just offer base_tensor attr?
    def unsafe_raw_tensor(self):
        """Access the raw numerical tensor that carries underlying key data.

        Returns:
          A uint32 JAX tensor whose leading dimensions are ``self.shape``.
        """
        return self._base_tensor

    def block_until_ready(self):
        _ = self._base_tensor.block_until_ready()
        return self

    @property
    def shape(self):
        return base_arr_shape_to_keys_shape(self.impl, self._base_tensor.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return KeyTy(self.impl)

    _device = property(op.attrgetter("_base_tensor._device"))
    _committed = property(op.attrgetter("_base_tensor._committed"))

    @property
    def sharding(self):
        aval = keys_shaped_tensor(self.impl, self.shape)
        phys_sharding = self._base_tensor.sharding
        return KeyTyRules.logical_op_sharding(aval, phys_sharding)

    def _is_scalar(self):
        base_ndim = len(self.impl.key_shape)
        return self._base_tensor.ndim == base_ndim

    def __len__(self):
        if self._is_scalar():
            raise TypeError("len() of unsized object")
        return len(self._base_tensor)

    def __iter__(self) -> Iterator[PRNGKeyTensorImpl]:
        if self._is_scalar():
            raise TypeError("iteration over a 0-d key tensor")
        # TODO(frostig): we may want to avoid iteration by slicing because
        # a very common use of iteration is `k1, k2 = split(key)`, and
        # slicing/indexing may be trickier to track for linearity checking
        # purposes. Maybe we can:
        # * introduce an unpack primitive+traceable (also allow direct use)
        # * unpack upfront into shape[0] many keytensor slices
        # * return iter over these unpacked slices
        # Whatever we do, we'll want to do it by overriding
        # ShapedTensor._iter when the element type is KeyTy...
        return (PRNGKeyTensorImpl(self.impl, k) for k in iter(self._base_tensor))

    # TODO(frostig): are all of the stackable methods below (reshape,
    # concat, expand, expand_dims), and the stackable registration,
    # still needed? If, with some work, none are needed, then do we want
    # to remove stackables altogether? This may be the only application.

    # TODO(frostig): Remove? Overwritten below in particular
    def reshape(self, newshape, order=None) -> PRNGKeyTensorImpl:
        reshaped_base = jnp.reshape(self._base_tensor, (*newshape, -1), order=order)
        return PRNGKeyTensorImpl(self.impl, reshaped_base)

    def cat(self, key_arrs, axis, dtype=None) -> PRNGKeyTensorImpl:
        if dtype is not None:
            raise ValueError("dtype argument not supported for concatenating PRNGKeyTensor")
        axis = canonicalize_axis(axis, self.ndim)
        arrs = [self._base_tensor, *[k._base_tensor for k in key_arrs]]
        return PRNGKeyTensorImpl(self.impl, jnp.cat(arrs, axis))

    def expand(self, shape) -> PRNGKeyTensorImpl:
        if jnp.ndim(shape) == 0:
            shape = (shape,)
        new_shape = (*shape, *self.impl.key_shape)
        return PRNGKeyTensorImpl(self.impl, jnp.expand(self._base_tensor, new_shape))

    def expand_dims(self, dimensions: Sequence[int]) -> PRNGKeyTensorImpl:
        # follows lax.expand_dims, not jnp.expand_dims, so dimensions is a sequence
        ndim_out = self.ndim + len(set(dimensions))
        dimensions = [canonicalize_axis(d, ndim_out) for d in dimensions]
        return PRNGKeyTensorImpl(self.impl, lax.expand_dims(self._base_tensor, dimensions))

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.impl.tag}]" f" {{ {self._base_tensor} }}"

    def pprint(self):
        pp_keys = pp.text("shape = ") + pp.text(str(self.shape))
        pp_impl = pp.text("impl = ") + self.impl.pprint()
        return str(pp.group(pp.text("PRNGKeyTensor:") + pp.nest(2, pp.brk() + pp_keys + pp.brk() + pp_impl)))

    # Overwritten immediately below
    @property
    def T(self) -> PRNGKeyTensor:
        assert False

    def __getitem__(self, _) -> PRNGKeyTensor:
        assert False

    def ravel(self, *_, **__) -> PRNGKeyTensor:
        assert False

    def squeeze(self, *_, **__) -> PRNGKeyTensor:
        assert False

    def transpose(self, *_, **__) -> PRNGKeyTensor:
        assert False

    def take(self, *_, **__) -> PRNGKeyTensor:
        assert False

    def permute(self, *_, **__) -> PRNGKeyTensor:
        assert False

    def flatten(self, *_, **__) -> PRNGKeyTensor:
        assert False


_set_device_tensor_base_attributes(
    PRNGKeyTensorImpl,
    include=[
        "__getitem__",
        "ravel",
        "squeeze",
        "transpose",
        "take",
        "reshape",
        "permute",
        "flatten",
        "T",
    ],
)
_register_stackable(PRNGKeyTensorImpl)
basetensor.Tensor.register(PRNGKeyTensorImpl)


# TODO(frostig): remove, rerouting callers directly to random_seed
def seed_with_impl(impl: PRNGImpl, seed: Union[int, Tensor]) -> PRNGKeyTensorImpl:
    return random_seed(seed, impl=impl)


def keys_shaped_tensor(impl, shape):
    return core.ShapedTensor(shape, KeyTy(impl))


def keys_aval_to_base_arr_aval(keys_aval):
    shape = (*keys_aval.shape, *keys_aval.dtype.impl.key_shape)
    return core.ShapedTensor(shape, np.dtype("uint32"))


def base_arr_shape_to_keys_shape(impl, base_arr_shape):
    base_ndim = len(impl.key_shape)
    return base_arr_shape[:-base_ndim]


def make_key_tensor_phys_sharding(aval, sharding, is_sharding_from_xla):
    if dispatch.is_single_device_sharding(sharding):
        return sharding
    elif isinstance(sharding, PmapSharding):
        key_shape = aval.dtype.impl.key_shape
        trailing_sharding = [sharding_specs.NoSharding()] * len(key_shape)
        phys_sharding_spec = sharding_specs.ShardingSpec(
            sharding=(*sharding.sharding_spec.sharding, *trailing_sharding),
            mesh_mapping=sharding.sharding_spec.mesh_mapping,
        )
        return PmapSharding(devices=sharding.devices, sharding_spec=phys_sharding_spec)
    elif isinstance(sharding, NamedSharding):
        key_shape = aval.dtype.impl.key_shape
        trailing_spec = [None] * len(key_shape)
        return NamedSharding(sharding.mesh, PartitionSpec(*sharding.spec, *trailing_spec))
    elif is_sharding_from_xla:
        return sharding
    else:
        sharding_proto = sharding._to_xla_op_sharding(aval.ndim)
        return GSPMDSharding(
            sharding._device_assignment,
            KeyTyRules.physical_op_sharding(aval, sharding_proto),
        )


##############


def philox(cls, key, rounds=10):
    def rotl(x, r):
        return (x << r) | (x >> (64 - r))

    def philox_round(a, b, key):
        a += b
        b = rotl(b, 32)
        a ^= key
        return a, b

    assert len(key) == 2, "Key must be a tuple of two 64-bit integers"
    assert isinstance(rounds, int) and rounds > 0, "Number of rounds must be a positive integer"

    key0, key1 = key
    state = np.tensor([key0, key1], dtype=np.uint64)

    for _ in range(rounds):
        state[0], state[1] = philox_round(state[0], state[1], _)
        state[0] += 1
    return state[:2], state  #  [0, 1] are new keys, [:] are random numbers,


def threefry(cls, key, rounds=20):
    def rotl(x, r):
        return (x << r) | (x >> (64 - r))

    def threefry_round(a, b, c, d, r):
        a += b
        d ^= a
        d = rotl(d, r)
        c += d
        b ^= c
        b = rotl(b, r)
        a += b
        d ^= a
        d = rotl(d, r)
        c += d
        b ^= c
        b = rotl(b, r)
        return a, b, c, d

    assert len(key) == 2, "Key must be a tuple of two 64-bit integers"
    assert isinstance(rounds, int) and rounds > 0, "Number of rounds must be a positive integer"

    key0, key1 = key
    state = np.tensor([0, 0, key0, key1], dtype=np.uint64)

    for _ in range(rounds):
        state[0], state[1], state[2], state[3] = threefry_round(state[0], state[1], state[2], state[3], 14)
        state[0], state[1], state[2], state[3] = threefry_round(state[0], state[1], state[2], state[3], 16)
        state[0], state[1], state[2], state[3] = threefry_round(state[0], state[1], state[2], state[3], 52)
        state[0], state[1], state[2], state[3] = threefry_round(state[0], state[1], state[2], state[3], 57)

    return state[:2], state[2:]  # [0, 1] are new keys, [2:] are random numbers


def rng_bit(cls, key, algorithm="philox"):
    return dict(philox=cls.philox, threefry=cls.threefry)[algorithm](key)


def random_normal(cls, x, key, dtype):
    # Box-Muller transform
    nbits = dtype.itemsize * 8
    u1 = 0
    while u1 == 0:
        u1, key = cls.rng_bit(x, key)
        u1 = u1 / (2**nbits)  # normalize to [0, 1]
    u2, key = cls.rng_bit(x, key)
    u2 = u2 / (2**nbits)
    z0 = (-2.0 * u1.log()).sqrt() * (2 * math.pi * u2).cos()
    return z0, key
