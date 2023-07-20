import slope
from slope import utils
from functools import partial, lru_cache
from typing import Tuple


class BaseBackend:
    class BaseOpImpl:
        @classmethod
        def do(cls, *args, **kwargs):
            raise NotImplementedError

        @classmethod
        def ir(cls, *args, **kwargs):
            raise NotImplementedError

    class BaseJitFn:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError

        def __call__(self, *args, **kwargs):
            raise NotImplementedError

    @classmethod
    @lru_cache()
    def callable(
        cls,
        hashable_prog: utils.IDHashable,
        hashable_consts: Tuple[utils.IDHashable, ...],
    ):
        prog: slope.ad.Prog = hashable_prog.val
        slope.ad.typecheck_prog(prog)
        consts = [x.val for x in hashable_consts]
        in_avals = [v.aval for v in prog.in_binders[len(consts) :]]
        compiled = cls.compile(
            prog, consts, in_avals, name=f"{cls.__name__.lower()}_fn"
        )
        return compiled

    @classmethod
    def compile(cls, prog, consts, in_avals, name: str):
        raise NotImplementedError

    ConvertImpl = BaseOpImpl
    StopGradientImpl = BaseOpImpl
    NegImpl = BaseOpImpl
    ExpImpl = BaseOpImpl
    LogImpl = BaseOpImpl
    SqrtImpl = BaseOpImpl

    AddImpl = BaseOpImpl
    SubImpl = BaseOpImpl
    MulImpl = BaseOpImpl
    DivImpl = BaseOpImpl
    MaximumImpl = BaseOpImpl
    EqualImpl = BaseOpImpl
    NotEqualImpl = BaseOpImpl

    MaxImpl = BaseOpImpl
    SumImpl = BaseOpImpl

    ConstantImpl = BaseOpImpl
    FullImpl = BaseOpImpl
    ArangeImpl = BaseOpImpl
    RandomUniformImpl = BaseOpImpl
    RandomNormalImpl = BaseOpImpl

    ReshapeImpl = BaseOpImpl
    TransposeImpl = BaseOpImpl
    BroadcastImpl = BaseOpImpl
    FlipImpl = BaseOpImpl
    PadImpl = BaseOpImpl
    ConcatenateImpl = BaseOpImpl
    SliceImpl = BaseOpImpl
    GatherImpl = BaseOpImpl
    ScatterImpl = BaseOpImpl

    
    stop_gradient = classmethod(lambda cls, x: cls.StopGradientImpl.do(x))
    convert = classmethod(lambda cls, x, dtype: cls.ConvertImpl.do(x, dtype=dtype))
    astype = convert
    neg = classmethod(lambda cls, x: cls.NegImpl.do(x))
    sqrt = classmethod(lambda cls, x: cls.SqrtImpl.do(x))
    exp = classmethod(lambda cls, x: cls.ExpImpl.do(x))
    log = classmethod(lambda cls, x: cls.LogImpl.do(x))

    add = classmethod(lambda cls, x, other: cls.AddImpl.do(x, other))
    sub = classmethod(lambda cls, x, other: cls.SubImpl.do(x, other))
    mul = classmethod(lambda cls, x, other: cls.MulImpl.do(x, other))
    div = classmethod(lambda cls, x, other: cls.DivImpl.do(x, other))
    maximum = classmethod(lambda cls, x, other: cls.MaximumImpl.do(x, other))
    equal = classmethod(lambda cls, x, other: cls.EqualImpl.do(x, other))
    not_equal = classmethod(lambda cls, x, other: cls.NotEqualImpl.do(x, other))

    max = classmethod(lambda cls, x, other: cls.MaxImpl.do(x, other))
    sum = classmethod(lambda cls, x, other: cls.SumImpl.do(x, other))

    constant = classmethod(
        lambda cls, val, dtype: cls.ConstantImpl.do(val=val, dtype=dtype)
    )
    full = classmethod(
        lambda cls, val, dtype: cls.FullImpl.do(val=val, dtype=dtype)
    )
    arange = classmethod(lambda cls, start, stop, stride, dtype: cls.ArangeImpl.do(start, stop, stride, dtype))
    random_uniform = classmethod(lambda cls, x, other: cls.RandomUniform.do(x, other))
    rand = random_uniform
    random_normal = classmethod(lambda cls, x, other: cls.RandomNormal.do(x, other))
    randn = random_normal
    broadcast = classmethod(
        lambda cls, x, shape, axes: cls.BroadcastImpl.do(x, shape=shape, axes=axes)
    )
    reshape = classmethod(lambda cls, x, shape: cls.ReshapeImpl.do(x, shape=shape))
    transpose = classmethod(lambda cls, x, axes: cls.TransposeImpl.do(x, axes=axes))
    flip = classmethod(lambda cls, x, axes: cls.FlipImpl.do(x, axes=axes))
    pad = classmethod(lambda cls, x, lo, hi, interior, value: cls.PadImpl.do(x, lo, hi, interior, value))
    slice = classmethod(lambda cls, x, axes: cls.SliceImpl.do(x, axes=axes))
    concatenate = classmethod(lambda cls, x, axes: cls.ConcatenateImpl.do(x, axes=axes))
    gather = classmethod(lambda cls, x, axes: cls.GatherImpl.do(x, axes=axes))
    scatter = classmethod(lambda cls, x, axes: cls.ScatterImpl.do(x, axes=axes))


# class BaseBuilder:
#     pass

# class BaseConst:
#     pass


# class BaseParam:
#     pass
