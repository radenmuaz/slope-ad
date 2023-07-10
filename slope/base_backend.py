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

    NegImpl = BaseOpImpl
    ExpImpl = BaseOpImpl
    LogImpl = BaseOpImpl
    ConvertImpl = BaseOpImpl

    AddImpl = BaseOpImpl
    SubImpl = BaseOpImpl
    MulImpl = BaseOpImpl
    DivImpl = BaseOpImpl
    MaximumImpl = BaseOpImpl
    EqualImpl = BaseOpImpl
    NotEqualImpl = BaseOpImpl

    MaxImpl = BaseOpImpl
    SumImpl = BaseOpImpl

    FullImpl = BaseOpImpl
    ReshapeImpl = BaseOpImpl
    TransposeImpl = BaseOpImpl
    BroadcastImpl = BaseOpImpl
    GatherImpl = BaseOpImpl
    ScatterImpl = BaseOpImpl

    convert = classmethod(lambda cls, x, dtype: cls.ConvertImpl(x, dtype=dtype))
    astype = convert
    neg = classmethod(lambda cls, x: cls.NegImpl.do(x))
    exp = classmethod(lambda cls, x: cls.ExpImpl.do(x))
    log = classmethod(lambda cls, x: cls.LogImpl.do(x))
    add = classmethod(lambda cls, x, other: cls.AddImpl.do(x, other))
    sub = classmethod(lambda cls, x, other: cls.SubImpl.do(x, other))
    mul = classmethod(lambda cls, x, other: cls.MulImpl.do(x, other))
    div = classmethod(lambda cls, x, other: cls.DivImpl.do(x, other))
    equal = classmethod(lambda cls, x, other: cls.EqualImpl.do(x, other))
    not_equal = classmethod(lambda cls, x, other: cls.NotEqualImpl.do(x, other))
    maximum = classmethod(lambda cls, x, other: cls.MaximumImpl.do(x, other))
    max = classmethod(lambda cls, x, other: cls.MaxImpl.do(x, other))
    sum = classmethod(lambda cls, x, other: cls.SumImpl.do(x, other))
    full = classmethod(lambda cls, x, other: cls.FullImpl.do(x, other))
    broadcast = classmethod(
        lambda cls, x, shape, axes: cls.BroadcastImpl.do(x, shape=shape, axes=axes)
    )


# class BaseBuilder:
#     pass

# class BaseConst:
#     pass


# class BaseParam:
#     pass
