import slope
from slope import utils
from functools import partial, lru_cache
from typing import Tuple
from slope.array import Array

class BaseBackend:

    class BaseOpImpl:
        @classmethod
        def do(cls, *args, **kwargs):
            raise NotImplementedError

        @classmethod
        def ir(cls, *args, **kwargs):
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
        compiled = cls.compile(prog, consts, in_avals, name="numpy_fn")
        return compiled

    @classmethod
    def compile(cls, prog, consts, in_avals, name: str):
        raise NotImplementedError
    
    AddImpl = BaseOpImpl
    SubImpl = BaseOpImpl
    MulImpl = BaseOpImpl
    DivImpl = BaseOpImpl
    ExpImpl = BaseOpImpl
    LogImpl = BaseOpImpl
    FullImpl = BaseOpImpl
    ReshapeImpl = BaseOpImpl

    convert = classmethod(lambda cls, arr, dtype: cls.ConvertImpl(arr, dtype=dtype))
    astype = convert
    neg = classmethod(lambda cls, arr: cls.NegImpl.do(arr))
    exp = classmethod(lambda cls, arr: cls.ExpImpl.do(arr))
    log = classmethod(lambda cls, arr: cls.LogImpl.do(arr))
    add = classmethod(lambda cls, arr, other: cls.AddImpl.do(arr, other))
    sub = classmethod(lambda cls, arr, other: cls.SubImpl.do(arr, other))
    mul = classmethod(lambda cls, arr, other: cls.MulImpl.do(arr, other))
    div = classmethod(lambda cls, arr, other: cls.DivImpl.do(arr, other))
    equal = classmethod(lambda cls, arr, other: cls.EqualImpl.do(arr, other))
    not_equal = classmethod(lambda cls, arr, other: cls.NotEqualImpl.do(arr, other))
    maximum = classmethod(lambda cls, arr, other: cls.MaximumImpl.do(arr, other))


# class BaseBuilder:
#     pass

# class BaseConst:
#     pass


# class BaseParam:
#     pass
