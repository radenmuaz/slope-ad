import slope
from slope import utils
from functools import partial, lru_cache
from typing import Tuple
from slope.array import Array


class BaseOpImpl:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
    def ir(self, *args, **kwargs):
        raise NotImplementedError

class BaseBackend:
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
        in_avals = [v.aval for v in prog.in_binders[len(consts):]]
        compiled = cls.compile(prog, consts, in_avals, name="numpy_fn")
        return compiled
    
    @classmethod
    def compile(cls, prog, consts, in_avals, name: str):
        raise NotImplementedError
    
    convert_impl = BaseOpImpl()
    neg_impl = BaseOpImpl()
    exp_impl = BaseOpImpl()
    log_impl = BaseOpImpl()
    add_impl = BaseOpImpl()
    sub_impl = BaseOpImpl()
    mul_impl = BaseOpImpl()
    div_impl = BaseOpImpl()
    equal_impl = BaseOpImpl()
    not_equal_impl = BaseOpImpl()
    maximum_impl = BaseOpImpl()
    max_impl = BaseOpImpl()
    sum_impl = BaseOpImpl()
    choose_impl = BaseOpImpl()
    where_impl = BaseOpImpl()
    
    convert = classmethod(lambda cls, arr, dtype: cls.convert_impl(arr, dtype=dtype))
    astype = convert
    neg = classmethod(lambda cls, arr: cls.neg_impl(arr))
    exp = classmethod(lambda cls, arr: cls.exp_impl(arr))
    log = classmethod(lambda cls, arr: cls.log_impl(arr))
    add = classmethod(lambda cls, arr, other: cls.add_impl(arr, other))
    sub = classmethod(lambda cls, arr, other: cls.sub_impl(arr, other))
    mul =  classmethod(lambda cls, arr, other: cls.mul_impl(arr, other))
    div =  classmethod(lambda cls, arr, other: cls.div_impl(arr, other))
    equal = classmethod(lambda cls, arr, other: cls.equal_impl(arr, other))
    not_equal =  classmethod(lambda cls, arr, other: cls.not_equal_impl(arr, other))
    maximum = classmethod(lambda cls, arr, other: cls.maximum_impl(arr, other))

# class BaseBuilder:
#     pass

# class BaseConst:
#     pass


# class BaseParam:
#     pass

