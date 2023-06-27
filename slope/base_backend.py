import slope
from slope import utils
from functools import partial, lru_cache
from typing import Tuple


class BaseBuilder:
    pass


class BaseBackend:
    Builder: BaseBuilder = None

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
        c = cls.Builder("numpy_call")
        backend_prog = c.prog_subcomp(
            c, prog, c.get_consts(consts) + c.get_params(in_avals)
        )
        compiled = c.compile(backend_prog)
        return partial(c.execute_compiled, compiled, [v.aval for v in prog.outs])


class BaseConst:
    pass


class BaseParam:
    pass


class BaseOp:
    pass
