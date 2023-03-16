from typing import NamedTuple
from typing import Type, List, Tuple, Sequence, Optional, Any, DefaultDict
from typing import Callable, Type, Hashable, Dict, Iterable, Iterator
from typing import Union
from typing import Set
import string


class PPrint:
    lines: List[Tuple[int, str]]

    def __init__(self, lines):
        self.lines = lines

    def indent(self, indent: int) -> "PPrint":
        return PPrint([(indent + orig_indent, s) for orig_indent, s in self.lines])

    def __add__(self, rhs: "PPrint") -> "PPrint":
        return PPrint(self.lines + rhs.lines)

    def __rshift__(self, rhs: "PPrint") -> "PPrint":
        if not rhs.lines:
            return self
        if not self.lines:
            return rhs
        indent, s = self.lines[-1]
        indented_block = rhs.indent(indent + len(s))
        common_line = s + " " * rhs.lines[0][0] + rhs.lines[0][1]
        return PPrint(
            self.lines[:-1] + [(indent, common_line)] + indented_block.lines[1:]
        )

    def __str__(self) -> str:
        return "\n".join(" " * indent + s for indent, s in self.lines)


def pp(s: Any) -> PPrint:
    return PPrint([(0, line) for line in str(s).splitlines()])


def vcat(ps: List[PPrint]) -> PPrint:
    return sum(ps, pp(""))


# def var_str(names: DefaultDict[Var, str], v: Var) -> str:
#     return f'{names[v]}:{v.aval.str_short()}'

# def pp_eqn(names: DefaultDict[Var, str], eqn: JaxprEqn) -> PPrint:
#     rule = pp_rules.get(eqn.LLOp)
#     if rule:
#         return rule(names, eqn)
#     else:
#         lhs = pp(' '.join(var_str(names, v) for v in eqn.out_binders))
#         rhs = (pp(eqn.LLOp.name) >> pp_params(eqn.params) >>
#            pp(' '.join(names[x] if isinstance(x, Var) else str(x.val)
#                        for x in eqn.inputs)))
#         return lhs >> pp(' = ') >> rhs

# def pp_params(params: Dict[str, Any]) -> PPrint:
#     items = sorted(params.items())
#     if items:
#         return pp(' [ ') >> vcat([pp(f'{k}={v}') for k, v in items]) >> pp(' ] ')
#     else:
#         return pp(' ')
