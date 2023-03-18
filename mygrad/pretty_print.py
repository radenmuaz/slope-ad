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


