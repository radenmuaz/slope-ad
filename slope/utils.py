from __future__ import annotations

from typing import (
    Tuple,
    List,
    Any,
)


class IDHashable:
    val: Any

    def __init__(self, val):
        self.val = val

    def __hash__(self) -> int:
        return id(self.val)

    def __eq__(self, other):
        return type(other) is IDHashable and id(self.val) == id(other.val)


def swap(f):
    return lambda x, y: f(y, x)


def unzip2(pairs):
    lst1, lst2 = [], []
    for x1, x2 in pairs:
        lst1.append(x1)
        lst2.append(x2)
    return lst1, lst2


def list_map(f: Any, *xs: Any) -> Any:
    return list(map(f, *xs))


def list_zip(*args: Any) -> Any:
    fst, *rest = args = list_map(list, args)
    n = len(fst)
    for arg in rest:
        assert len(arg) == n
    return list(zip(*args))


def split_half(lst: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert not len(lst) % 2
    return split_list(lst, len(lst) // 2)


def merge_lists(which: List[bool], l1: List[Any], l2: List[Any]) -> List[Any]:
    l1, l2 = iter(l1), iter(l2)
    out = [next(l2) if b else next(l1) for b in which]
    assert next(l1, None) is next(l2, None) is None
    return out


def split_list(lst: List[Any], n: int) -> Tuple[List[Any], List[Any]]:
    assert 0 <= n <= len(lst)
    return lst[:n], lst[n:]


def partition_list(bs: List[bool], l: List[Any]) -> Tuple[List[Any], List[Any]]:
    assert len(bs) == len(l)
    lst1: List[Any] = []
    lst2: List[Any] = []
    lists = lst1, lst2
    # lists = lst1: List[Any], lst2: List[Any] = list(), list()
    for b, x in list_zip(bs, l):
        lists[b].append(x)
    return lst1, lst2


from dataclasses import dataclass, asdict
import os, math, functools
import numpy as np
from typing import (
    Tuple,
    Union,
    List,
    NamedTuple,
    Final,
    Iterator,
    ClassVar,
    Optional,
    Callable,
    Any,
)

ShapeType = Tuple[int, ...]
# NOTE: helpers is not allowed to import from anything else in tinygrad


def dedup(x):
    return list(dict.fromkeys(x))  # retains list order


def prod(x: Union[List[int], Tuple[int, ...]]) -> int:
    return math.prod(x)


def argfix(*x):
    return (
        tuple()
        if len(x) == 0
        else tuple(x[0])
        if isinstance(x[0], (tuple, list))
        else tuple(x)
    )


def argsort(x):
    return type(x)(
        sorted(range(len(x)), key=x.__getitem__)
    )  # https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python


def all_same(items):
    return all(x == items[0] for x in items) if len(items) > 0 else True


def colored(st, color, background=False, bright=False):
    return (
        f"\u001b[{10*background+60*bright+30+['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'].index(color)}m{st}\u001b[0m"
        if color is not None
        else st
    )  # replace the termcolor library with one line


def partition(lst, fxn):
    return [x for x in lst if fxn(x)], [x for x in lst if not fxn(x)]


def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
    return (x,) * cnt if isinstance(x, int) else x


def flatten(l: Iterator):
    return [item for sublist in l for item in sublist]


def mnum(i) -> str:
    return str(i) if i >= 0 else f"m{-i}"


@functools.lru_cache(maxsize=None)
def getenv(key, default=0):
    return type(default)(os.getenv(key, default))

