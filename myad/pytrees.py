from typing import (
    Callable,
    NamedTuple,
    Dict,
    Type,
    Hashable,
    Tuple,
    List,
    Any,
    Iterable,
    Iterator,
)
import myad
from myad import utils


class Empty:
    pass


empty = Empty()

import itertools


class Store:
    val = empty

    def set_value(self, val):
        assert self.val is empty
        self.val = val

    def __call__(self):
        return self.val


class NodeType(NamedTuple):
    name: str
    to_iterable: Callable
    from_iterable: Callable


class PyTreeDef(NamedTuple):
    node_type: NodeType
    node_metadata: Hashable
    child_treedefs: Tuple["PyTreeDef", ...]

    class Leaf:
        pass

    leaf = Leaf()


def tree_flatten(x: Any) -> Tuple[List[Any], Any]:
    def _tree_flatten(x: Any) -> Tuple[Iterable, PyTreeDef]:
        node_type = myad.RT.node_types.get(type(x))
        if node_type:
            node_metadata, children = node_type.to_iterable(x)
            children_flat, child_trees = utils.unzip2(map(_tree_flatten, children))
            flattened = itertools.chain.from_iterable(children_flat)
            return flattened, PyTreeDef(node_type, node_metadata, tuple(child_trees))
        else:
            return [x], PyTreeDef.leaf

    children_iter, treedef = _tree_flatten(x)
    return list(children_iter), treedef


def tree_unflatten(treedef: PyTreeDef, xs: List[Any]) -> Any:
    def _tree_unflatten(treedef: PyTreeDef, xs: Iterator) -> Any:
        if treedef is PyTreeDef.leaf:
            return next(xs)
        else:
            children = (_tree_unflatten(t, xs) for t in treedef.child_treedefs)
            return treedef.node_type.from_iterable(treedef.node_metadata, children)

    return _tree_unflatten(treedef, iter(xs))


def flatten_fun(f, in_tree):
    store = Store()

    def flat_fun(*args_flat):
        pytree_args = tree_unflatten(in_tree, args_flat)
        out = f(*pytree_args)
        out_flat, out_tree = tree_flatten(out)
        store.set_value(out_tree)
        return out_flat

    return flat_fun, store
