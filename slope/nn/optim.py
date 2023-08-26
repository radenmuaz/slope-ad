import slope as sp
from typing import Any, Callable, NamedTuple, Tuple, Union
import numpy as np
from collections import namedtuple
import functools
from functools import partial
from slope.core import unzip2, list_map

OptimizerState = namedtuple(
    "OptimizerState", ["packed_state", "tree_def", "subtree_defs"]
)
sp.rt.register_pytree_node(
    OptimizerState,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs: OptimizerState(xs[0], data[0], data[1]),
)


Array = Any
Params = Any  # Parameters are arbitrary nests of `jnp.ndarrays`.
State = Any  # internal State
Updates = Params  # Gradient updates are of the same type as parameters.

InitFn = Callable[[Params], OptimizerState]
Step = int
UpdateFn = Callable[[Step, Updates, OptimizerState], OptimizerState]
ParamsFn = Callable[[OptimizerState], Params]


class Optimizer(NamedTuple):
    init_fn: InitFn
    update_fn: UpdateFn
    params_fn: ParamsFn


Schedule = Callable[[Step], float]


def optimizer(
    opt_maker: Callable[
        ...,
        Tuple[
            Callable[[Params], State],
            Callable[[Step, Updates, Params], Params],
            Callable[[State], Params],
        ],
    ]
) -> Callable[..., Optimizer]:
    @functools.wraps(opt_maker)
    def tree_opt_maker(*args, **kwargs):
        init, update, get_params = opt_maker(*args, **kwargs)

        @functools.wraps(init)
        def tree_init(x0_tree):
            x0_flat, tree = sp.rt.tree_flatten(x0_tree)
            initial_states = [init(x0) for x0 in x0_flat]
            states_flat, subtrees = unzip2(list_map(sp.rt.tree_flatten, initial_states))
            return OptimizerState(states_flat, tree, subtrees)

        @functools.wraps(update)
        def tree_update(i, grad_tree, opt_state):
            states_flat, tree, subtrees = opt_state
            grad_flat, tree2 = sp.rt.tree_flatten(grad_tree)
            if tree2 != tree:
                msg = (
                    "optimizer update function was passed a gradient tree that did "
                    "not match the parameter tree structure with which it was "
                    "initialized: parameter tree {} and grad tree {}."
                )
                raise TypeError(msg.format(tree, tree2))
            states = list_map(sp.rt.tree_unflatten, subtrees, states_flat)
            new_states = list_map(partial(update, i), grad_flat, states)
            new_states_flat, subtrees2 = unzip2(
                list_map(sp.rt.tree_flatten, new_states)
            )
            for subtree, subtree2 in zip(subtrees, subtrees2):
                if subtree2 != subtree:
                    msg = (
                        "optimizer update function produced an output structure that "
                        "did not match its input structure: input {} and output {}."
                    )
                    raise TypeError(msg.format(subtree, subtree2))
            return OptimizerState(new_states_flat, tree, subtrees)

        @functools.wraps(get_params)
        def tree_get_params(opt_state):
            states_flat, tree, subtrees = opt_state
            states = list_map(sp.rt.tree_unflatten, subtrees, states_flat)
            params = list_map(get_params, states)
            return sp.rt.tree_unflatten(tree, params)

        return Optimizer(tree_init, tree_update, tree_get_params)

    return tree_opt_maker


### optimizers


@optimizer
def sgd(step_size):
    """Construct optimizer triple for stochastic gradient descent.

    Args:
      step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to a positive scalar.

    Returns:
      An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)

    def init(x0):
        return x0

    def update(i, g, x):
        return x - step_size(i) * g

    def get_params(x):
        return x

    return Optimizer(init, update, get_params)


@optimizer
def sgd_momentum(step_size: Schedule, mass: float):
    step_size = make_schedule(step_size)

    def init(x0):
        v0 = sp.rt.procs.zeros_like(x0)
        return x0, v0

    def update(i, g, state):
        x, velocity = state
        velocity = mass * velocity + g
        x = x - step_size(i) * velocity
        return x, velocity

    def get_params(state):
        x, _ = state
        return x

    return init, update, get_params


@optimizer
def nesterov(step_size: Schedule, mass: float):
    step_size = make_schedule(step_size)

    def init(x0):
        v0 = np.zeros_like(x0)
        return x0, v0

    def update(i, g, state):
        x, velocity = state
        velocity = mass * velocity + g
        x = x - step_size(i) * (mass * velocity + g)
        return x, velocity

    def get_params(state):
        x, _ = state
        return x

    return init, update, get_params


@optimizer
def adam(step_size, b1=0.9, b2=0.999, eps=1e-2):
    """Construct optimizer triple for Adam.

    Args:
      step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to a positive scalar.
      b1: optional, a positive scalar value for beta_1, the exponential decay rate
        for the first moment estimates (default 0.9).
      b2: optional, a positive scalar value for beta_2, the exponential decay rate
        for the second moment estimates (default 0.999).
      eps: optional, a positive scalar value for epsilon, a small constant for
        numerical stability (default 1e-8).

    Returns:
      An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)

    def init(x0):
        m0 = sp.rt.procs.zeros_like(x0)
        v0 = sp.rt.procs.zeros_like(x0)
        return x0, m0, v0

    def update(i, g, state):
        x, m, v = state
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * g * g + b2 * v  # Second moment estimate.
        mhat = m / (1 - (b1 ** (i + 1)))  # Bias correction.
        vhat = v / (1 - (b2 ** (i + 1)))
        x = x - step_size(i) * mhat / (vhat.sqrt() + eps)
        # print(mhat, vhat, x)
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params


### learning rate schedules


def constant(step_size) -> Schedule:
    def schedule(i):
        return step_size

    return schedule


def exponential_decay(step_size, decay_steps, decay_rate):
    def schedule(i):
        return step_size * decay_rate ** (i / decay_steps)

    return schedule


def inverse_time_decay(step_size, decay_steps, decay_rate, staircase=False):
    if staircase:

        def schedule(i):
            return step_size / (1 + decay_rate * jnp.floor(i / decay_steps))

    else:

        def schedule(i):
            return step_size / (1 + decay_rate * i / decay_steps)

    return schedule


def polynomial_decay(step_size, decay_steps, final_step_size, power=1.0):
    def schedule(step_num):
        step_num = sp.rt.procs.minimum(step_num, decay_steps)
        step_mult = (1 - step_num / decay_steps) ** power
        return step_mult * (step_size - final_step_size) + final_step_size

    return schedule


def piecewise_constant(boundaries: Any, values: Any):
    boundaries = np.array(boundaries)
    values = np.array(values)
    if not boundaries.ndim == values.ndim == 1:
        raise ValueError("boundaries and values must be sequences")
    if not boundaries.shape[0] == values.shape[0] - 1:
        raise ValueError("boundaries length must be one shorter than values length")

    def schedule(i):
        return values[sp.rt.ops.sum(i > boundaries)]

    return schedule


def make_schedule(scalar_or_schedule: Union[float, Schedule]) -> Schedule:
    if callable(scalar_or_schedule):
        return scalar_or_schedule
    elif np.ndim(scalar_or_schedule) == 0:
        return constant(scalar_or_schedule)
    else:
        raise TypeError(type(scalar_or_schedule))


### serialization utilities


class JoinPoint:
    """Marks the boundary between two joined (nested) pytrees."""

    def __init__(self, subtree):
        self.subtree = subtree

    # Since pytrees are containers of numpy arrays, look iterable.
    def __iter__(self):
        yield self.subtree


def unpack_optimizer_state(opt_state):
    """Converts an OptimizerState to a marked pytree.

    Converts an OptimizerState to a marked pytree with the leaves of the outer
    pytree represented as JoinPoints to avoid losing information. This function is
    intended to be useful when serializing optimizer states.

    Args:
      opt_state: An OptimizerState
    Returns:
      A pytree with JoinPoint leaves that contain a second level of pytrees.
    """
    states_flat, tree_def, subtree_defs = opt_state
    subtrees = map(sp.tree_unflatten, subtree_defs, states_flat)
    sentinels = [JoinPoint(subtree) for subtree in subtrees]
    return sp.tree_unflatten(tree_def, sentinels)


def pack_optimizer_state(marked_pytree):
    """Converts a marked pytree to an OptimizerState.

    The inverse of unpack_optimizer_state. Converts a marked pytree with the
    leaves of the outer pytree represented as JoinPoints back into an
    OptimizerState. This function is intended to be useful when deserializing
    optimizer states.

    Args:
      marked_pytree: A pytree containing JoinPoint leaves that hold more pytrees.
    Returns:
      An equivalent OptimizerState to the input argument.
    """
    sentinels, tree_def = sp.tree_flatten(marked_pytree)
    assert all(isinstance(s, JoinPoint) for s in sentinels)
    subtrees = [s.subtree for s in sentinels]
    states_flat, subtree_defs = sp.unzip2(map(sp.tree_flatten, subtrees))
    return OptimizerState(states_flat, tree_def, subtree_defs)
