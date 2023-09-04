from slope.envs.v1 import v1_env
from slope import core
import weakref

machine = core.Machine(env=v1_env)
M = weakref.ref(machine)() # to avoid circular import

# shortcuts
jvp = M.jvp
vmap = M.vmap
jit = M.jit
linearize = M.linearize
vjp = M.vjp
grad = M.grad
register_pytree_node = M.register_pytree_node
tree_flatten = M.tree_flatten
tree_unflatten = M.tree_unflatten

env = M.env
numpy = env
