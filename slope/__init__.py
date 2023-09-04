from slope.envs.v1 import v1_env
from slope import core

machine = core.Machine(env=v1_env)

jvp = machine.jvp
vmap = machine.vmap
jit = machine.jit
linearize = machine.linearize
vjp = machine.vjp
grad = machine.grad

register_pytree_node = machine.register_pytree_node
tree_flatten = machine.tree_flatten
tree_unflatten = machine.tree_unflatten

env = machine.env
numpy = env
