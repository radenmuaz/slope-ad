from slope.systems.v1 import v1_system
from slope import core

machine = core.Machine(system=v1_system)

jvp = machine.jvp
vmap = machine.vmap
jit = machine.jit
linearize = machine.linearize
vjp = machine.vjp
grad = machine.grad

register_pytree_node = machine.register_pytree_node
tree_flatten = machine.tree_flatten
tree_unflatten = machine.tree_unflatten

system = machine.system
numpy = system
