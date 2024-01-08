import slope

# @slope.jit.with_options(static_argnames="p")
@slope.jit
def f(x):
    # return x + 1.
    return x + g(x) + slope.tensor([1.0])

@slope.jit
def g(x):
    return x * 2

x = slope.ones(1)
# y = f(x, p=1.)
jit_object = f.get_jit_object(x)
print(jit_object.program)
print(jit_object.program.num_consts)
print(jit_object.consts)
# jit_object.program.save(x, dir_path="/tmp/slope_program", dry_run=True)
jit_object.program.save(x, dir_path="/tmp/slope_program")