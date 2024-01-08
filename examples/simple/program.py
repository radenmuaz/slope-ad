import slope

@slope.jit.with_options(static_argnames="p")
def f(x, p):
    return x + g(x) + x.full_like(p)

@slope.jit
def g(x):
    return x * 2

x = slope.ones(1)
y = f(x, p=1.)
program = f.get_jit_object(x, p=1.).program
print(program)