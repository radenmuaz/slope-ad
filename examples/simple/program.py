import slope

@slope.jit
def f(x):
    return x + g(x)


@slope.jit
def g(x):
    return x * 2

x = slope.ones(1)
y = f(x)
program = f.get_jit_object(x).program
print(program)