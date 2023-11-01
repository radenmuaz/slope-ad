import slope

def f(x):
    out = x
    out = out.pad(((1, 0),))
    out = out.sum()
    return out


x = slope.ones(3)
x_dot = slope.ones(3)
print(slope.grad(f)(x))
