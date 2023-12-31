import slope

def f(x):
    return x + 1

x = slope.ones(3,1)
y = slope.vmap(f)(x)

print(x.shape, y.shape)