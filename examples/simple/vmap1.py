import slope

def f(x):
    return x + x.ones_like()

x = slope.ones(3,1)
# y = (f)(x)
y = slope.vmap(f)(x)
print(y, x.shape, y.shape)