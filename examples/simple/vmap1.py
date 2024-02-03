import slope


def f(x):
    # return x + 1
    return x + x.ones_like()


x = slope.ones(3, 1)
# y = (f)(x)
y = slope.vmap(f)(x)
print(y)
print(x.shape, y.shape)


# pad
def f(x):
    return x.pad((1, 2))


x = slope.ones(1, 3)
# y = f(x)
y = slope.vmap(f)(x)
print(y)
print(x.shape, y.shape)

slice


def f(x):
    return x.slice((0,), (2,))


x = slope.ones(1, 5)
# y = f(x)
y = slope.vmap(f)(x)
print(y)
print(x.shape, y.shape)


# cat
def f(x1, x2, x3):
    return slope.cat((x1, x2, x3), 0)


x1 = slope.ones(1, 2)
x2 = slope.ones(1, 3)
x3 = slope.ones(1, 4)
y = slope.vmap(f)(x1, x2, x3)
print(y.shape)
