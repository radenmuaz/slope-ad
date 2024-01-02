import slope

# binaryop
# def f(x):
#     # return x + 1
#     return x + x.ones_like()
# x = slope.ones(3,1)
# # y = (f)(x)
# y = slope.vmap(f)(x)
# print(y)
# print(x.shape, y.shape)


# binaryop
def f(x):
    return x.pad((1,2))
x = slope.ones(1,3)
# y = f(x)
y = slope.vmap(f)(x)
print(y)
print(x.shape, y.shape)