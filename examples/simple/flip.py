import slope

x = slope.arange(20).reshape(2,2,5)
y = x.flip(1)
print(x)
print(y)