import slope

x1 = slope.tensor([1.])
x2 = slope.tensor([2.])
# x1 = x2 = slope.full((), 1.)

y1 = x1 + x2
print(f"{y1=}")
y2 = x1 + x2
print(f"{y2=}")


x3 = slope.ones(2,3)
y3 = x3.sum(0, keepdim=True)
# y3 = x3.sum()
print(f"{y3=}")