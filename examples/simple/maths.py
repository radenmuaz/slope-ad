import slope

# x1 = slope.tensor([1.])
# x2 = slope.tensor([2.])
x1 = x2 = slope.full((), 1.)

y1 = x1 + x2
print(f"{y1=}\n")
y2 = x1 + x2
print(f"{y2=}")
