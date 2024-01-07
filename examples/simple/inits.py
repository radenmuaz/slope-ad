import slope

XDIMS = (1,4)
x1 = slope.ones(XDIMS)
x2 = slope.randn(XDIMS)
x3 = slope.rand(XDIMS)
x4 = slope.arange(4).unsqueeze(0)
print(f"{x1=}")
print(f"{x2=}")
print(f"{x3=}")
print(f"{x4=}")
