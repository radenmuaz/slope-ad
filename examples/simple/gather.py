import slope

x = slope.arange(10).reshape(2,5)
w = slope.tensor([2,3], dtype=slope.int32)[..., None]
y = x.gather_nd(w)

print(f"{x=}")
print(f"{w=}")
print(f"{y=}")
