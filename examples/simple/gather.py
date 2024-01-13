import slope

x = slope.arange(10, dtype=slope.float32).reshape(2,5)
w = slope.tensor([1,0])[..., None]
w = w.cast(slope.int64)
y = x.gather_nd(w)

print(f"{x=}")
print(f"{w=}")
print(f"{y=}")
