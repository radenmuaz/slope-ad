import slope

# x = slope.arange(10, dtype=slope.float32).reshape(2,5)
# w = slope.tensor([1,0])[..., None]
# w = w.cast(slope.int64)
# y1 = x.gather_nd(w)
# print(f"{x=}")
# print(f"{w=}")
# print(f"{y=}")

x = slope.ones(8)
print(f"{x=}")
w = slope.tensor([[4], [3], [1], [7]], dtype=slope.int32)
u = slope.tensor([9., 10., 11., 12.])
y = x.scatter_nd(w,u)

print(f"{x=}")
print(f"{w=}")
print(f"{u=}")
print(f"{y=}")