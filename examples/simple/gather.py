import slope

# x = slope.arange(10, dtype=slope.float32).reshape(2,5)
# w = slope.tensor([1,0])[..., None]
# w = w.cast(slope.int64)
# y = x.gather_nd(w)
# print(f"{x=}")
# print(f"{w=}")
# print(f"{y=}")


x = slope.tensor([[0.,1.],[2.,3.]], dtype=slope.float32)
w = slope.tensor([[0,0],[1,1]], dtype=slope.int32)
print(f"{x=}")
print(f"{w=}")
y = x.gather_nd(w)
print(f"{y=}")

# x = slope.tensor([[0.,1.],[2.,3.]], dtype=slope.float32)
# w = slope.tensor([[1],[0]]).cast(slope.int64)
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather_nd(w)
# print(f"{y=}")


# x = slope.ones(8)
# print(f"before: {x=}")
# w = slope.tensor([[4], [3], [1], [7]], dtype=slope.int32)
# u = slope.tensor([9., 10., 11., 12.])
# y = slope.scatter_nd(x,w,u)

# print(f"{w=}")
# print(f"{u=}")
# print(f"{x=}")
# print(f"{y=}")