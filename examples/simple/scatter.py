import slope

x = slope.zeros(8)
print(f"before: {x=}")
w = slope.tensor([[4], [3], [1], [7]], dtype=slope.int32)
u = slope.tensor([9., 10., 11., 12.])
y = slope.scatter(x,w,u)
# print(f"{w=}")
# print(f"{u=}")
# print(f"{x=}")
print(f"{y=}")

