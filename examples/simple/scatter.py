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

# y_ans = slope.tensor([ 1., 12.,  1., 11., 10.,  1.,  1., 13.], dtype=slope.float32)


# x = slope.tensor([
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0],
# ], dtype=slope.float32)
# w = slope.tensor([
#     [1, 0, 2],
#     [0, 2, 1],
# ], dtype=slope.int32)

# u = slope.tensor([
#     [1.0, 1.1, 1.2],
#     [2.0, 2.1, 2.2],
# ], dtype=slope.float32)
# # u = u.unsqueeze(-1)
# y = x.scatter(w, u, axis=0)
# print(f"{y=}")
# print(f"{w=}")
# print(f"{u=}")
# print(f"{x=}")

# y_ans = slope.tensor([
#     [2.0, 1.1, 0.0],
#     [1.0, 0.0, 2.2],
#     [0.0, 2.1, 1.2]
# ], dtype=slope.float32)

# print(f"{y_ans=}")

# x = slope.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=slope.float32)
# w = slope.tensor([[1, 3]], dtype=slope.int32)
# u = slope.tensor([[1.1, 2.1]], dtype=slope.float32)
# y = x.scatter(w, u, axis=1)
# y_ans = slope.tensor([[1.0, 1.1, 3.0, 2.1, 5.0]], dtype=slope.float32)


# x = slope.tensor(
#     [0.0, 0.0, 0.0]
# , dtype=slope.float32)
# w = slope.tensor(
#     [[1],[0]]
# , dtype=slope.int32)

# u = slope.tensor(
#     [1.0, 2.0],
#  dtype=slope.float32)
# y = x.scatter(w, u, axis=0)
# print(f"{y=}")
