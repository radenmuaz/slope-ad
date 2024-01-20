import slope

print('#1')
x = slope.tensor([[0.,1.],[2.,3.]], dtype=slope.float32)
w = slope.tensor([[1,0],[0,1]], dtype=slope.int32)
# w = slope.tensor([[0,0],[1,1]], dtype=slope.int32)
print(f"{x=}")
print(f"{w=}")
y = x.gather(w,1)
print(f"{y=}")

# print('\n#2')
# x = slope.tensor([[0.,1.],[2.,3.]], dtype=slope.float32)
# w = slope.tensor([[1],[0]]).cast(slope.int64)
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather(w)
# print(f"{y=}")

# print('\n#3')
# x = slope.tensor([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=slope.float32)
# w = slope.tensor([[0,1],[1,0]], dtype=slope.int32)
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather(w)
# print(f"{y=}")


# print('\n#4')
# x = slope.tensor([[[0,1],[2,3]],[[4,5],[6,7]]], dtype=slope.float32)
# w = slope.tensor([[[0,1]],[[1,0]]], dtype=slope.int32)
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather(w)
# print(f"{y=}")

###############

# x = slope.arange(10, dtype=slope.float32).reshape(2,5)
# w = slope.tensor([1,0])[..., None]
# w = w.cast(slope.int64)
# y = x.gather(w)
# print(f"{x=}")
# print(f"{w=}")
# print(f"{y=}")

# x = slope.arange(24, dtype=slope.float32).reshape(4,3,2)
# w = x
# print(f"{x=}")
# print(f"{w=}")
# y = x.gather(w)
# breakpoint()
# print(f"{y=}")
