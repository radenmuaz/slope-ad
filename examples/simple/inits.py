import slope

# x1 = slope.arange(5)
# print(f"{x1=}")
x2 = slope.randn((1, 4))
print(f"{x2=}")

# x2 = x2.expand((2,3))
# x2 = x2.reshape(2,2)
# x2 = x2.pad((1,0))
# x2 = x2.slice((0,),(3,))
# x2 = x2[:3]
x2 = x2.transpose(0,1)
print(f"{x2=}")
