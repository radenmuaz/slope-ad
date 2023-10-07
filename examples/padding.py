import slope



def f(x):
    out = x
    out = out.pad(((1, 0),))
    # out = out.pad_xla( (1,), (0,))
    out = out.sum()
    return out


x = slope.ones(3)
x_dot = slope.ones(3)
# print(f(x))
# print(slope.jvp(f, (x,), (x_dot,)))
print(slope.grad(f)(x))


# def f(x):
#     out = x
#     out = out.pad( ((1,1), (2,2)) )
#     out = out.pad( ((1,1), (2,2)) )
#     out = out.sum()
#     return out

# x = slope.ones((2, 3))
# x_dot = slope.ones((2,3))
# # print(f(x))
# # print(slope.jvp(f, (x,), (x_dot,)))
# print(slope.grad(f)(x))
