import slope

x = slope.ones(5)
w = slope.arange(5, dtype=x.dtype)
y_eq = (x == w).int()
y_ne = (x != w).int()
y_lt = (x < w).int()
y_le = (x <= w).int()
y_ge = (x >= w).int()
y_gt = (x > w).int()


print(f"{x=}")
print(f"{w=}")
print(f"x == w = {y_eq}")
print(f"x != w = {y_ne}")
print(f"x <  w = {y_lt}")
print(f"x <= w = {y_le}")
print(f"x >  w = {y_gt}")
print(f"x >= w = {y_ge}")
