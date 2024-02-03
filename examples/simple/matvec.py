import slope

y_matmat = slope.ones(4, 5) @ slope.ones(5, 6)
print(f"{y_matmat=}")

y_vecmat = slope.ones(5) @ slope.ones(5, 6)
print(f"{y_vecmat=}")

y_matvec = slope.ones(4, 5) @ slope.ones(5)
print(f"{y_matvec=}")

y_bmatvec = slope.ones(2, 4, 5) @ slope.ones(5)
print(f"{y_bmatvec=}")

y_bvecmat = slope.ones(5) @ slope.ones(2, 5, 6)
print(f"{y_bvecmat=}")
