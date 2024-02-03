import slope

a = slope.ones(3)
print(f"{a=}")
slope.save(a, "/tmp/a.safetensors")
ah = slope.load("/tmp/a.safetensors")
print(f"{ah=}")
print(f"{(a==ah)=}")


b = {"b1": slope.ones(3), "b2": slope.zeros(3)}
print(f"{b=}")
slope.save(b, "/tmp/b.safetensors")
bh = slope.load("/tmp/b.safetensors")
print(f"{bh=}")
