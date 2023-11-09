import slope

a = slope.ones(3)
print(f"{a=}")
slope.save(a,'/tmp/a.safetensors')
ah =slope.load('/tmp/a.safetensors')
print(f"{ah=}")
print(f"{(a==ah)=}")