import slope
from slope import environment as sev

x = sev.ones((3,))
y = sev.concatenate([x,x])
print(y)