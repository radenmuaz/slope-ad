import slope
from math import prod

bsz = 2
w_size = (64, 32, 3, 3)
x_size = (bsz, 32, 16, 16)
y_size = (bsz, 64, 9, 9)

stride = 2
padding = 2
dilation = 1
groups=1

x = slope.arange(prod(x_size)).reshape(*x_size).float()
# x = (x - x.mean()) / x.std() 
x.requires_grad = True

w = slope.arange(prod(w_size)).reshape(*w_size).float()
# w = (w - w.mean()) / w.std() 

y = x.conv2d(w, stride=stride, padding=padding, dilation=dilation)
x_grad = y.conv_transpose2d(w, stride=stride, padding=padding, dilation=dilation, output_padding=1, groups=groups)
breakpoint()

# w_grad = F.conv2d(x.transpose(0,1), y.grad.transpose(0,1), stride=dilation, padding=padding, dilation=stride, groups=groups).transpose(0,1)
# w_grad = w_grad[:,:,:w.size(2),:w.size(3)]
# assert slope.allclose(w_grad, w.grad)
# print(w.grad[0][0], '\n', w_grad[0][0])

breakpoint()