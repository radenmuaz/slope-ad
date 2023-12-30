import slope
import slope.system.nn as nn
import math
x = slope.ones((1, 2))
y = slope.full((1,), 1)
model = nn.Linear(2, 1)
# model = MLP(2, 3, 1)
# model_flat, model_treedef = slope.tree_flatten(model)
# model2 = slope.tree_unflatten(model_treedef, model_flat)
# print(model == model2)
# print(model2.bias)
@slope.jit
def loss_fn(model, batch):
    x, y = batch
    y_hat = model(x)
    loss = (y - y_hat).sum()
    return loss

print(loss_fn(model, (x, y)))
grad_loss_fn = slope.grad(loss_fn)
print(grad_loss_fn(model, (x, y)).flatten())



