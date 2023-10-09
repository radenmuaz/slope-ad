import slope
import slope.nn as nn
import math
# x = slope.ones((1, 2))
# y = slope.full((1,), 1)
model = nn.Linear(2, 1)
# # model = MLP(2, 3, 1)

# model_flat, model_treedef = slope.tree_flatten(model)
# model2 = slope.tree_unflatten(model_treedef, model_flat)
# print(model == model2)
# print(model2.bias)

models = (model, (model,))
models_flat, models_treedef = slope.tree_flatten(models)
breakpoint()
# @slope.jit
# def loss_fn(model, batch):
#     x, y = batch
#     y_hat = model(x)
#     loss = (y - y_hat).sum()
#     return loss
# print(loss_fn(model, (x, y)))
# print(slope.M().backend.gen_jit_fn.cache_info())
# print(loss_fn(model, (x, y)))
# print(slope.M().backend.gen_jit_fn.cache_info())


# model = nn.Module()
# model.w = slope.ones(3)

# def loss_fn(model):
#     loss = model.w.sum()
#     return loss

# print(loss_fn(model))


# # print(loss_fn(model, (x, y)))
# g_loss_fn = slope.grad(loss_fn)
# print(g_loss_fn(model, (x, y)).flatten())



# print(g_loss_fn(model, (x, y)).flatten()[1])
# breakpoint()
# print(slope.jit(g_loss_fn)(model, (x, y)).flatten()[1])

# g_loss_fn = slope.jit(g_loss_fn)
# print(loss_fn(model, (x, y)))


# a = (y,(y,y))
# af, at = slope.tree_flatten(a)
# b = slope.tree_unflatten(at, af)
# print(a)
# print(b)

# print(f"{af=}")
# print(f"{ta=}")


# m = Linear(2, 1, True)
# m = MLP(2, 3, 1)
# mf, mt = slope.tree_flatten(m)
# m_hat = slope.tree_unflatten(mt, mf)
# print(f"{mf=}")
# print(f"{mt=}")


