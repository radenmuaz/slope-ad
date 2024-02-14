import slope
from lib.models.cv.resnet_cifar import resnet

model = resnet(depth=20)
# model = resnet(depth=8)
N = 100
y = slope.arange(10, dtype=slope.float32)[None].expand((N // 10, 10)).reshape(-1)
x = slope.randn(N, 3, 32, 32)
optimizer = slope.nn.SGD(model, lr=0.1, momentum=0.9, weight_decay=1e-4)


# @slope.jit
# def train_step(model, batch, optimizer):
#     def train_loss_fn(model, batch):
#         x, y = batch
#         # logits, model = model(x, training=True)
#         logits = model(x, training=False)
#         loss = logits.cross_entropy(y) / x.size(0)

#         return loss, (model, logits)
#     # return train_loss_fn(model, batch)

#     (loss, (model, logits)), gmodel = slope.value_and_grad(train_loss_fn, has_aux=True)(model, batch)
#     model, optimizer = optimizer(model, gmodel)
#     return loss, logits, model, optimizer, gmodel

# @slope.jit
# def fn(model, x, y):
#     def f(model, x, y):
#         logits = model(x, training=False)
#         loss = logits.cross_entropy(y) / x.size(0)
#         return loss
#     return slope.jvp(f, (model, x, y),(model, x, y))
# out, out_dot = fn(model, x, y)
# # with slope.core.Profiling(): fn(model, x, y)
# for i in range(10):
#     with slope.core.Timing(f"RUN {i}: "):
#         fn(model, x, y)


def train_loss_fn(model, batch):
    x, y = batch
    # logits, model = model(x, training=True)
    logits = model(x, training=False)
    loss = logits.cross_entropy(y) / x.size(0)
    return loss, (model, logits)
gtrain_loss_fn = slope.value_and_grad(train_loss_fn, has_aux=True)
@slope.jit
def train_step(model, batch, optimizer):
    # return train_loss_fn(model, batch)
    (loss, (model, logits)), gmodel = gtrain_loss_fn(model, batch)
    model, optimizer = optimizer(model, gmodel)
    return loss, logits, model, optimizer, gmodel


train_step(model, (x, y), optimizer)
# with slope.core.Profiling(): loss, logits, model, optimizer, gmodel = train_step(model, (x, y), optimizer)
for i in range(10):
    with slope.core.Timing(f"RUN {i}: "):
        train_step(model, (x, y), optimizer)