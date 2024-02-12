import slope
from lib.models.cv.resnet_cifar import resnet
model = resnet(depth=20)

# @slope.jit
# def f(x):
#     return model(x)
# for i in range(100):
#     with slope.core.Timing(f"RUN {i}\n"):
#         x = slope.randn(100,3,32,32)
#         y_hat = f(x)
#         print(y_hat)

# def one_hot(y, n=10):
#     y_counter = slope.arange(n, dtype=slope.int32)[None, ..., None]
#     y_oh = (y_counter == y[..., None, None]).where(-1.0, 0.0).squeeze(-1).cast(slope.float32)
#     return y_oh

# def cross_entropy(x, y_one_hot):
#     return (x.log_softmax(-1) * y_one_hot).sum()

# @slope.jit
# def train_step(model, batch):
#     def train_loss_fn(model, batch):
#         x, y_oh = batch
#         logits, model = model(x, training=True)
#         return logits
#         loss = cross_entropy(logits, y_oh)
#         return loss
#         # return loss, (model, logits)
#     return train_loss_fn(model, batch)
#     loss = slope.grad(train_loss_fn)(model, batch)
#     return loss
#     # (loss, (model, logits)), gmodel = slope.value_and_grad(train_loss_fn, has_aux=True)(model, batch)
#     # return loss, logits, model, gmodel

# N = 100
# y = slope.arange(10, dtype=slope.float32)[None].expand((N//10,10)).reshape(-1)
# y_oh = one_hot(y)
# x = slope.randn(N,3,32,32)
# for i in range(100):
#     with slope.core.Timing(f"RUN {i}\n"):
#         loss = train_step(model, (x, y_oh))


@slope.backend.operator_set.register("relu")
class ReLUOp(slope.core.UnaryOperator):
    def jvp(self, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        y = x.relu()
        y_dot = x_dot * (x == y).where(slope.ones_like(x), slope.zeros_like(x))
        return [y], [y_dot]

    def T(self, cotangents, x):
        (gL_y,) = cotangents
        gL_x = (x.relu() > x.zeros_like()).cast(gL_y.dtype) * gL_y
        return [gL_x]

@slope.jit
def train_step(model, batch, optimizer):
    def train_loss_fn(model, batch):
        x, y = batch
        logits, model = model(x, training=True)
        loss = logits.cross_entropy(y) / x.size(0)

        return loss, (model, logits)

    (loss, (model, logits)), gmodel = slope.value_and_grad(train_loss_fn, has_aux=True)(model, batch)
    model, optimizer = optimizer(model, gmodel)
    return loss, logits, model, optimizer, gmodel

optimizer = slope.nn.SGD(model, lr=0.1, momentum=0.9, weight_decay=1e-4)
N = 100
y = slope.arange(10, dtype=slope.float32)[None].expand((N//10,10)).reshape(-1)
x = slope.randn(N,3,32,32)
train_step(model, (x, y), optimizer)
with slope.core.Profiling(f"RUN"):
    loss, logits, model, optimizer, gmodel = train_step(model, (x, y), optimizer)