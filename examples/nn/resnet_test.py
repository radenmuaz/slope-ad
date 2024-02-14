import slope
from lib.models.cv.resnet_cifar import resnet

model = resnet(depth=20)
# model = resnet(depth=8)
N = 100
y = slope.arange(10, dtype=slope.float32)[None].expand((N // 10, 10)).reshape(-1)
x = slope.randn(N, 3, 32, 32)
optimizer = slope.nn.SGD(model, lr=0.1, momentum=0.9, weight_decay=1e-4)

def train_loss_fn(model, x, y):
    logits, model = model(x, training=True)
    # logits = model(x, training=False)
    loss = logits.cross_entropy(y) / x.size(0)
    return loss, ()#(model, logits)
    # return loss, (model, logits)

slope.jit(train_loss_fn)(model, x, y)
with slope.core.Timing(f"RUN: "): slope.jit(train_loss_fn)(model, x, y)
grad_fn = slope.jit(slope.value_and_grad(train_loss_fn, has_aux=True))
grad_fn(model, x, y)
with slope.core.Timing(f"GRAD: "): grad_fn(model, x, y)

# with slope.core.Profiling(): loss, logits, model, optimizer, gmodel = train_step(model, (x, y), optimizer)