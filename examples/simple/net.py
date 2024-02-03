import slope
import slope.nn as nn


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(out_dim)

    def __call__(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = x.relu()
        return x


class Net(nn.Module):
    def __init__(self):
        self.block1 = Block(3, 8)
        self.flatten_fn = nn.Fn(lambda x: x.reshape((x.shape[0], -1)))
        self.linear = nn.Linear(8192, 10)

    def __call__(self, x, training=False):
        x = self.block1(x, training)
        x = self.flatten_fn(x)
        x = self.linear(x)
        return (x, self) if training else x


def loss_fn(model, batch):
    inputs, targets = batch
    preds, model_ = model(inputs, training=True)
    return -(preds.log_softmax() * targets).sum(), model_


g_loss_fn = slope.value_and_grad(loss_fn, has_aux=True)


@slope.jit
def gstep(model, batch):
    (loss, model_), g_model = g_loss_fn(model, batch)
    return loss, g_model, model_


model = Net()
_, treedef = slope.core.tree_flatten(model)
print(treedef)
x = slope.rand((1, 3, 32, 32))
y = slope.ones((1,)).one_hot(10, dtype=slope.float32)
loss, g_model, model_ = gstep(model, (x, y))
