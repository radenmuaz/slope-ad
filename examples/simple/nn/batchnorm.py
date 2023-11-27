import slope
import slope.nn as nn
import math


class Net(nn.Module):
    def __init__(self):
        # self.linear1 = nn.(2,3)
        self.bn1 = nn.BatchNorm(3)
        # self.linear2 = nn.Linear(3,1)
    
    def __call__(self, x, training=False):
        # x = self.linear1(x)
        x = self.bn1(x, training)

        x = x.sum((1,2,3))
        # x = self.linear2(x)
        return (x, self) if training else x
    
def loss_fn(model, batch):
    x, y = batch
    # y_hat = model(x)
    y_hat, model = model(x, training=True)
    loss = (y - y_hat).sum()
    # return loss
    return loss, model

value_and_grad_loss_fn = slope.value_and_grad(loss_fn, has_aux=True)

@slope.jit
def train_step(model, batch, optimizer):
    ((loss, model), grad_loss_model) = value_and_grad_loss_fn(model, batch)
    model, optimizer = optimizer(model, grad_loss_model)
    return loss, model, optimizer

# loss = loss_fn(model, (x, y))

x = slope.ones((1, 3, 16, 16))
y = slope.full((1,), 1)
batch = (x,y)
model = Net()
optimizer = nn.SGD(model, lr=1e-3, momentum=0.8, weight_decay=1e-5)
loss, model_, optimizer_ = train_step(model, batch, optimizer)
breakpoint()
