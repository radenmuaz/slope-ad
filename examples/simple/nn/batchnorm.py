import slope
import slope.nn as nn
import math

class Block(nn.Module):
    def __init__(self, embed_dim):
        self.linear1 = nn.Linear(embed_dim,embed_dim)
        self.bn1 = nn.BatchNorm(embed_dim)
    def __call__(self, x, training=False):
        x = self.linear1(x)
        x = self.bn1(x, training=training)
        x = x.relu()
        return x

class Net(nn.Module):
    def __init__(self, num_blocks, in_dim, embed_dim, out_dim):
        self.in_proj = nn.Linear(in_dim, embed_dim)
        self.encoder = nn.Serial([Block(embed_dim)] * num_blocks)
        self.bn_out = nn.BatchNorm(embed_dim)
        self.out_proj = nn.Linear(embed_dim, out_dim)
    
    def __call__(self, x, training=False):
        breakpoint()
        x = self.in_proj(x)
        x = self.encoder(x, training=training)
        x = self.bn_out(x, training=training)
        x = self.out_proj(x)
        return (x, self) if training else x
    
def loss_fn(model, batch):
    x, y = batch
    # y_hat = model(x)
    y_hat, model = model(x, training=True)
    breakpoint()
    loss = (y - y_hat).sum()
    # return loss
    return loss, model

value_and_grad_loss_fn = slope.value_and_grad(loss_fn, has_aux=True)

# @slope.jit
def train_step(model, batch, optimizer):
    ((loss, model), grad_loss_model) = value_and_grad_loss_fn(model, batch)
    model, optimizer = optimizer(model, grad_loss_model)
    return loss, model, optimizer

x = slope.ones((1, 3, 16, 16))
y = slope.full((1,), 1)
batch = (x,y)
model = Net(2, 3, 5, 1)
optimizer = nn.SGD(model, lr=1e-3, momentum=0.8, weight_decay=1e-5)
loss, model_, optimizer_ = train_step(model, batch, optimizer)
breakpoint()