import slope
from slope import environment as sev
from slope.core import Tensor, Module
from typing import Tuple
def dedup(x): return list(dict.fromkeys(x))   # retains list order


class Optimizer(Module):
    def __init__(self, params, lr: float):
        self.params = params
        self.state = Module()
        self.hp = Module()
        self.hp.lr = slope.full((), lr)
        self.iters = slope.zeros(())

    
    def step(self, p, g, *state_attrs):
        state = Module()
        for i, a in enumerate(state_attrs):
            setattr(state, f'attr{i}', a)
        return p, g, state
    
    def __call__(self, params, g_params):
        state_attrs = self.state.get_modules()
        out = slope.tree_map(self.step, params, g_params, *state_attrs)
        params = slope.tree_map(lambda x: x[0], out)
        self.state = slope.tree_map(lambda x: x[1], out)
        self.iters = self.iters + 1
        return params, self



class SGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum: float=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr)
        self.hp.momentum = momentum
        self.hp.wd = weight_decay
        self.hp.nesterov = nesterov
        self.state.b = slope.tree_map(lambda x: x.zeros_like(), self.params)

    def step(self, p, g, b):
        lr, m = self.hp.lr, self.hp.momentum
        b = p * b + g
        g = (g + m * b) if self.hp.nesterov else b
        p = lr * g + p
        state = Module()
        state.b = b
        return p, state

# LAMB is essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 its just Adam/W.
def AdamW(params: Tuple[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
    return LAMB(params, lr, b1, b2, eps, wd, adam=True)


def Adam(params: Tuple[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    return LAMB(params, lr, b1, b2, eps, 0.0, adam=True)


class LAMB(Optimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, adam=False):
        super().__init__(params, lr)
        self.hp.b1 = b1
        self.hp.b2 = b2
        self.hp.eps = eps
        self.hp.wd = weight_decay
        self.hp.adam = adam
        self.state.m = slope.tree_map(lambda x: x.zeros_like(), self.params)
        self.state.v = slope.tree_map(lambda x: x.zeros_like(), self.params)

    def step(self, p, g, m, v):
        lr, wd, adam = self.hp.lr, self.hp.wd, self.hp.adam
        b1, b2, eps = self.hp.b1, self.hp.b2, self.hp.eps
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        m_hat = m / (1.0 - b1**self.iters)
        v_hat = v / (1.0 - b2**self.iters)
        up = (m_hat / (v_hat.sqrt() + eps)) + wd * p
        if not adam:
            r1 = p.square().sum().sqrt()
            r2 = up.square().sum().sqrt()
            r = slope.where(r1 > 0, slope.where(r2 > 0, r1 / r2, 1.0), 1.0)
        else:
            r = 1.0
        p = p * lr * r * up
        state = Module()
        state.m = m
        state.v = v
        return p, state
