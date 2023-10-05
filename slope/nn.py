import slope
from slope import environment as sev
from slope.core import Tensor, Module
from typing import Tuple
def dedup(x): return list(dict.fromkeys(x))   # retains list order


class Optimizer(Module):
    def __init__(self, module, lr: float):
        self.module = module
        self.lr = sev.full((), lr)
        self.states = Module()
    
    def step(self, p, g, s):
        return p, g, s

    def __call__(self, module, g_module):
        m_flat, m_treedef = slope.tree_flatten(module)
        g_flat, g_treedef = slope.tree_flatten(g_module)
        
        s_names, s_flats, s_treedefs = [], [], []
        for state_name, state in self.states.get_named_modules():
            s_names = [state_name]
            s_flat, s_treedef = slope.tree_flatten(state)
            s_flats += [s_flat]
            s_treedefs += [s_treedef]

        m_flat_new = []
        s_flats_new = []
        for p, g, ss in zip(m_flat, g_flat, s_flats):
            p_new, ss_new = self.step(p, g, ss)
            m_flat_new += [p_new]
            s_flat_new = []
            for s_new in ss_new:
                s_flat_new += [s_new]
            s_flats_new += [s_flat_new]

        module = slope.unflatten(m_treedef, tuple(m_flat_new))
        self.states = Module()
        for s_name, s_treedef, s_flat_new in zip(s_names, s_treedefs, s_flats_new):
            setattr(self.states, s_name, slope.tree_unflatten(s_treedef, s_flat_new))
        
        return module, self



class SGD(Optimizer):
    def __init__(self, module, lr=0.001, momentum: float=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(module, lr)
        self.momentum = momentum
        self.wd = weight_decay
        self.nesterov = nesterov
        self.states.b = slope.tree_map(lambda x: x.zeros_like(), self.module)
        breakpoint()

    def step(self, p, g, **s):
        b, lr, m = s['b'], self.lr, self.momentum
        b = p * b + g
        g = (g + m * b) if self.nesterov else b
        p = lr * g + p
        return p, s

# # LAMB is essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 its just Adam/W.
# def AdamW(params: Tuple[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
#     return LAMB(params, lr, b1, b2, eps, wd, adam=True)


# def Adam(params: Tuple[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
#     return LAMB(params, lr, b1, b2, eps, 0.0, adam=True)


# class LAMB(Optimizer):
#     def __init__(self, params: Tuple[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, wd=0.0, adam=False):
#         super().__init__(params, lr)
#         self.b1, self.b2, self.eps, self.wd, self.adam, self.t = (
#             b1,
#             b2,
#             eps,
#             wd,
#             adam,
#             Tensor([0], requires_grad=False).realize(),
#         )
#         self.m = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in self.params]
#         self.v = [Tensor.zeros(*t.shape, device=t.device, requires_grad=False) for t in self.params]

#     def step(self) -> None:
#         self.t.assign(self.t + 1).realize()
#         for i, t in enumerate(self.params):
#             assert t.grad is not None
#             g = t.grad.realize()
#             self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * g).realize()
#             self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * (g * g)).realize()
#             m_hat = self.m[i] / (1.0 - self.b1**self.t)
#             v_hat = self.v[i] / (1.0 - self.b2**self.t)
#             up = (m_hat / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()
#             if not self.adam:
#                 r1 = t.detach().square().sum().sqrt()
#                 r2 = up.square().sum().sqrt()
#                 r = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
#             else:
#                 r = 1.0
#             t.assign(t.detach() - self.lr * r * up)
#         self.realize([self.t] + self.m + self.v)
