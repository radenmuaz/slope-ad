import slope
from slope.core import Tensor

import slope.system.nn as nn

import time
import itertools
import math
import numpy as np
from tqdm import tqdm
from typing import Dict, Any

import numpy as np

BS, EVAL_BS, STEPS = 512, 50, 1000
np_dtype = np.float32
bias_scaler = 58
hyp: Dict[str, Any] = {
    "seed": 209,
    "opt": {
        "bias_lr": 1.76 * bias_scaler / 512,
        "non_bias_lr": 1.76 / 512,
        "bias_decay": 1.08 * 6.45e-4 * BS / bias_scaler,
        "non_bias_decay": 1.08 * 6.45e-4 * BS,
        "final_lr_ratio": 0.025,
        "initial_div_factor": 1e16,
        "label_smoothing": 0.20,
        "momentum": 0.85,
        "percent_start": 0.23,
        "loss_scale_scaler": 1.0 / 128,  # (range: ~1/512 - 16+, 1/128 w/ FP16)
    },
    "net": {
        "kernel_size": 2,  # kernel size for the whitening layer
        "cutmix_size": 3,
        "cutmix_steps": 499,
        "pad_amount": 2,
    },
    "ema": {
        "steps": 399,
        "decay_base": 0.95,
        "decay_pow": 1.6,
        "every_n_steps": 5,
    },
}


def whitening(X, kernel_size=hyp["net"]["kernel_size"]):
    def _cov(X):
        X = X / np.sqrt(X.shape[0] - 1)
        return X.T @ X

    def _patches(data, patch_size=(kernel_size, kernel_size)):
        h, w = patch_size
        c = data.shape[1]
        dim: SupportsIndex = (2, 3)  # type: ignore
        return (
            np.lib.stride_tricks.sliding_window_view(data, window_shape=(h, w), dim=dim)
            .transpose((0, 3, 2, 1, 4, 5))
            .reshape((-1, c, h, w))
        )

    def _eigens(patches):
        n, c, h, w = patches.shape
        Σ = _cov(patches.reshape(n, c * h * w))
        Λ, V = np.linalg.eigh(Σ, UPLO="U")
        return np.flip(Λ, 0), np.flip(V.T.reshape(c * h * w, c, h, w), 0)

    Λ, V = _eigens(_patches(X.numpy()))
    W = V / np.sqrt(Λ + 1e-2)[:, None, None, None]

    return Tensor(W.astype(np_dtype), requires_grad=False)


class ConvGroup:
    def __init__(self, channels_in, channels_out):
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)

        self.norm1 = nn.BatchNorm(channels_out)
        self.norm2 = nn.BatchNorm(channels_out)

    def __call__(self, x):
        x = self.conv1(x)
        x = x.max_pool2d(2)
        x = x.float()
        x = self.norm1(x)
        x = x.gelu()
        residual = x
        x = self.conv2(x)
        x = x.float()
        x = self.norm2(x)
        x = x.gelu()
        return x + residual


class SpeedyResNet:
    def __init__(self, W):
        self.whitening = W
        self.net = [
            nn.Conv2d(12, 32, kernel_size=1, bias=False),
            lambda x: x.gelu(),
            ConvGroup(32, 64),
            ConvGroup(64, 256),
            ConvGroup(256, 512),
            lambda x: x.max((2, 3)),
            nn.Linear(512, 10, bias=False),
            lambda x: x.mul(1.0 / 9),
        ]

    def __call__(self, x, training=True):
        forward = lambda x: x.conv2d(self.whitening).pad2d((1, 0, 0, 1)).sequential(self.net)
        return forward(x) if training else forward(x) * 0.5 + forward(x[..., ::-1]) * 0.5


if __name__ == "__main__":
    net = SpeedyResNet(whitening)
