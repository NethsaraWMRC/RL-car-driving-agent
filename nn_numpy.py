"""
A tiny neural network written from scratch with numpy (no PyTorch).

Shapes use the "batch of rows" convention:  x is (batch, features).

Forward, layer i = 1..L:
    z_i = h_{i-1} @ W_i + b_i
    h_i = ReLU(z_i) = max(0, z_i)     (hidden layers)
    out = z_L                          (last layer: no ReLU, Q-values can be negative)

Backward (chain rule), given d = dLoss/d(out):
    dW_i = h_{i-1}^T @ d
    db_i = sum of d over the batch
    d    = (d @ W_i^T) * (z_{i-1} > 0)     # send error back through ReLU

Adam optimizer, per parameter:
    m     = b1*m + (1-b1)*g               # running average direction
    v     = b2*v + (1-b2)*g^2             # running average size
    m_hat = m / (1 - b1^t)                # bias fix for early steps
    v_hat = v / (1 - b2^t)
    theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import json

import numpy as np


class MLP:
    def __init__(self, sizes, seed: int = 0, dtype=np.float32):
        self.sizes = [int(s) for s in sizes]
        self.L = len(self.sizes) - 1
        self.dtype = dtype
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {}
        for i in range(1, self.L + 1):
            n_in, n_out = self.sizes[i - 1], self.sizes[i]
            # He init: keeps signal size stable through ReLU layers
            self.params[f"W{i}"] = (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(dtype)
            self.params[f"b{i}"] = np.zeros(n_out, dtype=dtype)
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        self._cache = None

    # ----------------------------------------------------------- forward
    def forward(self, x: np.ndarray, keep: bool = False) -> np.ndarray:
        h = x
        acts, zs = [x], []
        for i in range(1, self.L + 1):
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            h = np.maximum(z, 0) if i < self.L else z
            zs.append(z)
            acts.append(h)
        if keep:
            self._cache = (acts, zs)
        return h

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, keep=False)

    # ---------------------------------------------------------- backward
    def backward(self, dout: np.ndarray) -> dict[str, np.ndarray]:
        """dout = dLoss/d(output) for the batch used in the last forward(keep=True)."""
        if self._cache is None:
            raise RuntimeError("call forward(x, keep=True) before backward()")
        acts, zs = self._cache
        grads = {}
        d = dout
        for i in range(self.L, 0, -1):
            grads[f"W{i}"] = acts[i - 1].T @ d
            grads[f"b{i}"] = d.sum(axis=0)
            if i > 1:
                d = (d @ self.params[f"W{i}"].T) * (zs[i - 2] > 0)
        return grads

    # --------------------------------------------------------- optimizer
    def adam_step(self, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, max_norm=10.0) -> None:
        if max_norm is not None:  # gradient clipping: cap the total step size
            norm = float(np.sqrt(sum(float((g * g).sum()) for g in grads.values())))
            if norm > max_norm:
                grads = {k: g * (max_norm / norm) for k, g in grads.items()}
        self.t += 1
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** self.t)
            v_hat = self.v[k] / (1 - b2 ** self.t)
            self.params[k] -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(self.dtype)

    # ------------------------------------------------------------- utils
    def copy_from(self, other: "MLP") -> None:
        for k in self.params:
            self.params[k][...] = other.params[k]

    def clone(self) -> "MLP":
        new = MLP(self.sizes, dtype=self.dtype)
        new.copy_from(self)
        return new

    def n_params(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def save(self, path: str, **meta) -> None:
        np.savez(path, sizes=np.array(self.sizes), meta=np.array(json.dumps(meta)), **self.params)

    @classmethod
    def load(cls, path: str) -> tuple["MLP", dict]:
        data = np.load(path)
        net = cls(data["sizes"].tolist(), dtype=data["W1"].dtype)
        for k in net.params:
            net.params[k][...] = data[k]
        meta = json.loads(str(data["meta"])) if "meta" in data else {}
        return net, meta
