"""
DQN agent built on the from-scratch MLP (nn_numpy.py).

    act      : epsilon-greedy  -> random action with prob eps, else argmax Q(s)
    target   : y     = r + gamma * (1 - done) * max_a' Q_target(s', a')
    error    : delta = Q(s, a) - y
    loss     : Huber(delta) = 0.5*delta^2        if |delta| <= 1
                              |delta| - 0.5      otherwise
    gradient : dL/dQ(s,a) = clip(delta, -1, 1) / batch     (other 2 actions get 0)
    target net: frozen copy of Q, refreshed every N steps
"""
from __future__ import annotations

import numpy as np

from nn_numpy import MLP


class ReplayBuffer:
    """Circular memory of past steps. Random samples break step-to-step correlation."""

    def __init__(self, capacity: int, obs_size: int, seed: int = 0):
        self.capacity = capacity
        self.s = np.zeros((capacity, obs_size), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_size), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.i = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, s, a, r, s2, done) -> None:
        self.s[self.i], self.a[self.i], self.r[self.i] = s, a, r
        self.s2[self.i], self.done[self.i] = s2, done
        self.i = (self.i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int):
        idx = self.rng.integers(0, self.size, n)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.done[idx]


class DQNAgent:
    def __init__(self, obs_size: int, n_actions: int, hidden=(64, 64), lr: float = 1e-3,
                 gamma: float = 0.95, seed: int = 0, dtype=np.float32):
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.q = MLP([obs_size, *hidden, n_actions], seed=seed, dtype=dtype)
        self.q_target = self.q.clone()

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        return self.q.predict(obs[None])[0]

    def act(self, obs: np.ndarray, epsilon: float) -> tuple[int, bool]:
        """Returns (action, was_random)."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions)), True
        return int(np.argmax(self.q_values(obs))), False

    def loss_and_grads(self, batch):
        s, a, r, s2, done = batch
        B = len(a)
        idx = np.arange(B)
        q_next = self.q_target.predict(s2).max(axis=1)        # best future value (frozen net)
        y = r + self.gamma * (1.0 - done) * q_next              # target
        q_all = self.q.forward(s, keep=True)                    # current guesses
        delta = q_all[idx, a] - y                               # error on taken action only
        abs_d = np.abs(delta)
        loss = float(np.where(abs_d <= 1.0, 0.5 * delta ** 2, abs_d - 0.5).mean())
        dq = np.zeros_like(q_all)
        dq[idx, a] = np.clip(delta, -1.0, 1.0) / B
        grads = self.q.backward(dq)
        return loss, grads, float(q_all[idx, a].mean())

    def learn(self, batch) -> tuple[float, float]:
        loss, grads, q_mean = self.loss_and_grads(batch)
        self.q.adam_step(grads, lr=self.lr)
        return loss, q_mean

    def sync_target(self) -> None:
        self.q_target.copy_from(self.q)

    def save(self, path: str, **meta) -> None:
        self.q.save(path, **meta)
