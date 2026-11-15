"""
Proof that the from-scratch code is correct.  Run:  python test_nn.py

  1. gradient check  - backprop matches numeric slope (L(w+e) - L(w-e)) / 2e
  2. DQN loss check  - Huber/target gradient matches numeric slope
  3. overfit test    - network can learn a fixed function (loss -> ~0)
  4. smoke test      - short DQN training on the game: no NaN, no crash
"""
import numpy as np

from dqn_agent import DQNAgent, ReplayBuffer
from lane_env import LaneEnv
from nn_numpy import MLP


def rel_err(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def numeric_grad(loss_fn, param, eps=1e-6):
    g = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = param[i]
        param[i] = old + eps
        up = loss_fn()
        param[i] = old - eps
        down = loss_fn()
        param[i] = old
        g[i] = (up - down) / (2 * eps)
    return g


def test_gradient_check():
    rng = np.random.default_rng(1)
    net = MLP([6, 5, 4, 3], seed=1, dtype=np.float64)
    x = rng.standard_normal((4, 6))
    R = rng.standard_normal((4, 3))           # loss = sum(out * R)  ->  dL/dout = R
    loss_fn = lambda: float((net.predict(x) * R).sum())
    net.forward(x, keep=True)
    grads = net.backward(R)
    worst = max(rel_err(grads[k], numeric_grad(loss_fn, net.params[k])) for k in net.params)
    assert worst < 1e-6, f"backprop mismatch: {worst:.2e}"
    return f"max relative error {worst:.1e}"


def test_dqn_loss_gradient():
    rng = np.random.default_rng(2)
    agent = DQNAgent(obs_size=6, n_actions=3, hidden=(5, 4), seed=2, dtype=np.float64)
    agent.q_target.params["W3"] += 0.3        # make target net differ from online net
    B = 8
    batch = (rng.standard_normal((B, 6)), rng.integers(0, 3, B), rng.standard_normal(B) * 3,
             rng.standard_normal((B, 6)), (rng.random(B) < 0.3).astype(np.float64))
    _, grads, _ = agent.loss_and_grads(batch)
    loss_fn = lambda: agent.loss_and_grads(batch)[0]
    worst = max(rel_err(grads[k], numeric_grad(loss_fn, agent.q.params[k])) for k in agent.q.params)
    assert worst < 1e-5, f"DQN loss gradient mismatch: {worst:.2e}"
    return f"max relative error {worst:.1e}"


def test_overfit():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, (256, 8))
    Y = np.stack([X.sum(1), X[:, 0] * X[:, 1], np.abs(X[:, 2])], axis=1)
    net = MLP([8, 64, 64, 3], seed=3, dtype=np.float64)
    start = float(((net.predict(X) - Y) ** 2).mean())
    for _ in range(3000):
        out = net.forward(X, keep=True)
        net.adam_step(net.backward((out - Y) / len(X)), lr=3e-3)
    end = float(((net.predict(X) - Y) ** 2).mean())
    assert end < 0.01, f"could not fit: mse {end:.4f}"
    return f"mse {start:.3f} -> {end:.5f}"


def test_smoke():
    env = LaneEnv(seed=0)
    agent = DQNAgent(env.obs_size, env.n_actions, seed=0)
    buf = ReplayBuffer(5000, env.obs_size)
    obs = env.reset()
    losses = []
    for step in range(3000):
        a, _ = agent.act(obs, epsilon=1.0)
        obs2, r, done, info = env.step(a)
        buf.add(obs, a, r, obs2, float(info["crashed"]))
        obs = env.reset() if done else obs2
        if buf.size >= 500:
            losses.append(agent.learn(buf.sample(64))[0])
        if step % 500 == 0:
            agent.sync_target()
    assert all(np.isfinite(losses)), "NaN/inf loss"
    assert all(np.isfinite(p).all() for p in agent.q.params.values()), "NaN/inf weights"
    return f"{len(losses)} updates, last loss {losses[-1]:.4f}"


if __name__ == "__main__":
    tests = [test_gradient_check, test_dqn_loss_gradient, test_overfit, test_smoke]
    failed = 0
    for t in tests:
        try:
            print(f"PASS  {t.__name__:<24} {t()}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__:<24} {e}")
    print("all tests passed" if not failed else f"{failed} test(s) failed")
    raise SystemExit(1 if failed else 0)
