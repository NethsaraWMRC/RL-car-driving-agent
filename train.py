"""
Train the DQN agent on Lane Dodge (no window, fast).

    python train.py                   # default run  -> runs/dqn
    python train.py --steps 150000    # train longer
    python train.py --name test2      # separate run folder

Outputs in runs/<name>/:
    checkpoints/ckpt_XXXXXXX.npz   brain snapshots + stats (watch.py replays these)
    final.npz / best.npz           last brain / best-tested brain
    history.json                   per-episode survival, checkpoint list, config
    learning_curve.png             plots
"""
import argparse
import json
import os
import time

import numpy as np

from dqn_agent import DQNAgent, ReplayBuffer
from lane_env import LaneEnv

HERE = os.path.dirname(os.path.abspath(__file__))


def evaluate(net, env_kwargs: dict, seeds, max_steps: int):
    """Greedy play (no randomness) on fixed seeds -> mean steps survived."""
    env = LaneEnv(**env_kwargs, max_steps=max_steps)
    lengths = []
    for sd in seeds:
        obs, done = env.reset(seed=sd), False
        while not done:
            obs, _, done, info = env.step(int(np.argmax(net.predict(obs[None])[0])))
        lengths.append(info["steps"])
    return float(np.mean(lengths))


def random_baseline(env_kwargs: dict, episodes: int = 200, max_steps: int = 1000) -> float:
    env = LaneEnv(**env_kwargs, max_steps=max_steps, seed=999)
    lengths = []
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, _, done, info = env.step(int(env.rng.integers(3)))
        lengths.append(info["steps"])
    return float(np.mean(lengths))


def moving_avg(x, k=50):
    x = np.asarray(x, dtype=float)
    if len(x) < k:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[k:] - c[:-k]) / k


def plot(history: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = np.array(history["episode_steps"])
    ck = history["checkpoints"]
    cap = history["config"]["max_ep_steps"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5))

    ax1.plot(ep, color="tab:blue", alpha=0.25, lw=0.8, label="each episode")
    if len(ep) >= 50:
        ax1.plot(np.arange(49, len(ep)), moving_avg(ep, 50), color="tab:blue", lw=2.2, label="average of last 50")
    ax1.axhline(history["random_baseline"], color="gray", ls="--", label=f"random agent ({history['random_baseline']:.0f})")
    ax1.axhline(cap, color="tab:green", ls=":", label=f"time limit ({cap})")
    ax1.set(title="Lane Dodge DQN - steps survived per training episode", xlabel="episode", ylabel="steps survived")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    steps = [c["step"] for c in ck]
    ax2.plot(steps, [c["eval"] for c in ck], color="tab:green", marker="o", ms=3, lw=1.8, label="test score (no randomness)")
    ax2.set(title="Test score at each checkpoint", xlabel="training steps", ylabel="avg steps survived")
    ax2.grid(alpha=0.3)
    ax3 = ax2.twinx()
    ax3.plot(steps, [c["epsilon"] for c in ck], color="tab:orange", alpha=0.7, lw=1.4, label="epsilon (exploration)")
    ax3.set_ylabel("epsilon", color="tab:orange")
    ax3.set_ylim(0, 1.05)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train DQN on Lane Dodge")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--name", default="dqn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=9)
    ap.add_argument("--spawn", type=float, default=0.6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--buffer", type=int, default=50_000)
    ap.add_argument("--learn-start", type=int, default=1_000)
    ap.add_argument("--target-sync", type=int, default=1_000)
    ap.add_argument("--eps-start", type=float, default=1.0)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--eps-decay", type=int, default=30_000)
    ap.add_argument("--ckpt-every", type=int, default=2_500)
    ap.add_argument("--early-every", type=int, default=500, help="snapshot interval during early training")
    ap.add_argument("--early-until", type=int, default=10_000)
    ap.add_argument("--eval-episodes", type=int, default=5)
    ap.add_argument("--max-ep-steps", type=int, default=1_000)
    args = ap.parse_args()

    out_dir = os.path.join(HERE, "runs", args.name)
    ck_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    for f in os.listdir(ck_dir):  # fresh run: drop old snapshots
        if f.endswith(".npz"):
            os.remove(os.path.join(ck_dir, f))

    env_kwargs = dict(cols=args.cols, rows=args.rows, spawn_prob=args.spawn)
    env = LaneEnv(**env_kwargs, max_steps=args.max_ep_steps, seed=args.seed)
    agent = DQNAgent(env.obs_size, env.n_actions, hidden=args.hidden, lr=args.lr, gamma=args.gamma, seed=args.seed)
    buf = ReplayBuffer(args.buffer, env.obs_size, seed=args.seed)
    eval_seeds = list(range(10_000, 10_000 + args.eval_episodes))
    rand_base = random_baseline(env_kwargs, max_steps=args.max_ep_steps)

    print(f"network {[env.obs_size, *args.hidden, env.n_actions]}  ({agent.q.n_params():,} weights)")
    print(f"random agent survives {rand_base:.1f} steps on average\n")

    episode_steps: list[int] = []
    checkpoints: list[dict] = []
    recent_losses: list[float] = []
    best_eval = -1.0

    def checkpoint(step: int, eps: float) -> dict:
        nonlocal best_eval
        meta = dict(
            step=step,
            episode=len(episode_steps),
            epsilon=round(eps, 4),
            eval=evaluate(agent.q, env_kwargs, eval_seeds, args.max_ep_steps),
            train_avg=float(np.mean(episode_steps[-50:])) if episode_steps else 0.0,
        )
        path = os.path.join(ck_dir, f"ckpt_{step:07d}.npz")
        agent.save(path, **meta)
        if meta["eval"] >= best_eval:
            best_eval = meta["eval"]
            agent.save(os.path.join(out_dir, "best.npz"), **meta)
        checkpoints.append({**meta, "path": os.path.relpath(path, out_dir).replace("\\", "/")})
        return meta

    obs = env.reset()
    t0 = time.time()
    checkpoint(0, args.eps_start)
    for step in range(1, args.steps + 1):
        eps = max(args.eps_end, args.eps_start - (args.eps_start - args.eps_end) * step / args.eps_decay)
        a, _ = agent.act(obs, eps)
        obs2, r, done, info = env.step(a)
        # done=1 ONLY on crash: a time-limit stop is not a real ending
        buf.add(obs, a, r, obs2, float(info["crashed"]))
        obs = obs2
        if done:
            episode_steps.append(info["steps"])
            obs = env.reset()

        if buf.size >= args.learn_start:
            recent_losses.append(agent.learn(buf.sample(args.batch))[0])
        if step % args.target_sync == 0:
            agent.sync_target()

        # dense snapshots early (that's where learning happens), sparse later
        early = step <= args.early_until and step % args.early_every == 0
        if early or step % args.ckpt_every == 0:
            m = checkpoint(step, eps)
            loss = float(np.mean(recent_losses)) if recent_losses else float("nan")
            recent_losses.clear()
            sps = step / (time.time() - t0)
            print(f"step {step:>7,} | episode {m['episode']:>5} | eps {eps:4.2f} | "
                  f"train avg50 {m['train_avg']:6.1f} | test {m['eval']:6.1f} | loss {loss:.4f} | {sps:,.0f} steps/s")

    agent.save(os.path.join(out_dir, "final.npz"), **{k: v for k, v in checkpoints[-1].items() if k != "path"})
    history = dict(
        config={**vars(args), **env_kwargs},
        random_baseline=rand_base,
        episode_steps=episode_steps,
        checkpoints=checkpoints,
    )
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f)
    plot(history, os.path.join(out_dir, "learning_curve.png"))

    print(f"\ndone in {time.time() - t0:.0f}s  |  episodes {len(episode_steps)}  |  best test {best_eval:.1f} / {args.max_ep_steps}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
