"""
Record short MP4 clips of the agent playing, rendered offline frame by frame, so every clip has
an exact length (for videos and posts). No window opens.

    python record_clips.py --out clips_dir

Each clip is described in CLIPS below: which brain plays, which road (seed), how long, and an
optional check the recording must pass (e.g. "crashes within 2.5 s"). For clips with a check the
script searches road seeds until one passes, so the result is deterministic and repeatable.
"""
import argparse
import json
import os
import subprocess
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from lane_env import LEFT, RIGHT, STAY, LaneEnv
from nn_numpy import MLP
from renderer import ACCENT, TEXT, Renderer

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30


class OldRulesEnv(LaneEnv):
    """The first version of the game: no side crash (the loophole the agent found)."""

    def step(self, action):
        beside = self.grid[-1].copy()
        self.grid[-1] = 0                     # hide the car row from the side-crash check
        out = super().step(action)
        self.exited = beside                  # the hidden row is the one leaving the screen
        self.dodged += int(beside.sum())
        return out


def brain(path):
    net, _ = MLP.load(os.path.join(HERE, path))
    return lambda env: int(np.argmax(net.predict(env.get_obs()[None])[0]))


def hack_policy(env):
    """Uses the loophole: dodges by slipping into a lane where a block is still beside the car."""
    c = env.car
    sides = [d for d in (-1, 1) if 0 <= c + d < env.cols]
    hack = [d for d in sides if env.grid[-1, c + d] and not env.grid[-2, c + d]]
    if hack:
        return LEFT if hack[0] < 0 else RIGHT
    if env.grid[-2, c]:
        free = [d for d in sides if not env.grid[-2, c + d]]
        return LEFT if free and free[0] < 0 else RIGHT
    return STAY


def side_crash_policy(play, trigger_step):
    """Plays normally, then after `trigger_step` drives into the first block beside the car."""
    def pol(env):
        c = env.car
        if env.steps >= trigger_step:
            for d in (1, -1):
                if 0 <= c + d < env.cols and env.grid[-1, c + d]:
                    return RIGHT if d > 0 else LEFT
        return play(env)
    return pol


def simulate(env_cls, policy, seed, warm, steps):
    """Dry run (no drawing): returns stats used by the clip checks."""
    env = env_cls(max_steps=None)
    env.reset(seed=seed)
    for _ in range(warm):
        env.step(policy(env))
        if env.crashed:
            return None
    moves, slips = 0, 0
    for i in range(steps):
        a = policy(env)
        if a != STAY:
            moves += 1
            target = env.car + (1 if a == RIGHT else -1)
            if 0 <= target < env.cols and env.grid[-1, target]:
                slips += 1
        env.step(a)
        if env.crashed:
            return {"crash_step": i + 1, "moves": moves, "slips": slips, "dodged": env.dodged, "type": env.crash_type}
    return {"crash_step": None, "moves": moves, "slips": slips, "dodged": env.dodged, "type": None}


def record(path, ui, env_cls, policy, seed, seconds, warm=0, fps_step=4):
    env = env_cls(max_steps=None)
    env.reset(seed=seed)
    for _ in range(warm):
        env.step(policy(env))
    total = int(round(seconds * FPS))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{ui.w}x{ui.h}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", path]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    prev_car, action, k = env.car, None, fps_step
    crash_frame = None
    ui.particles.clear()
    ui.shake = 0.0
    first_png = os.path.splitext(path)[0] + ".png"
    for f in range(total):
        if crash_frame is None and k >= fps_step:
            action = policy(env)
            prev_car = env.car
            env.step(action)
            k = 0
            if env.crashed:
                crash_frame = f
        t = min(1.0, k / fps_step)
        if crash_frame is not None and f - crash_frame == fps_step:
            ui.explode(env.car)
        ui._last_frame = time.perf_counter() - 1.0 / FPS   # fixed dt for particles
        hud = [("SCORE", str(env.dodged), ACCENT), ("STEPS", str(env.steps), TEXT)]
        ui.draw(env, t, prev_car, hud, action)
        if f == 0:
            pygame.image.save(ui.screen, first_png)
        ff.stdin.write(pygame.image.tostring(ui.screen, "RGB"))
        k += 1
    ff.stdin.close()
    ff.wait()
    return {"steps": env.steps, "dodged": env.dodged, "crashed": env.crashed, "crash_type": env.crash_type}


def find_seed(env_cls, policy, warm, steps, ok, start=0, tries=3000):
    for seed in range(start, start + tries):
        s = simulate(env_cls, policy, seed, warm, steps)
        if s is not None and ok(s):
            return seed, s
    raise RuntimeError("no seed satisfies the clip check")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", help="JSON file with clip durations {name: seconds}")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dur = json.load(open(args.spec)) if args.spec else {}
    D = lambda name, default: float(dur.get(name, default))

    final = brain("runs/dqn/final.npz")
    b0 = brain("runs/dqn/checkpoints/ckpt_0000000.npz")
    b2500 = brain("runs/dqn/checkpoints/ckpt_0002500.npz")
    b5000 = brain("runs/dqn/checkpoints/ckpt_0005000.npz")
    b6000 = brain("runs/dqn/checkpoints/ckpt_0006000.npz")
    steps_in = lambda sec, fs=4: int(sec * FPS / fs)

    # (name, env, policy, warm, check(stats), seconds, frames per step)
    clips = [
        ("hook", LaneEnv, final, 8, lambda s: s["crash_step"] is None and s["moves"] >= 6, D("hook", 12), 4),
        ("learn_0", LaneEnv, b0, 0, lambda s: s["crash_step"] is not None and 10 <= s["crash_step"] <= 15, D("learn_0", 2.7), 4),
        ("learn_2500", LaneEnv, b2500, 6,
         lambda s: s["crash_step"] is not None and 18 <= s["crash_step"] <= 24 and s["moves"] >= 3, D("learn_2500", 4.4), 4),
        ("learn_5000", LaneEnv, b5000, 8, lambda s: s["crash_step"] is None and s["moves"] >= 2, D("learn_5000", 2.3), 4),
        ("learn_6000", LaneEnv, b6000, 8, lambda s: s["crash_step"] is None and s["moves"] >= 4, D("learn_6000", 5.6), 4),
        ("loophole", OldRulesEnv, hack_policy, 8, lambda s: s["crash_step"] is None and s["slips"] >= 5, D("loophole", 10.5), 5),
        ("fixed", LaneEnv, side_crash_policy(final, 11), 8,
         lambda s: s["crash_step"] is not None and s["type"] == "side" and 4 <= s["crash_step"] <= 6, D("fixed", 1.8), 4),
        ("final", LaneEnv, final, 8, lambda s: s["crash_step"] is None and s["moves"] >= 4, D("final", 5.5), 4),
    ]
    ui = Renderer(5, 9)
    report = {}
    for i, (name, env_cls, pol, warm, ok, sec, fs) in enumerate(clips):
        seed, stats = find_seed(env_cls, pol, warm, steps_in(sec, fs), ok, start=100 * (i + 1))
        out = record(os.path.join(args.out, f"{name}.mp4"), ui, env_cls, pol, seed, sec, warm, fs)
        report[name] = {"seed": seed, "seconds": sec, **stats, "check": stats}
        print(f"{name:<11} seed {seed:<5} {sec:5.2f}s  {out}")

    # still for "the next roads will be harder": wider, busier road
    hard = LaneEnv(cols=7, rows=9, spawn_prob=1.0, max_steps=None)
    hard.reset(seed=5)
    rng = np.random.default_rng(5)
    hard.grid[:] = 0
    for r in range(hard.rows - 1):
        cols = rng.choice(7, size=2 if r % 2 == 0 else 1, replace=False)
        hard.grid[r, cols] = 1
    hard.grid[-1] = 0
    hard.grid[-2, hard.car] = 0
    ui2 = Renderer(7, 9)
    ui2._last_frame = time.perf_counter() - 1.0 / FPS
    ui2.draw(hard, 1.0, hard.car, [("SCORE", "?", ACCENT), ("LEVEL", "2", TEXT)], None)
    pygame.image.save(ui2.screen, os.path.join(args.out, "harder.png"))
    json.dump(report, open(os.path.join(args.out, "clips.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
