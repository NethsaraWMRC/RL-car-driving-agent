"""
Watch the DQN agent learn: untrained -> learning stages -> final trained brain.
Every stage plays the SAME road (same seed), so the only thing that changes is the brain.

    python watch.py                              # montage from runs/dqn
    python watch.py --video lane_dodge.mp4       # also record an MP4 for posts
    python watch.py --ckpt runs/dqn/final.npz    # play just one brain
    python watch.py --middle 3 --final-episodes 3 --max-watch-steps 150 --tick 120

Keys:  SPACE skip   UP/DOWN speed   P screenshot   ESC quit
"""
import argparse
import json
import os

import numpy as np
import pygame

from lane_env import LaneEnv
from nn_numpy import MLP
from renderer import ACCENT, DANGER, GOOD, TEXT, TEXT_DIM, Renderer

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_W = 300


class Quit(Exception):
    pass


class VideoRecorder:
    """H.264 MP4 via bundled ffmpeg (small, works on social media). Falls back to OpenCV mp4v."""

    def __init__(self, path: str, size: tuple[int, int], fps: int):
        self.path = path
        try:
            import imageio_ffmpeg
            self.gen = imageio_ffmpeg.write_frames(
                path, size, fps=fps, codec="libx264", pix_fmt_out="yuv420p", quality=None,
                macro_block_size=8, output_params=["-crf", "20", "-preset", "medium"],
            )
            self.gen.send(None)
            self.kind, self.codec = "ffmpeg", "H.264"
        except ImportError:
            import cv2
            self.cv2 = cv2
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            self.kind, self.codec = "cv2", "mp4v"

    def add(self, surface: pygame.Surface) -> None:
        rgb = np.ascontiguousarray(pygame.surfarray.array3d(surface).transpose(1, 0, 2))
        if self.kind == "ffmpeg":
            self.gen.send(rgb)
        else:
            self.writer.write(self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.kind == "ffmpeg":
            self.gen.close()
        else:
            self.writer.release()


def road_survival(net: MLP, env: LaneEnv, seed: int, cap: int) -> int:
    """How far this brain gets on the watch road (greedy, no window)."""
    env.reset(seed=seed)
    while not env.crashed and env.steps < cap:
        env.step(int(np.argmax(net.predict(env.get_obs()[None])[0])))
    return env.steps


def select_stages(ckpts, n_middle, score) -> list[dict]:
    """
    start (untrained) + middle checkpoints that each got FURTHER on the watch road
    than any earlier one (up to the first that survives it) + final brain.
    """
    start, final = ckpts[0], ckpts[-1]
    records, top = [], score(start)
    for c in ckpts[1:-1]:
        s = score(c)
        if s > top:
            records.append((c, s))
            top = s
        if s >= score.cap:
            break
    return [start, *[c for c, _ in records[-n_middle:]], final] if n_middle else [start, final]


def main() -> None:
    ap = argparse.ArgumentParser(description="Watch the DQN agent")
    ap.add_argument("--run", default="dqn", help="folder under runs/")
    ap.add_argument("--ckpt", help="play only this .npz brain")
    ap.add_argument("--episodes", type=int, default=3, help="episodes for --ckpt mode")
    ap.add_argument("--middle", type=int, default=3, help="number of middle stages")
    ap.add_argument("--final-episodes", type=int, default=3)
    ap.add_argument("--max-watch-steps", type=int, default=150, help="stop an episode after this many steps")
    ap.add_argument("--tick", type=float, default=120, help="ms per step")
    ap.add_argument("--seed", type=int, default=7, help="road seed (same for every stage)")
    ap.add_argument("--video", help="record to this .mp4 file")
    ap.add_argument("--fps", type=int, default=30, help="video frame rate")
    args = ap.parse_args()

    run_dir = os.path.join(HERE, "runs", args.run)
    history = None
    hist_path = os.path.join(run_dir, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)

    cfg = (history or {}).get("config", {})
    test_cap = cfg.get("max_ep_steps", 1000)
    env = LaneEnv(cols=cfg.get("cols", 5), rows=cfg.get("rows", 9), spawn_prob=cfg.get("spawn", 0.6), max_steps=None)

    # ---------------------------------------------------------- pick brains
    stages = []
    if args.ckpt:
        net, meta = MLP.load(args.ckpt)
        stages.append(dict(net=net, meta=meta, title="AGENT", color=GOOD, n_eps=args.episodes))
    else:
        if history is None:
            raise SystemExit(f"no training run found at {run_dir} - run train.py first")

        cache = {}

        def score(c):
            if c["path"] not in cache:
                cache[c["path"]] = road_survival(MLP.load(os.path.join(run_dir, c["path"]))[0],
                                                 env, args.seed, args.max_watch_steps)
            return cache[c["path"]]
        score.cap = args.max_watch_steps

        picked = select_stages(history["checkpoints"], args.middle, score)
        for i, c in enumerate(picked):
            net, meta = MLP.load(os.path.join(run_dir, c["path"]))
            if i == 0:
                title, color, n = "UNTRAINED", DANGER, 1
            elif i == len(picked) - 1:
                title, color, n = "TRAINED", GOOD, args.final_episodes
            else:
                title, color, n = "LEARNING", ACCENT, 1
            stages.append(dict(net=net, meta=meta, title=title, color=color, n_eps=n))

    # test-score curve for the panel: zoom on the part where learning happens
    curve_xy, x_max = None, None
    if history:
        ck = history["checkpoints"]
        best = max(c["eval"] for c in ck)
        first_best = next(c["step"] for c in ck if c["eval"] >= best)
        x_max = min(ck[-1]["step"], max(3 * first_best, 5_000))
        curve_xy = [(c["step"], c["eval"]) for c in ck if c["step"] <= x_max]

    # ---------------------------------------------------------- window
    ui = Renderer(env.cols, env.rows, "Lane Dodge - DQN agent", panel_w=PANEL_W)
    clock = pygame.time.Clock()
    rec = VideoRecorder(args.video, (ui.w, ui.h), args.fps) if args.video else None
    fps = args.fps if rec else 60
    speed = {"tick": args.tick}

    def pump() -> bool:
        skip = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise Quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise Quit
                if event.key == pygame.K_SPACE:
                    skip = True
                elif event.key == pygame.K_UP:
                    speed["tick"] = max(30.0, speed["tick"] * 0.75)
                elif event.key == pygame.K_DOWN:
                    speed["tick"] = min(600.0, speed["tick"] / 0.75)
                elif event.key == pygame.K_p:
                    print("screenshot:", ui.screenshot(os.path.join(HERE, "screenshots")))
        return skip

    def frame(t, prev_car, hud, action, overlay, panel) -> bool:
        ui.draw(env, t, prev_car, hud, action, overlay, panel)
        if rec:
            rec.add(ui.screen)
        clock.tick(fps)
        return pump()

    def hold(seconds, prev_car, hud, action, overlay, panel) -> None:
        for _ in range(int(seconds * fps)):
            if frame(1.0, prev_car, hud, action, overlay, panel):
                return

    # ---------------------------------------------------------- play
    results = []
    try:
        for si, stg in enumerate(stages, 1):
            net, meta = stg["net"], stg["meta"]
            base_panel = {
                "title": stg["title"], "color": stg["color"],
                "subtitle": f"stage {si} of {len(stages)}",
                "stats": [
                    ("TRAINED", f"{meta.get('step', 0):,} steps"),
                    ("EPISODES SEEN", f"{meta.get('episode', 0):,}"),
                    ("EXPLORE  ε", f"{meta.get('epsilon', 0):.2f}"),
                    ("TEST SCORE", f"{meta.get('eval', 0):.0f} / {test_cap}"),
                ],
                "curve_xy": curve_xy, "x_max": x_max, "y_max": test_cap, "marker_x": meta.get("step", 0),
            }
            for e in range(stg["n_eps"]):
                env.reset(seed=args.seed + e)
                prev_car, action = env.car, None
                q = net.predict(env.get_obs()[None])[0]
                hud = lambda: [("SCORE", str(env.dodged), ACCENT), ("STEPS", str(env.steps), TEXT),
                               ("STAGE", f"{si}/{len(stages)}", TEXT)]

                if e == 0:
                    card = {"title": stg["title"], "color": stg["color"], "lines": [
                        (f"after {meta.get('episode', 0):,} episodes", TEXT),
                        (f"{meta.get('step', 0):,} training steps", TEXT_DIM),
                        ("SPACE skip   ↑↓ speed", TEXT_DIM),
                    ]}
                    if stg["n_eps"] > 1:
                        card["lines"].insert(2, (f"run 1 of {stg['n_eps']}", TEXT_DIM))
                    hold(2.0, prev_car, hud(), action, card, {**base_panel, "q": list(q)})
                else:
                    card = {"title": f"RUN {e + 1} / {stg['n_eps']}", "color": stg["color"], "lines": [("new road", TEXT_DIM)]}
                    hold(0.9, prev_car, hud(), action, card, {**base_panel, "q": list(q)})

                last, result = pygame.time.get_ticks(), None
                while result is None:
                    now = pygame.time.get_ticks()
                    if now - last >= speed["tick"]:
                        q = net.predict(env.get_obs()[None])[0]   # the brain decides
                        action = int(np.argmax(q))
                        prev_car = env.car
                        env.step(action)
                        last = now
                        if env.crashed:
                            result = "crash"
                        elif env.steps >= args.max_watch_steps:
                            result = "alive"
                    t = (now - last) / speed["tick"]
                    if frame(t, prev_car, hud(), action, None, {**base_panel, "q": list(q)}) and result is None:
                        result = "skip"

                if result == "skip":
                    results.append((stg["title"], env.steps, "skipped"))
                    continue
                start = pygame.time.get_ticks()  # finish the last slide animation
                while (pygame.time.get_ticks() - start) < speed["tick"]:
                    t = (pygame.time.get_ticks() - start) / speed["tick"]
                    frame(t, prev_car, hud(), action, None, {**base_panel, "q": list(q)})
                if result == "crash":
                    ui.explode(env.car)
                    end = {"title": "CRASH", "color": DANGER, "lines": [
                        (f"survived {env.steps} steps", TEXT), (f"dodged {env.dodged} blocks", TEXT_DIM)]}
                else:
                    end = {"title": "STILL ALIVE", "color": GOOD, "lines": [
                        (f"{env.steps}+ steps, no crash", TEXT), (f"dodged {env.dodged} blocks", TEXT_DIM)]}
                results.append((stg["title"], env.steps, result))
                hold(1.6, prev_car, hud(), action, end, {**base_panel, "q": list(q)})

        if len(results) > 1:  # summary card
            lines = [("", TEXT)]
            for title, steps, res in results:
                tag = {"crash": "crashed", "alive": "no crash", "skipped": "skipped"}[res]
                lines.append((f"{title:<10} {steps:>4}{'+' if res == 'alive' else ''} steps  ({tag})",
                              GOOD if res == "alive" else DANGER if res == "crash" else TEXT_DIM))
            lines.append(("each stage's 1st run: same road, only the brain changed", TEXT_DIM))
            hold(5.0, env.car, [("SCORE", str(env.dodged), ACCENT)], None,
                 {"title": "RESULTS", "color": ACCENT, "lines": lines},
                 {"title": "SUMMARY", "subtitle": "test score vs training", "curve_xy": curve_xy,
                  "x_max": x_max, "y_max": test_cap})
    except Quit:
        pass
    finally:
        if rec:
            rec.close()
            print(f"video saved -> {os.path.abspath(args.video)}  ({rec.codec})")
        pygame.quit()

    for title, steps, res in results:
        print(f"{title:<10} {steps:>5} steps  {res}")


if __name__ == "__main__":
    main()
