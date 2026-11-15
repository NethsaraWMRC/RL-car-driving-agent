"""
Play Lane Dodge yourself.

    python play.py              # default 5-lane road
    python play.py --cols 7     # wider road
    python play.py --no-speedup # constant speed

Controls:  LEFT/RIGHT or A/D = move   SPACE = start/restart   P = screenshot   ESC = quit

You play the EXACT same game the RL agent will play: the world advances one
step per tick, and the key you press decides that step's action (no key = STAY).
"""
import argparse
import os
import sys

import pygame

from lane_env import LEFT, RIGHT, STAY, LaneEnv
from renderer import ACCENT, DANGER, GOOD, TEXT, TEXT_DIM, Renderer

HERE = os.path.dirname(os.path.abspath(__file__))
BEST_FILE = os.path.join(HERE, "highscore.txt")
SHOTS_DIR = os.path.join(HERE, "screenshots")


def load_best() -> int:
    try:
        with open(BEST_FILE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return 0


def save_best(value: int) -> None:
    try:
        with open(BEST_FILE, "w") as f:
            f.write(str(value))
    except OSError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Play Lane Dodge")
    ap.add_argument("--cols", type=int, default=5, help="road width in lanes")
    ap.add_argument("--rows", type=int, default=9, help="visible rows")
    ap.add_argument("--tick", type=int, default=300, help="start ms per step")
    ap.add_argument("--min-tick", type=int, default=110, help="fastest ms per step")
    ap.add_argument("--ramp", type=float, default=1.0, help="ms faster per step (higher = speeds up quicker)")
    ap.add_argument("--no-speedup", action="store_true", help="keep speed constant")
    args = ap.parse_args()

    env = LaneEnv(cols=args.cols, rows=args.rows, max_steps=None)
    ui = Renderer(env.cols, env.rows)
    clock = pygame.time.Clock()

    best = load_best()
    state = "menu"            # menu -> play -> dying -> over -> play ...
    pending = None            # key tapped since last tick
    action = None
    prev_car = env.car
    last_tick = 0
    dying_start = 0
    new_best = False
    flash = ""

    def tick_ms() -> float:
        if args.no_speedup:
            return args.tick
        return max(args.min_tick, args.tick - env.steps * args.ramp)

    while True:
        clock.tick(60)
        now = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
            if event.key == pygame.K_p:
                path = ui.screenshot(SHOTS_DIR)
                flash = "saved " + os.path.basename(path)
                print("screenshot:", path)
            if state in ("menu", "over") and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                env.reset()
                prev_car, action, pending, new_best = env.car, None, None, False
                state, last_tick = "play", now
            elif state == "play":
                if event.key in (pygame.K_LEFT, pygame.K_a):
                    pending = LEFT
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    pending = RIGHT

        # ---- advance the world one step per tick
        if state == "play" and now - last_tick >= tick_ms():
            if pending is not None:
                action = pending
            else:
                keys = pygame.key.get_pressed()
                left = keys[pygame.K_LEFT] or keys[pygame.K_a]
                right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
                action = LEFT if left and not right else RIGHT if right and not left else STAY
            pending = None
            prev_car = env.car
            _, _, done, info = env.step(action)
            last_tick = now
            if info["crashed"]:
                state, dying_start = "dying", now

        # ---- animation progress
        if state == "play":
            t = (now - last_tick) / tick_ms()
        elif state == "dying":
            t = (now - dying_start) / tick_ms()
            if t >= 1:
                ui.explode(env.car)
                if env.dodged > best:
                    best, new_best = env.dodged, True
                    save_best(best)
                state = "over"
        else:
            t = 1.0

        # ---- overlay
        overlay = None
        if state == "menu":
            overlay = {
                "title": "LANE DODGE",
                "color": ACCENT,
                "lines": [
                    ("Dodge the blocks", TEXT),
                    ("← →  or  A D    move", TEXT_DIM),
                    ("SPACE    start", TEXT_DIM),
                    ("P  screenshot     ESC  quit", TEXT_DIM),
                ],
            }
        elif state == "over":
            overlay = {
                "title": "CRASH!",
                "color": DANGER,
                "lines": [
                    (f"Score {env.dodged}   •   {env.steps} steps", TEXT),
                    ("NEW BEST!" if new_best else f"Best {best}", GOOD if new_best else TEXT_DIM),
                    ("SPACE  play again", TEXT_DIM),
                ],
            }
            if flash:
                overlay["lines"].append((flash, TEXT_DIM))

        speed = args.tick / tick_ms()
        hud = [
            ("SCORE", str(env.dodged), ACCENT),
            ("BEST", str(max(best, env.dodged)), TEXT),
            ("SPEED", f"x{speed:.1f}", TEXT),
        ]
        ui.draw(env, t, prev_car, hud, action, overlay)


if __name__ == "__main__":
    main()
