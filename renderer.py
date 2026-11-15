"""
Pygame drawing for Lane Dodge.
Used by play.py (you play) and later by the DQN "watch" script (agent plays).
"""
from __future__ import annotations

import math
import os
import random
import time

import pygame

from lane_env import LEFT, RIGHT, STAY

CELL = 72   # pixel size of one grid cell
SIDE = 70   # shoulder width on each side of the road
HUD = 64    # top bar height

# ---- palette -------------------------------------------------------------
BG = (11, 15, 25)
SHOULDER = (17, 24, 39)
SHOULDER_MARK = (30, 41, 59)
ROAD = (30, 36, 52)
LANE_DASH = (71, 85, 105)
CURB_A = (239, 68, 68)
CURB_B = (241, 245, 249)
CAR = (34, 211, 238)
CAR_DARK = (8, 145, 178)
CAR_GLASS = (12, 74, 110)
CAR_STRIPE = (165, 243, 252)
HEADLIGHT = (253, 224, 71)
TAILLIGHT = (239, 68, 68)
WHEEL = (15, 23, 42)
BLOCK = (249, 115, 22)
BLOCK_DARK = (154, 52, 18)
BLOCK_STRIPE = (254, 243, 199)
TEXT = (226, 232, 240)
TEXT_DIM = (148, 163, 184)
ACCENT = (250, 204, 21)
DANGER = (248, 113, 113)
GOOD = (74, 222, 128)
HUD_BG = (15, 20, 33)
PANEL_BG = (13, 18, 30)
HUD_LINE = (51, 65, 85)
ICON_OFF = (71, 85, 105)


def _font(size: int, bold: bool = True) -> pygame.font.Font:
    return pygame.font.SysFont("segoeui,arial,helvetica", size, bold=bold)


class Renderer:
    def __init__(self, cols: int, rows: int, title: str = "Lane Dodge", panel_w: int = 0):
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()
        pygame.display.set_caption(title)
        self.cols, self.rows = cols, rows
        self.road_w = cols * CELL
        self.game_w = self.road_w + 2 * SIDE
        self.w = self.game_w + panel_w          # optional stats panel on the right
        self.world_h = rows * CELL
        self.h = HUD + self.world_h
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.world = pygame.Surface((self.game_w, self.world_h))
        self.fx = pygame.Surface((self.game_w, self.world_h), pygame.SRCALPHA)

        self.f_big = _font(54)
        self.f_mid = _font(26)
        self.f_small = _font(18, bold=False)
        self.f_hud = _font(22)
        self.f_label = _font(12)

        self.car_img = self._make_car()
        self.car_wreck = self.car_img.copy()
        self.car_wreck.fill((150, 70, 70, 255), special_flags=pygame.BLEND_RGBA_MULT)
        self.block_img = self._make_block()

        self._car_center = None
        self.particles: list[list] = []
        self.shake = 0.0
        self._last_frame = time.perf_counter()

    # ------------------------------------------------------------ sprites
    def _make_car(self) -> pygame.Surface:
        s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        bw, bh = CELL - 28, CELL - 12
        bx, by = (CELL - bw) // 2, 6
        for wy in (by + 8, by + bh - 22):  # wheels
            pygame.draw.rect(s, WHEEL, (bx - 5, wy, 9, 15), border_radius=3)
            pygame.draw.rect(s, WHEEL, (bx + bw - 4, wy, 9, 15), border_radius=3)
        pygame.draw.rect(s, CAR, (bx, by, bw, bh), border_radius=11)
        pygame.draw.rect(s, CAR_DARK, (bx, by, bw, bh), 2, border_radius=11)
        pygame.draw.rect(s, CAR_GLASS, (bx + 5, by + 14, bw - 10, 12), border_radius=4)  # windshield
        pygame.draw.rect(s, CAR_GLASS, (bx + 6, by + bh - 17, bw - 12, 8), border_radius=3)  # rear window
        pygame.draw.rect(s, CAR_STRIPE, (bx + bw // 2 - 3, by + 29, 6, bh - 49), border_radius=2)
        pygame.draw.rect(s, HEADLIGHT, (bx + 4, by + 2, 9, 5), border_radius=2)
        pygame.draw.rect(s, HEADLIGHT, (bx + bw - 13, by + 2, 9, 5), border_radius=2)
        pygame.draw.rect(s, TAILLIGHT, (bx + 4, by + bh - 5, 9, 4), border_radius=2)
        pygame.draw.rect(s, TAILLIGHT, (bx + bw - 13, by + bh - 5, 9, 4), border_radius=2)
        return s

    def _make_block(self) -> pygame.Surface:
        s = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        m = 9
        rect = pygame.Rect(m, m, CELL - 2 * m, CELL - 2 * m - 4)
        pygame.draw.rect(s, BLOCK_DARK, rect.move(0, 5), border_radius=9)  # 3D base
        pygame.draw.rect(s, BLOCK, rect, border_radius=9)
        stripes = pygame.Surface(rect.size, pygame.SRCALPHA)
        for i in range(-rect.h, rect.w + rect.h, 16):
            pygame.draw.polygon(
                stripes, (*BLOCK_STRIPE, 120),
                [(i, 0), (i + 8, 0), (i + 8 + rect.h, rect.h), (i + rect.h, rect.h)],
            )
        mask = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=9)
        stripes.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        s.blit(stripes, rect.topleft)
        pygame.draw.rect(s, BLOCK_DARK, rect, 2, border_radius=9)
        return s

    # ------------------------------------------------------------ effects
    def explode(self, car_col: int) -> None:
        """Burst at the car's last drawn position (falls back to the column centre)."""
        cx, cy = self._car_center or (SIDE + car_col * CELL + CELL / 2, (self.rows - 1) * CELL + CELL / 2)
        for _ in range(48):
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(80, 380)
            color = random.choice([BLOCK, ACCENT, CAR, (255, 255, 255), DANGER])
            life = random.uniform(0.5, 1.1)
            self.particles.append([cx, cy, math.cos(ang) * spd, math.sin(ang) * spd - 120, life, life, color])
        self.shake = 14.0

    def _update_particles(self, dt: float) -> None:
        alive = []
        for p in self.particles:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 520 * dt
            p[4] -= dt
            if p[4] > 0:
                alive.append(p)
        self.particles = alive

    # ------------------------------------------------------------ drawing
    def draw(self, env, t: float, prev_car: int, hud: list[tuple[str, str, tuple]],
             action: int | None = None, overlay: dict | None = None, panel: dict | None = None) -> None:
        """
        panel    : stats panel content (see _draw_panel), needs panel_w > 0
        t        : 0..1 animation progress since the last env.step (1 = settled)
        prev_car : car column before the last step (for smooth sliding)
        hud      : [(label, value, color), ...] shown left-to-right in the top bar
        action   : last action, highlighted in the top-right indicator
        overlay  : {"title": str, "color": rgb, "lines": [(text, rgb), ...]} or None
        """
        now = time.perf_counter()
        dt = min(0.05, now - self._last_frame)
        self._last_frame = now
        t = max(0.0, min(1.0, t))
        scroll = max(0.0, env.steps - 1 + t) * CELL

        w = self.world
        w.fill(SHOULDER)
        self.fx.fill((0, 0, 0, 0))

        # shoulders: scrolling marks
        off = scroll % CELL
        for sx in (SIDE // 2, SIDE + self.road_w + SIDE // 2):
            for i in range(-1, self.rows + 1):
                pygame.draw.rect(w, SHOULDER_MARK, (sx - 3, i * CELL + off, 6, CELL // 3), border_radius=3)

        # road + curbs
        pygame.draw.rect(w, ROAD, (SIDE, 0, self.road_w, self.world_h))
        seg = CELL // 2
        off = scroll % (2 * seg)
        for x in (SIDE - 8, SIDE + self.road_w):
            for i in range(-2, self.world_h // seg + 3):
                pygame.draw.rect(w, CURB_A if i % 2 == 0 else CURB_B, (x, i * seg + off - 2 * seg, 8, seg))

        # lane dashes
        off = scroll % CELL
        for c in range(1, self.cols):
            x = SIDE + c * CELL
            for i in range(-1, self.rows + 1):
                pygame.draw.rect(w, LANE_DASH, (x - 2, i * CELL + off, 4, int(CELL * 0.45)), border_radius=2)

        # blocks (slide from previous row to current row; frozen on a side crash)
        shifted = getattr(env, "last_shifted", True)
        yoff = (t - 1) if shifted else 0.0
        for r in range(self.rows):
            for c in range(self.cols):
                if env.grid[r, c]:
                    w.blit(self.block_img, (SIDE + c * CELL, (r + yoff) * CELL))
        if shifted:
            for c in range(self.cols):
                if env.exited[c]:
                    w.blit(self.block_img, (SIDE + c * CELL, (self.rows - 1 + t) * CELL))

        # car (slides + tilts toward its move; on a side crash it bumps half-way into the block)
        ct = 1 - (1 - min(1.0, t / 0.4)) ** 3
        move = env.car - prev_car
        reach = 0.45 if getattr(env, "crash_type", None) == "side" else 1.0
        car_x = SIDE + (prev_car + move * reach * ct) * CELL
        car_y = (self.rows - 1) * CELL
        wrecked = env.crashed and t >= 0.999
        img = self.car_wreck if wrecked else self.car_img
        angle = -move * 14 * math.sin(math.pi * ct)
        if abs(angle) > 0.5:
            img = pygame.transform.rotate(img, angle)
        rect = img.get_rect(center=(car_x + CELL / 2, car_y + CELL / 2))
        self._car_center = (car_x + CELL / 2 + (move * CELL * 0.3 if reach < 1 else 0), car_y + CELL / 2)

        if not wrecked:  # headlight beams + glow
            cx = car_x + CELL / 2
            pygame.draw.polygon(self.fx, (253, 224, 71, 28),
                                [(cx - 14, car_y + 8), (cx + 14, car_y + 8), (cx + 30, car_y - CELL * 1.2), (cx - 30, car_y - CELL * 1.2)])
            pygame.draw.ellipse(self.fx, (34, 211, 238, 45), (car_x + 6, car_y + 4, CELL - 12, CELL - 4))
        w.blit(self.fx, (0, 0))
        w.blit(img, rect)

        # particles
        self._update_particles(dt)
        for p in self.particles:
            radius = max(1, int(5 * p[4] / p[5]))
            pygame.draw.circle(w, p[6], (int(p[0]), int(p[1])), radius)

        # compose screen (with crash shake)
        ox = oy = 0
        if self.shake > 0.3:
            ox = random.uniform(-self.shake, self.shake)
            oy = random.uniform(-self.shake, self.shake)
            self.shake *= 0.88
        self.screen.fill(BG)
        self.screen.blit(w, (ox, HUD + oy))
        self._draw_hud(hud, action)
        if self.w > self.game_w:
            self._draw_panel(panel or {})
        if overlay:
            self._draw_overlay(overlay)
        pygame.display.flip()

    def _draw_hud(self, hud, action) -> None:
        pygame.draw.rect(self.screen, HUD_BG, (0, 0, self.game_w, HUD))
        pygame.draw.line(self.screen, HUD_LINE, (0, HUD - 1), (self.game_w, HUD - 1), 2)
        x = 16
        for label, value, color in hud:
            self.screen.blit(self.f_label.render(label, True, TEXT_DIM), (x, 9))
            self.screen.blit(self.f_hud.render(value, True, color), (x, 24))
            x += 96

        # action indicator  ◀ ● ▶
        size, gap = 18, 8
        right = self.game_w - 16
        xs = [right - 3 * size - 2 * gap, right - 2 * size - gap, right - size]
        self.screen.blit(self.f_label.render("ACTION", True, TEXT_DIM), (xs[0], 9))
        y0 = 30
        for a, x0 in zip((LEFT, STAY, RIGHT), xs):
            col = ACCENT if action == a else ICON_OFF
            if a == LEFT:
                pygame.draw.polygon(self.screen, col, [(x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size / 2)])
            elif a == RIGHT:
                pygame.draw.polygon(self.screen, col, [(x0, y0), (x0, y0 + size), (x0 + size, y0 + size / 2)])
            else:
                pygame.draw.circle(self.screen, col, (x0 + size / 2, y0 + size / 2), size / 2 - 2)

    def _draw_overlay(self, overlay: dict) -> None:
        dim = pygame.Surface((self.game_w, self.world_h), pygame.SRCALPHA)
        dim.fill((8, 12, 20, 170))
        self.screen.blit(dim, (0, HUD))

        title = self.f_big.render(overlay["title"], True, overlay.get("color", ACCENT))
        lines = [(self.f_small if i else self.f_mid).render(txt, True, col)
                 for i, (txt, col) in enumerate(overlay.get("lines", []))]
        pad = 28
        panel_w = max([title.get_width()] + [s.get_width() for s in lines]) + 2 * pad
        panel_h = title.get_height() + sum(s.get_height() + 10 for s in lines) + 2 * pad
        panel = pygame.Rect(0, 0, min(panel_w, self.game_w - 24), panel_h)
        panel.center = (self.game_w // 2, HUD + self.world_h // 2)
        pygame.draw.rect(self.screen, (15, 23, 42), panel, border_radius=16)
        pygame.draw.rect(self.screen, overlay.get("color", ACCENT), panel, 2, border_radius=16)

        y = panel.top + pad
        self.screen.blit(title, title.get_rect(midtop=(panel.centerx, y)))
        y += title.get_height() + 6
        for s in lines:
            self.screen.blit(s, s.get_rect(midtop=(panel.centerx, y)))
            y += s.get_height() + 10

    def _draw_panel(self, p: dict) -> None:
        """
        p = {"title": str, "color": rgb, "subtitle": str,
             "stats": [(label, value), ...],
             "q": [q_left, q_stay, q_right] or None, "action": int,
             "curve": [floats], "curve_max": float, "marker": index into curve}
        """
        x0 = self.game_w
        pw = self.w - x0
        pygame.draw.rect(self.screen, PANEL_BG, (x0, 0, pw, self.h))
        pygame.draw.line(self.screen, HUD_LINE, (x0, 0), (x0, self.h), 2)
        x, y, inner = x0 + 20, 14, pw - 40

        if p.get("title"):
            s = self.f_mid.render(p["title"], True, p.get("color", ACCENT))
            self.screen.blit(s, (x, y))
            y += s.get_height()
        if p.get("subtitle"):
            s = self.f_small.render(p["subtitle"], True, TEXT_DIM)
            self.screen.blit(s, (x, y))
            y += s.get_height()
        y += 12

        for label, value in p.get("stats", []):
            self.screen.blit(self.f_label.render(label, True, TEXT_DIM), (x, y + 6))
            v = self.f_hud.render(value, True, TEXT)
            self.screen.blit(v, (x + inner - v.get_width(), y))
            y += 30
        y += 14

        # Q-value bars, zero in the middle: right = good, left = bad
        self.screen.blit(self.f_label.render("Q-VALUES  (the agent's opinion)", True, TEXT_DIM), (x, y))
        y += 22
        q = p.get("q")
        name_w, val_w = 52, 50
        bar_x, bar_w = x + name_w, inner - name_w - val_w
        mid = bar_x + bar_w // 2
        best = max(range(3), key=lambda i: q[i]) if q is not None else None
        scale = max(1e-6, max(abs(v) for v in q)) if q is not None else 1.0
        for i, name in enumerate(("LEFT", "STAY", "RIGHT")):
            chosen = best == i
            self.screen.blit(self.f_label.render(name, True, ACCENT if chosen else TEXT_DIM), (x, y + 3))
            pygame.draw.rect(self.screen, (30, 41, 59), (bar_x, y, bar_w, 18), border_radius=4)
            if q is not None:
                length = int(abs(q[i]) / scale * (bar_w / 2 - 2))
                col = ACCENT if chosen else (DANGER if q[i] < 0 else CAR)
                rect = (mid, y + 2, length, 14) if q[i] >= 0 else (mid - length, y + 2, length, 14)
                pygame.draw.rect(self.screen, col, rect, border_radius=3)
                v = self.f_label.render(f"{q[i]:+.1f}", True, ACCENT if chosen else TEXT)
                self.screen.blit(v, (x + inner - v.get_width(), y + 3))
            pygame.draw.line(self.screen, TEXT_DIM, (mid, y - 2), (mid, y + 19), 1)
            y += 26
        y += 16

        # learning curve with "you are here" marker
        cxy = p.get("curve_xy")
        if cxy and len(cxy) > 1:
            self.screen.blit(self.f_label.render(p.get("curve_label", "TEST SCORE DURING TRAINING"), True, TEXT_DIM), (x, y))
            y += 20
            box = pygame.Rect(x, y, inner, max(80, self.h - y - 30))
            pygame.draw.rect(self.screen, (17, 24, 39), box, border_radius=6)
            x_max = p.get("x_max") or max(px for px, _ in cxy) or 1.0
            y_max = p.get("y_max") or max(py for _, py in cxy) or 1.0

            def to_px(xv, yv):
                return (box.left + 8 + min(xv, x_max) / x_max * (box.w - 16),
                        box.bottom - 8 - min(yv, y_max) / y_max * (box.h - 16))

            pygame.draw.line(self.screen, (51, 65, 85), to_px(0, y_max), to_px(x_max, y_max), 1)
            pts = [to_px(px, py) for px, py in cxy if px <= x_max]
            pygame.draw.lines(self.screen, CAR, False, pts, 2)
            for pt in pts:
                pygame.draw.circle(self.screen, CAR, (int(pt[0]), int(pt[1])), 2)

            mxv = p.get("marker_x")
            if mxv is not None:
                y_at = next((py for px, py in reversed(cxy) if px <= mxv), cxy[0][1])
                mx, my = to_px(mxv, y_at)
                for yy in range(box.top + 22, box.bottom - 4, 8):
                    pygame.draw.line(self.screen, ACCENT, (mx, yy), (mx, yy + 4), 2)
                pygame.draw.circle(self.screen, ACCENT, (int(mx), int(my)), 6)
                tag = self.f_label.render(("→ " if mxv > x_max else "") + f"{int(mxv):,} steps", True, ACCENT)
                tx = max(box.left + 4, min(mx - tag.get_width() / 2, box.right - tag.get_width() - 4))
                self.screen.blit(tag, (tx, box.top + 4))
            self.screen.blit(self.f_label.render(f"{y_max:.0f}", True, TEXT_DIM), (box.left + 6, to_px(0, y_max)[1] + 3))
            self.screen.blit(self.f_label.render("0", True, TEXT_DIM), (box.left, box.bottom + 4))
            e = self.f_label.render(f"{x_max:,.0f} training steps", True, TEXT_DIM)
            self.screen.blit(e, (box.right - e.get_width(), box.bottom + 4))

    def screenshot(self, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime("lane_dodge_%Y%m%d_%H%M%S.png"))
        pygame.image.save(self.screen, path)
        return path
