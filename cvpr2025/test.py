#!/usr/bin/env python3
# Time‑optimal right‑hand turn with obstacle avoidance
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple


class Point(NamedTuple):
    x: float
    y: float


# ---------- geometry ---------------------------------------------------------
START = Point(0.0, 0.0)          # heading +x
CORNER = Point(5.0, 0.0)         # turn centreline
GOAL   = Point(5.0, 5.0)         # end of manoeuvre

HALF_W   = 1.0                   # half‑road width
CLEAR    = 0.05                  # extra safety margin
V_MAX    = 1.0                   # speed (u/s)

OBS_C = Point(5.0, 1.5)          # hidden obstacle
OBS_R = 0.5

# ---------- helpers ----------------------------------------------------------
def arc_pts(r: float, n: int = 200) -> np.ndarray:
    θ = np.linspace(np.pi, np.pi/2, n)
    cx, cy = CORNER.x, CORNER.y
    return np.column_stack((cx + r*np.cos(θ), cy + r*np.sin(θ)))

def line_pts(p0: Point, p1: Point, n: int = 50) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return np.column_stack((p0.x + t*(p1.x-p0.x), p0.y + t*(p1.y-p0.y)))

def min_dist(pts: np.ndarray, c: Point) -> float:
    return np.hypot(pts[:, 0]-c.x, pts[:, 1]-c.y).min()

def path_len(r: float) -> float:
    return (CORNER.x - r - START.x) + (np.pi*r/2) + (GOAL.y - r)

def feasible(r: float) -> bool:
    if r <= CLEAR or r >= HALF_W - CLEAR:
        return False  # leaves roadway
    pts = np.vstack((
        line_pts(START, Point(CORNER.x - r, 0.0)),
        arc_pts(r, 80),
        line_pts(Point(CORNER.x, r), GOAL)
    ))
    return min_dist(pts, OBS_C) >= OBS_R + CLEAR

# ---------- search radius ----------------------------------------------------
cands = np.linspace(CLEAR+1e-3, HALF_W - CLEAR, 400)
feas  = [r for r in cands if feasible(r)]
if not feas:
    raise RuntimeError("No feasible path – widen road or move obstacle")

R_opt = min(feas, key=path_len)
T_opt = path_len(R_opt) / V_MAX

# ---------- generate optimal path -------------------------------------------
path = np.vstack((
    line_pts(START, Point(CORNER.x - R_opt, 0.0), 30),
    arc_pts(R_opt, 120),
    line_pts(Point(CORNER.x, R_opt), GOAL, 30)
))

# ---------- plotting ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(path[:, 0], path[:, 1], 'b', lw=2, label='Optimal path')
ax.scatter([START.x, GOAL.x], [START.y, GOAL.y], c='g', zorder=3)

# road boundaries
ax.plot([0, CORNER.x], [HALF_W, HALF_W], 'k--', alpha=.3)
ax.plot([0, CORNER.x], [-HALF_W, -HALF_W], 'k--', alpha=.3)
ax.plot([CORNER.x - HALF_W, CORNER.x - HALF_W], [0, GOAL.y], 'k--', alpha=.3)
ax.plot([CORNER.x + HALF_W, CORNER.x + HALF_W], [0, GOAL.y], 'k--', alpha=.3)

# obstacle
obs = plt.Circle((OBS_C.x, OBS_C.y), OBS_R, color='r', alpha=.4, label='Obstacle')
ax.add_patch(obs)

ax.set_aspect('equal')
ax.set_xlim(-1, CORNER.x + HALF_W + 1)
ax.set_ylim(-HALF_W - 1, GOAL.y + 1)
ax.set_title(f'Time‑optimal path  (t ≈ {T_opt:.2f}s)')
ax.legend()
plt.tight_layout()
plt.show()
