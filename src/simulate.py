#!/usr/bin/env python3
"""
Conductive-media percolation in a 10000 nm cube.
Soft-core (penetrable) spherocylinders (medium A) and spheres (medium B).
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict

import numpy as np
from numba import njit
import openpyxl

# ---------------------------------------------------------------------------
# Physical constants (nm)
# ---------------------------------------------------------------------------
BOX = 10000.0
HALF = 5000.0
H_A = 5000.0
R_A = 30.0
R_B = 200.0
DELTA = 1.8
VOL_CUBE = BOX ** 3
VOL_A = np.pi * R_A ** 2 * H_A
VOL_B = 4.0 / 3.0 * np.pi * R_B ** 3
# cost: yuan / um^3 ;  1 um^3 = 1e9 nm^3
COST_A_PER_UM3 = 1.05
COST_B_PER_UM3 = 0.05
COST_ONE_A = VOL_A / 1e9 * COST_A_PER_UM3
COST_ONE_B = VOL_B / 1e9 * COST_B_PER_UM3

THRESH_AA = 2 * R_A + DELTA          # 61.8
THRESH_AE = R_A + DELTA              # 31.8
THRESH_BB = 2 * R_B + DELTA          # 401.8
THRESH_BE = R_B + DELTA              # 201.8
THRESH_AB = R_A + R_B + DELTA        # 231.8

DATA_XLSX = "/tmp/problem/2605（8月10日18：00）/附件.xlsx"
OUT_DIR = "/workspace/paper/results"
FIG_DIR = "/workspace/paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def n_from_phi_A(phi: float) -> int:
    return int(round(phi * VOL_CUBE / VOL_A))


def phi_from_n_A(n: int) -> float:
    return n * VOL_A / VOL_CUBE


def phi_from_n_B(n: int) -> float:
    return n * VOL_B / VOL_CUBE


# ---------------------------------------------------------------------------
# Geometry kernels (numba)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _seg_seg_dist2(p1x, p1y, p1z, q1x, q1y, q1z,
                   p2x, p2y, p2z, q2x, q2y, q2z):
    d1x = q1x - p1x
    d1y = q1y - p1y
    d1z = q1z - p1z
    d2x = q2x - p2x
    d2y = q2y - p2y
    d2z = q2z - p2z
    rx = p1x - p2x
    ry = p1y - p2y
    rz = p1z - p2z
    a = d1x * d1x + d1y * d1y + d1z * d1z
    e = d2x * d2x + d2y * d2y + d2z * d2z
    f = d2x * rx + d2y * ry + d2z * rz
    eps = 1e-18
    if a <= eps and e <= eps:
        dx = p1x - p2x
        dy = p1y - p2y
        dz = p1z - p2z
        return dx * dx + dy * dy + dz * dz
    if a <= eps:
        s = 0.0
        t = f / e
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
    else:
        c = d1x * rx + d1y * ry + d1z * rz
        if e <= eps:
            t = 0.0
            s = -c / a
            if s < 0.0:
                s = 0.0
            elif s > 1.0:
                s = 1.0
        else:
            b = d1x * d2x + d1y * d2y + d1z * d2z
            denom = a * e - b * b
            if denom > eps:
                s = (b * f - c * e) / denom
                if s < 0.0:
                    s = 0.0
                elif s > 1.0:
                    s = 1.0
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = -c / a
                if s < 0.0:
                    s = 0.0
                elif s > 1.0:
                    s = 1.0
            elif t > 1.0:
                t = 1.0
                s = (b - c) / a
                if s < 0.0:
                    s = 0.0
                elif s > 1.0:
                    s = 1.0
    cx = (p1x + s * d1x) - (p2x + t * d2x)
    cy = (p1y + s * d1y) - (p2y + t * d2y)
    cz = (p1z + s * d1z) - (p2z + t * d2z)
    return cx * cx + cy * cy + cz * cz


@njit(cache=True)
def _seg_plane_x(px, qx, x0):
    if (px - x0) * (qx - x0) <= 0.0:
        return 0.0
    d1 = px - x0
    d2 = qx - x0
    if d1 < 0.0:
        d1 = -d1
    if d2 < 0.0:
        d2 = -d2
    return d1 if d1 < d2 else d2


@njit(cache=True)
def _pt_seg_dist2(cx, cy, cz, px, py, pz, qx, qy, qz):
    dx = qx - px
    dy = qy - py
    dz = qz - pz
    L2 = dx * dx + dy * dy + dz * dz
    if L2 <= 1e-18:
        ex = cx - px
        ey = cy - py
        ez = cz - pz
        return ex * ex + ey * ey + ez * ez
    t = ((cx - px) * dx + (cy - py) * dy + (cz - pz) * dz) / L2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    ex = cx - (px + t * dx)
    ey = cy - (py + t * dy)
    ez = cz - (pz + t * dz)
    return ex * ex + ey * ey + ez * ez


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------
class UF:
    __slots__ = ("p", "r")

    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


# ---------------------------------------------------------------------------
# Generation: fully-contained isotropic rods
# ---------------------------------------------------------------------------
def sample_rods(n: int, rng: np.random.Generator):
    """Return (P, Q) each (n,3): endpoints of fully-contained cylinders."""
    if n <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    z = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    u = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))
    half_span = 0.5 * H_A * np.abs(u)  # (n,3)
    lo = -HALF + half_span
    hi = HALF - half_span
    c = rng.uniform(lo, hi)
    P = c - 0.5 * H_A * u
    Q = c + 0.5 * H_A * u
    return P, Q


def sample_spheres(n: int, rng: np.random.Generator):
    """Centers fully inside the cube (sphere body may kiss the face)."""
    if n <= 0:
        return np.zeros((0, 3))
    # keep the whole ball inside: center in [-HALF+R_B, HALF-R_B]
    lo = -HALF + R_B
    hi = HALF - R_B
    return rng.uniform(lo, hi, size=(n, 3))


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------
@njit(cache=True)
def _rod_contacts(P, Q, thresh2, ae):
    """Return (i_idx, j_idx) of rod-rod contacts and left/right masks."""
    n = P.shape[0]
    # upper bound n*(n-1)/2, we collect in lists via boolean then extract
    # first count
    cnt = 0
    left = np.zeros(n, dtype=np.uint8)
    right = np.zeros(n, dtype=np.uint8)
    for i in range(n):
        if _seg_plane_x(P[i, 0], Q[i, 0], -HALF) <= ae:
            left[i] = 1
        if _seg_plane_x(P[i, 0], Q[i, 0], HALF) <= ae:
            right[i] = 1
        for j in range(i + 1, n):
            d2 = _seg_seg_dist2(
                P[i, 0], P[i, 1], P[i, 2], Q[i, 0], Q[i, 1], Q[i, 2],
                P[j, 0], P[j, 1], P[j, 2], Q[j, 0], Q[j, 1], Q[j, 2],
            )
            if d2 <= thresh2:
                cnt += 1
    ii = np.empty(cnt, dtype=np.int32)
    jj = np.empty(cnt, dtype=np.int32)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _seg_seg_dist2(
                P[i, 0], P[i, 1], P[i, 2], Q[i, 0], Q[i, 1], Q[i, 2],
                P[j, 0], P[j, 1], P[j, 2], Q[j, 0], Q[j, 1], Q[j, 2],
            )
            if d2 <= thresh2:
                ii[k] = i
                jj[k] = j
                k += 1
    return ii, jj, left, right


@njit(cache=True)
def _mix_contacts(P, Q, C, aa2, ab2, bb2, ae, be):
    nA = P.shape[0]
    nB = C.shape[0]
    leftA = np.zeros(nA, dtype=np.uint8)
    rightA = np.zeros(nA, dtype=np.uint8)
    leftB = np.zeros(nB, dtype=np.uint8)
    rightB = np.zeros(nB, dtype=np.uint8)
    cnt_aa = 0
    cnt_ab = 0
    cnt_bb = 0
    for i in range(nA):
        if _seg_plane_x(P[i, 0], Q[i, 0], -HALF) <= ae:
            leftA[i] = 1
        if _seg_plane_x(P[i, 0], Q[i, 0], HALF) <= ae:
            rightA[i] = 1
    for i in range(nB):
        if (C[i, 0] + HALF) <= be:
            leftB[i] = 1
        if (HALF - C[i, 0]) <= be:
            rightB[i] = 1
    for i in range(nA):
        for j in range(i + 1, nA):
            d2 = _seg_seg_dist2(
                P[i, 0], P[i, 1], P[i, 2], Q[i, 0], Q[i, 1], Q[i, 2],
                P[j, 0], P[j, 1], P[j, 2], Q[j, 0], Q[j, 1], Q[j, 2],
            )
            if d2 <= aa2:
                cnt_aa += 1
    for i in range(nA):
        for j in range(nB):
            d2 = _pt_seg_dist2(
                C[j, 0], C[j, 1], C[j, 2],
                P[i, 0], P[i, 1], P[i, 2], Q[i, 0], Q[i, 1], Q[i, 2],
            )
            if d2 <= ab2:
                cnt_ab += 1
    for i in range(nB):
        for j in range(i + 1, nB):
            dx = C[i, 0] - C[j, 0]
            dy = C[i, 1] - C[j, 1]
            dz = C[i, 2] - C[j, 2]
            if dx * dx + dy * dy + dz * dz <= bb2:
                cnt_bb += 1
    ia = np.empty(cnt_aa, dtype=np.int32)
    ja = np.empty(cnt_aa, dtype=np.int32)
    iab = np.empty(cnt_ab, dtype=np.int32)
    jab = np.empty(cnt_ab, dtype=np.int32)
    ib = np.empty(cnt_bb, dtype=np.int32)
    jb = np.empty(cnt_bb, dtype=np.int32)
    k = 0
    for i in range(nA):
        for j in range(i + 1, nA):
            d2 = _seg_seg_dist2(
                P[i, 0], P[i, 1], P[i, 2], Q[i, 0], Q[i, 1], Q[i, 2],
                P[j, 0], P[j, 1], P[j, 2], Q[j, 0], Q[j, 1], Q[j, 2],
            )
            if d2 <= aa2:
                ia[k] = i
                ja[k] = j
                k += 1
    k = 0
    for i in range(nA):
        for j in range(nB):
            d2 = _pt_seg_dist2(
                C[j, 0], C[j, 1], C[j, 2],
                P[i, 0], P[i, 1], P[i, 2], Q[i, 0], Q[i, 1], Q[i, 2],
            )
            if d2 <= ab2:
                iab[k] = i
                jab[k] = j
                k += 1
    k = 0
    for i in range(nB):
        for j in range(i + 1, nB):
            dx = C[i, 0] - C[j, 0]
            dy = C[i, 1] - C[j, 1]
            dz = C[i, 2] - C[j, 2]
            if dx * dx + dy * dy + dz * dz <= bb2:
                ib[k] = i
                jb[k] = j
                k += 1
    return ia, ja, iab, jab, ib, jb, leftA, rightA, leftB, rightB


def is_percolating_rods(P, Q) -> tuple[bool, dict]:
    n = P.shape[0]
    LEFT, RIGHT = n, n + 1
    uf = UF(n + 2)
    ii, jj, left, right = _rod_contacts(P, Q, THRESH_AA ** 2, THRESH_AE)
    nL = nR = 0
    for i in range(n):
        if left[i]:
            uf.union(i, LEFT)
            nL += 1
        if right[i]:
            uf.union(i, RIGHT)
            nR += 1
    for a, b in zip(ii, jj):
        uf.union(int(a), int(b))
    cond = uf.find(LEFT) == uf.find(RIGHT)
    comps = Counter(uf.find(i) for i in range(n))
    info = {
        "n": n,
        "n_left": nL,
        "n_right": nR,
        "n_aa": int(len(ii)),
        "conducting": bool(cond),
        "n_comp": int(len(comps)),
        "largest": int(max(comps.values()) if comps else 0),
    }
    return cond, info


def is_percolating_mix(P, Q, C) -> tuple[bool, dict]:
    nA = P.shape[0]
    nB = C.shape[0]
    # node order: A[0..nA), B[0..nB), LEFT, RIGHT
    LEFT = nA + nB
    RIGHT = nA + nB + 1
    uf = UF(nA + nB + 2)
    ia, ja, iab, jab, ib, jb, lA, rA, lB, rB = _mix_contacts(
        P, Q, C, THRESH_AA ** 2, THRESH_AB ** 2, THRESH_BB ** 2,
        THRESH_AE, THRESH_BE,
    )
    nL = nR = 0
    for i in range(nA):
        if lA[i]:
            uf.union(i, LEFT)
            nL += 1
        if rA[i]:
            uf.union(i, RIGHT)
            nR += 1
    for i in range(nB):
        if lB[i]:
            uf.union(nA + i, LEFT)
            nL += 1
        if rB[i]:
            uf.union(nA + i, RIGHT)
            nR += 1
    for a, b in zip(ia, ja):
        uf.union(int(a), int(b))
    for a, b in zip(iab, jab):
        uf.union(int(a), nA + int(b))
    for a, b in zip(ib, jb):
        uf.union(nA + int(a), nA + int(b))
    cond = uf.find(LEFT) == uf.find(RIGHT)
    info = {
        "nA": nA, "nB": nB,
        "n_left": nL, "n_right": nR,
        "n_aa": int(len(ia)), "n_ab": int(len(iab)), "n_bb": int(len(ib)),
        "conducting": bool(cond),
    }
    return cond, info


# ---------------------------------------------------------------------------
# Incremental Monte Carlo for rods
# ---------------------------------------------------------------------------
def incremental_critical_N(nmax: int, rng: np.random.Generator) -> int:
    """Generate nmax rods, add one by one, return smallest N that percolates
    (or nmax+1 if never)."""
    P, Q = sample_rods(nmax, rng)
    n = nmax
    LEFT, RIGHT = n, n + 1
    uf = UF(n + 2)
    # We need incremental edges: for k = 0..nmax-1, connect rod k with 0..k-1
    # and electrodes
    for k in range(nmax):
        if _seg_plane_x(P[k, 0], Q[k, 0], -HALF) <= THRESH_AE:
            uf.union(k, LEFT)
        if _seg_plane_x(P[k, 0], Q[k, 0], HALF) <= THRESH_AE:
            uf.union(k, RIGHT)
        pkx, pky, pkz = P[k, 0], P[k, 1], P[k, 2]
        qkx, qky, qkz = Q[k, 0], Q[k, 1], Q[k, 2]
        aa2 = THRESH_AA ** 2
        for j in range(k):
            d2 = _seg_seg_dist2(
                pkx, pky, pkz, qkx, qky, qkz,
                P[j, 0], P[j, 1], P[j, 2], Q[j, 0], Q[j, 1], Q[j, 2],
            )
            if d2 <= aa2:
                uf.union(k, j)
        if uf.find(LEFT) == uf.find(RIGHT):
            return k + 1
    return nmax + 1


@njit(cache=True)
def _inc_critical_N(P, Q, ae, aa2):
    n = P.shape[0]
    parent = np.arange(n + 2, dtype=np.int32)
    rank = np.zeros(n + 2, dtype=np.int32)
    LEFT = n
    RIGHT = n + 1

    def find(x):
        # path compression iterative
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for k in range(n):
        if _seg_plane_x(P[k, 0], Q[k, 0], -HALF) <= ae:
            union(k, LEFT)
        if _seg_plane_x(P[k, 0], Q[k, 0], HALF) <= ae:
            union(k, RIGHT)
        pkx, pky, pkz = P[k, 0], P[k, 1], P[k, 2]
        qkx, qky, qkz = Q[k, 0], Q[k, 1], Q[k, 2]
        for j in range(k):
            d2 = _seg_seg_dist2(
                pkx, pky, pkz, qkx, qky, qkz,
                P[j, 0], P[j, 1], P[j, 2], Q[j, 0], Q[j, 1], Q[j, 2],
            )
            if d2 <= aa2:
                union(k, j)
        if find(LEFT) == find(RIGHT):
            return k + 1
    return n + 1


def run_incremental_mc(n_real: int, nmax: int, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    crit = np.empty(n_real, dtype=np.int32)
    t0 = time.time()
    for r in range(n_real):
        P, Q = sample_rods(nmax, rng)
        crit[r] = _inc_critical_N(P, Q, THRESH_AE, THRESH_AA ** 2)
        if (r + 1) % 20 == 0 or r == 0:
            dt = time.time() - t0
            print(f"  inc MC {r+1}/{n_real}  elapsed={dt:.1f}s  last_Nc={crit[r]}",
                  flush=True)
    return crit


def survival_curve(crit: np.ndarray, nmax: int):
    """P(N) = fraction with crit <= N, for N=0..nmax."""
    Ns = np.arange(0, nmax + 1)
    P = np.array([(crit <= n).mean() for n in Ns])
    # Wilson / normal CI
    m = len(crit)
    se = np.sqrt(P * (1 - P) / m)
    return Ns, P, se


# ---------------------------------------------------------------------------
# Problem 1: given configurations
# ---------------------------------------------------------------------------
def load_group(sheet: str):
    wb = openpyxl.load_workbook(DATA_XLSX, data_only=True)
    ws = wb[sheet]
    rows = list(ws.iter_rows(values_only=True))[2:]
    P, Q = [], []
    for r in rows:
        P.append([float(r[0]), float(r[1]), float(r[2])])
        Q.append([float(r[3]), float(r[4]), float(r[5])])
    return np.asarray(P), np.asarray(Q)


def problem1():
    out = {}
    for name in ("组1", "组2", "组3"):
        P, Q = load_group(name)
        cond, info = is_percolating_rods(P, Q)
        # extra geometry stats
        L = np.linalg.norm(Q - P, axis=1)
        info.update({
            "L_min": float(L.min()),
            "L_mean": float(L.mean()),
            "L_max": float(L.max()),
            "n_full": int(np.sum(np.abs(L - 5000) < 1e-3)),
            "xmin": float(min(P[:, 0].min(), Q[:, 0].min())),
            "xmax": float(max(P[:, 0].max(), Q[:, 0].max())),
            "ymin": float(min(P[:, 1].min(), Q[:, 1].min())),
            "ymax": float(max(P[:, 1].max(), Q[:, 1].max())),
            "zmin": float(min(P[:, 2].min(), Q[:, 2].min())),
            "zmax": float(max(P[:, 2].max(), Q[:, 2].max())),
        })
        out[name] = info
        print(name, info)
    with open(os.path.join(OUT_DIR, "problem1.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


# ---------------------------------------------------------------------------
# Problem 4 helpers
# ---------------------------------------------------------------------------
@njit(cache=True)
def _mix_percolates(P, Q, C, aa2, ab2, bb2, ae, be):
    nA = P.shape[0]
    nB = C.shape[0]
    n = nA + nB
    parent = np.arange(n + 2, dtype=np.int32)
    rank = np.zeros(n + 2, dtype=np.int32)
    LEFT = n
    RIGHT = n + 1

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i in range(nA):
        if _seg_plane_x(P[i, 0], Q[i, 0], -HALF) <= ae:
            union(i, LEFT)
        if _seg_plane_x(P[i, 0], Q[i, 0], HALF) <= ae:
            union(i, RIGHT)
    for i in range(nB):
        if (C[i, 0] + HALF) <= be:
            union(nA + i, LEFT)
        if (HALF - C[i, 0]) <= be:
            union(nA + i, RIGHT)
    for i in range(nA):
        for j in range(i + 1, nA):
            d2 = _seg_seg_dist2(
                P[i, 0], P[i, 1], P[i, 2], Q[i, 0], Q[i, 1], Q[i, 2],
                P[j, 0], P[j, 1], P[j, 2], Q[j, 0], Q[j, 1], Q[j, 2],
            )
            if d2 <= aa2:
                union(i, j)
    for i in range(nA):
        for j in range(nB):
            d2 = _pt_seg_dist2(
                C[j, 0], C[j, 1], C[j, 2],
                P[i, 0], P[i, 1], P[i, 2], Q[i, 0], Q[i, 1], Q[i, 2],
            )
            if d2 <= ab2:
                union(i, nA + j)
    for i in range(nB):
        for j in range(i + 1, nB):
            dx = C[i, 0] - C[j, 0]
            dy = C[i, 1] - C[j, 1]
            dz = C[i, 2] - C[j, 2]
            if dx * dx + dy * dy + dz * dz <= bb2:
                union(nA + i, nA + j)
    return 1 if find(LEFT) == find(RIGHT) else 0


def estimate_p_mix(nA, nB, n_real, seed):
    rng = np.random.default_rng(seed)
    hits = 0
    for r in range(n_real):
        P, Q = sample_rods(nA, rng)
        C = sample_spheres(nB, rng)
        hits += _mix_percolates(
            P, Q, C, THRESH_AA ** 2, THRESH_AB ** 2, THRESH_BB ** 2,
            THRESH_AE, THRESH_BE,
        )
    p = hits / n_real
    se = np.sqrt(p * (1 - p) / n_real)
    return p, se, hits


if __name__ == "__main__":
    print("VOL_A", VOL_A, "VOL_B", VOL_B)
    print("N at 0.50/0.60/0.70/1.00%",
          [n_from_phi_A(x) for x in (0.005, 0.006, 0.007, 0.010)])
    print("cost one A/B", COST_ONE_A, COST_ONE_B)
    print("--- warmup numba ---")
    rng = np.random.default_rng(0)
    P, Q = sample_rods(20, rng)
    print(is_percolating_rods(P, Q))
    print("--- problem 1 ---")
    problem1()
