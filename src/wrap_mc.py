#!/usr/bin/env python3
"""Wrap-split ensemble: generate full rods, periodically fold overflowing
axis pieces back into the cube, and treat EACH resulting segment as an
independent medium (consistent with the attachment listing).
Volume fraction is counted on the parent rods: N = round(phi * V / V_A).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from simulate import (
    BOX, HALF, H_A, THRESH_AA, THRESH_AE, THRESH_AB, THRESH_BB, THRESH_BE,
    R_B, VOL_A, VOL_CUBE, n_from_phi_A, phi_from_n_A,
    _seg_seg_dist2, _seg_plane_x, _pt_seg_dist2, _mix_percolates,
    sample_spheres, OUT_DIR, COST_ONE_A, COST_ONE_B,
)


def wrap_split_segments(c: np.ndarray, u: np.ndarray):
    p = c - 0.5 * H_A * u
    q = c + 0.5 * H_A * u
    d = q - p
    ts = [0.0, 1.0]
    for ax in range(3):
        if abs(d[ax]) < 1e-15:
            continue
        for face in (-HALF, HALF):
            t = (face - p[ax]) / d[ax]
            if 0.0 < t < 1.0:
                ts.append(float(t))
    ts = np.unique(np.round(np.asarray(ts), 12))
    segs = []
    for t0, t1 in zip(ts[:-1], ts[1:]):
        a = p + t0 * d
        b = p + t1 * d
        mid = 0.5 * (a + b)
        shift = np.zeros(3)
        for ax in range(3):
            if mid[ax] > HALF:
                shift[ax] = -BOX
            elif mid[ax] < -HALF:
                shift[ax] = BOX
        a2 = np.clip(a + shift, -HALF, HALF)
        b2 = np.clip(b + shift, -HALF, HALF)
        segs.append((a2, b2))
    return segs


def generate_wrap_rods(n_rod: int, rng: np.random.Generator):
    """Return P, Q (nseg,3) and rod_id (nseg,) mapping segment -> parent rod."""
    if n_rod <= 0:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,), dtype=np.int32))
    z = rng.uniform(-1.0, 1.0, size=n_rod)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_rod)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    u = np.column_stack((r * np.cos(phi), r * np.sin(phi), z))
    c = rng.uniform(-HALF, HALF, size=(n_rod, 3))
    Ps, Qs, ids = [], [], []
    for i in range(n_rod):
        for a, b in wrap_split_segments(c[i], u[i]):
            Ps.append(a)
            Qs.append(b)
            ids.append(i)
    return np.asarray(Ps), np.asarray(Qs), np.asarray(ids, dtype=np.int32)


def incremental_crit_wrap(nmax: int, rng: np.random.Generator) -> int:
    """Add parent rods one by one (all their segments). Return critical N_rod."""
    from simulate import UF, is_percolating_rods  # noqa
    # generate all rods' segments with parent ids 0..nmax-1
    P, Q, rid = generate_wrap_rods(nmax, rng)
    # group segments by rod
    buckets = [[] for _ in range(nmax)]
    for s, i in enumerate(rid):
        buckets[i].append(s)
    nseg = len(P)
    LEFT, RIGHT = nseg, nseg + 1
    parent = list(range(nseg + 2))
    rank = [0] * (nseg + 2)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    aa2 = THRESH_AA ** 2
    active = []
    for k in range(nmax):
        new = buckets[k]
        for s in new:
            if _seg_plane_x(P[s, 0], Q[s, 0], -HALF) <= THRESH_AE:
                union(s, LEFT)
            if _seg_plane_x(P[s, 0], Q[s, 0], HALF) <= THRESH_AE:
                union(s, RIGHT)
        # new-new
        for a in range(len(new)):
            sa = new[a]
            for b in range(a + 1, len(new)):
                sb = new[b]
                d2 = _seg_seg_dist2(
                    P[sa, 0], P[sa, 1], P[sa, 2], Q[sa, 0], Q[sa, 1], Q[sa, 2],
                    P[sb, 0], P[sb, 1], P[sb, 2], Q[sb, 0], Q[sb, 1], Q[sb, 2],
                )
                if d2 <= aa2:
                    union(sa, sb)
        # new-old
        for sa in new:
            for sb in active:
                d2 = _seg_seg_dist2(
                    P[sa, 0], P[sa, 1], P[sa, 2], Q[sa, 0], Q[sa, 1], Q[sa, 2],
                    P[sb, 0], P[sb, 1], P[sb, 2], Q[sb, 0], Q[sb, 1], Q[sb, 2],
                )
                if d2 <= aa2:
                    union(sa, sb)
        active.extend(new)
        if find(LEFT) == find(RIGHT):
            return k + 1
    return nmax + 1


def run_wrap_incremental(n_real: int, nmax: int, seed: int = 20260810):
    rng = np.random.default_rng(seed)
    crit = np.empty(n_real, dtype=np.int32)
    t0 = time.time()
    for r in range(n_real):
        crit[r] = incremental_crit_wrap(nmax, rng)
        if (r + 1) % 10 == 0 or r == 0:
            print(f"  wrap-inc {r+1}/{n_real}  t={time.time()-t0:.1f}s  last={crit[r]}",
                  flush=True)
    return crit


def p_mix_wrap(nA, nB, n_real, seed):
    """Rods wrap-split (segments independent) + fully contained spheres."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_real):
        P, Q, _ = generate_wrap_rods(nA, rng)
        C = sample_spheres(nB, rng)
        hits += _mix_percolates(
            P, Q, C, THRESH_AA ** 2, THRESH_AB ** 2, THRESH_BB ** 2,
            THRESH_AE, THRESH_BE,
        )
    p = hits / n_real
    se = (p * (1 - p) / n_real) ** 0.5
    return p, se, hits


if __name__ == "__main__":
    NMAX = 800
    NREAL = 250
    print(f"wrap incremental {NREAL} x {NMAX}", flush=True)
    t0 = time.time()
    crit = run_wrap_incremental(NREAL, NMAX, seed=20260810)
    print("elapsed", time.time() - t0)
    print("stats min/p10/med/mean/p90/max/never",
          crit.min(), np.quantile(crit, 0.1), np.median(crit), crit.mean(),
          np.quantile(crit, 0.9), crit.max(), int(np.sum(crit > NMAX)))
    Ns = np.arange(0, NMAX + 1)
    P = np.array([(crit <= n).mean() for n in Ns])
    se = np.sqrt(P * (1 - P) / NREAL)
    print("\nProblem 2 (wrap-independent)")
    p2 = {}
    for phi in (0.005, 0.006, 0.007, 0.010):
        n = n_from_phi_A(phi)
        if n > NMAX:
            print("  N>nmax", n)
            continue
        p2[f"{phi*100:.2f}%"] = {"N": n, "P": float(P[n]), "se": float(se[n])}
        print(f"  {phi*100:.2f}% N={n} P={P[n]:.4f} ± {1.96*se[n]:.4f}")
    first = None
    for k in range(40, 160):
        phi = k / 10000.0
        n = n_from_phi_A(phi)
        if n > NMAX:
            continue
        if P[n] >= 0.90 and first is None:
            first = (k / 100.0, n, float(P[n]), float(se[n]))
            print(f"P3 first 0.01%: {first}")
    np.save(os.path.join(OUT_DIR, "crit_wrap.npy"), crit)
    np.savez(os.path.join(OUT_DIR, "survival_wrap.npz"), Ns=Ns, P=P, se=se)
    out = {
        "nreal": NREAL, "nmax": NMAX,
        "crit_stats": {
            "min": int(crit.min()),
            "p10": float(np.quantile(crit, 0.1)),
            "median": float(np.median(crit)),
            "mean": float(float(crit.mean())),
            "p90": float(np.quantile(crit, 0.9)),
            "max": int(crit.max()),
            "never": int(np.sum(crit > NMAX)),
        },
        "problem2": p2,
        "problem3": {
            "pct": first[0], "N": first[1], "P": first[2], "se": first[3]
        } if first else None,
    }
    with open(os.path.join(OUT_DIR, "problem23_wrap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
