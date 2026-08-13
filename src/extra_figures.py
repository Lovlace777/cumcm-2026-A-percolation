#!/usr/bin/env python3
"""Additional publication figures from existing MC / attachment data."""
from __future__ import annotations

import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from simulate import (
    HALF, R_A, DELTA, THRESH_AA, THRESH_AE, H_A, VOL_A, VOL_CUBE,
    load_group, UF, _rod_contacts, FIG_DIR, n_from_phi_A,
)

os.makedirs(FIG_DIR, exist_ok=True)

mpl.rcParams.update({
    "font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.9,
    "axes.labelsize": 11,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "mathtext.fontset": "dejavuserif",
})

NAVY = "#1B365D"
STEEL = "#3D5A80"
TEAL = "#1F7A6B"
GOLD = "#C4A35A"
CORAL = "#B23A48"
SLATE = "#5C6B73"
CREAM = "#F7F4EC"
SAND = "#E7E0D0"
LEFT_C = "#2E6B9E"
RIGHT_C = "#C45C26"
SPAN_C = "#1F7A6B"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print("saved", name)


def fig_lenhist():
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.15), sharey=True)
    colors = [STEEL, TEAL, NAVY]
    for ax, name, col in zip(axes, ("组1", "组2", "组3"), colors):
        P, Q = load_group(name)
        L = np.linalg.norm(Q - P, axis=1)
        bins = np.linspace(0, 5200, 14)
        ax.hist(L, bins=bins, color=col, alpha=0.82, edgecolor="white", linewidth=0.6)
        ax.axvline(5000, color=CORAL, ls="--", lw=1.1)
        ax.set_title(f"{name}  n={len(L)}", loc="left", color=NAVY, fontsize=10.5)
        ax.set_xlabel("段长 / nm")
        ax.set_xlim(0, 5300)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(0.97, 0.92, f"均值 {L.mean():.0f}\n满长 {(np.abs(L-5000)<1e-3).sum()}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=SLATE)
    axes[0].set_ylabel("频数")
    fig.tight_layout()
    save(fig, "fig_lenhist")


def _cluster_colors(P, Q):
    n = len(P)
    ii, jj, left, right = _rod_contacts(P, Q, THRESH_AA ** 2, THRESH_AE)
    uf = UF(n + 2)
    L, R = n, n + 1
    for i in range(n):
        if left[i]:
            uf.union(i, L)
        if right[i]:
            uf.union(i, R)
    for a, b in zip(ii, jj):
        uf.union(int(a), int(b))
    rootL, rootR = uf.find(L), uf.find(R)
    cols = []
    for i in range(n):
        r = uf.find(i)
        if r == rootL and rootL == rootR:
            cols.append(SPAN_C)
        elif r == rootL:
            cols.append(LEFT_C)
        elif r == rootR:
            cols.append(RIGHT_C)
        else:
            cols.append("#B0B7BE")
    return cols, list(zip(ii.tolist(), jj.tolist())), left, right, uf.find(L) == uf.find(R)


def fig_graph():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    for ax, name in zip(axes, ("组1", "组2")):
        P, Q = load_group(name)
        mid = 0.5 * (P + Q)
        cols, edges, left, right, on = _cluster_colors(P, Q)
        n = len(P)
        for i, j in edges:
            ax.plot([mid[i, 0], mid[j, 0]], [mid[i, 1], mid[j, 1]],
                    color=SAND, lw=1.15, zorder=1)
        ax.axvline(-HALF, color=LEFT_C, lw=2.2, alpha=0.85)
        ax.axvline(HALF, color=RIGHT_C, lw=2.2, alpha=0.85)
        ax.scatter(mid[:, 0], mid[:, 1], c=cols, s=42, zorder=3,
                   edgecolors="white", linewidths=0.5)
        for i in range(n):
            ax.text(mid[i, 0], mid[i, 1] + 35, str(i + 1), ha="center",
                    va="bottom", fontsize=6.5, color=SLATE)
        ax.set_xlabel("X / nm")
        ax.set_ylabel("Y / nm")
        ax.set_title(f"{name}  导通={'是' if on else '否'}", loc="left", color=NAVY)
        ax.set_xlim(-5600, 5600)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_graph")


def fig_ecdf():
    crit = np.load("/workspace/paper/results/crit_wrap.npy")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = np.sort(crit)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs * VOL_A / VOL_CUBE * 100, ys, where="post", color=NAVY, lw=1.8, label=r"经验分布 $\hat F(N_c)$")
    marks = [("中位", np.median(crit), TEAL), ("90% 分位", np.quantile(crit, 0.9), CORAL)]
    for lab, v, c in marks:
        phi = v * VOL_A / VOL_CUBE * 100
        ax.axvline(phi, color=c, ls="--", lw=1.15)
        ax.axhline(ys[np.searchsorted(xs, v, side="right") - 1], color=c, ls=":", lw=0.8, alpha=0.6)
        ax.text(phi + 0.012, 0.12 if lab[0] == "中" else 0.78, f"{lab}\n{phi:.2f}%",
                color=c, fontsize=8.5)
    for pct, p in [(0.50, 0.104), (0.60, 0.228), (0.70, 0.520), (1.00, 1.0)]:
        ax.scatter([pct], [p], s=36, zorder=4, color=GOLD, edgecolors=NAVY, linewidths=0.6)
    ax.set_xlabel(r"体积分数 $\varphi_A$ / %")
    ax.set_ylabel(r"累计概率 $P(N_c\leq N)$")
    ax.set_xlim(0.25, 1.08)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_ecdf")


def fig_mixheat():
    rows = json.load(open("/workspace/paper/results/problem4_stage2.json"))
    nAs = sorted({r["nA"] for r in rows})
    nBs = sorted({r["nB"] for r in rows})
    Z = np.full((len(nBs), len(nAs)), np.nan)
    C = np.full_like(Z, np.nan)
    for r in rows:
        i = nBs.index(r["nB"])
        j = nAs.index(r["nA"])
        Z[i, j] = r["P"]
        C[i, j] = r["cost"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.55))
    ax = axes[0]
    im = ax.imshow(Z, origin="lower", cmap="YlGnBu", vmin=0.35, vmax=1.0,
                   aspect="auto", extent=[-0.5, len(nAs) - 0.5, -0.5, len(nBs) - 0.5])
    ax.set_xticks(range(len(nAs)))
    ax.set_xticklabels(nAs, rotation=45, ha="right")
    ax.set_yticks(range(len(nBs)))
    ax.set_yticklabels(nBs)
    ax.set_xlabel(r"$N_A$")
    ax.set_ylabel(r"$N_B$")
    ax.set_title(r"(a) 导通概率 $\hat P_{\rm on}$", loc="left", color=NAVY)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    # mark P>=0.9
    for r in rows:
        if r["P"] >= 0.90:
            j = nAs.index(r["nA"])
            i = nBs.index(r["nB"])
            ax.scatter([j], [i], s=42, facecolors="none", edgecolors=CORAL, linewidths=1.4)

    ax = axes[1]
    im2 = ax.imshow(C, origin="lower", cmap="magma_r", aspect="auto",
                    extent=[-0.5, len(nAs) - 0.5, -0.5, len(nBs) - 0.5])
    ax.set_xticks(range(len(nAs)))
    ax.set_xticklabels(nAs, rotation=45, ha="right")
    ax.set_yticks(range(len(nBs)))
    ax.set_yticklabels(nBs)
    ax.set_xlabel(r"$N_A$")
    ax.set_ylabel(r"$N_B$")
    ax.set_title("(b) 填充成本 / 元", loc="left", color=NAVY)
    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.03)
    cb2.ax.tick_params(labelsize=8)
    # star cheapest feasible
    feas = [r for r in rows if r["P"] >= 0.90]
    best = min(feas, key=lambda r: r["cost"])
    ax.scatter([nAs.index(best["nA"])], [nBs.index(best["nB"])],
               marker="*", s=140, color=GOLD, edgecolors=NAVY, zorder=5)
    fig.tight_layout()
    save(fig, "fig_mixheat")


def fig_vex():
    L = 5000.0
    D = 2 * R_A + DELTA
    alphas = np.linspace(10, 120, 220)
    Ls = alphas * 2 * R_A
    Vex = 0.5 * np.pi * Ls ** 2 * D + 2 * np.pi * Ls * D ** 2 + 4 * np.pi / 3 * D ** 3
    VA = np.pi * R_A ** 2 * Ls
    phic = 1.4 * VA / Vex
    fig, ax = plt.subplots(figsize=(6.3, 3.55))
    ax.plot(alphas, phic * 100, color=NAVY, lw=1.9, label=r"$\varphi_c^{\mathrm{th}}=B_c V_A/\langle V_{\mathrm{ex}}\rangle$")
    ax.axvline(H_A / (2 * R_A), color=CORAL, ls="--", lw=1.15)
    ax.axhline(0.816, color=TEAL, ls=":", lw=1.1)
    ax.scatter([H_A / (2 * R_A)], [0.816], s=48, color=CORAL, zorder=4)
    ax.annotate(r"本题 $\alpha\approx 83.3$"+ "\n"+r"$\varphi_c^{\mathrm{th}}=0.82\%$",
                xy=(83.3, 0.816), xytext=(52, 1.55),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.0),
                color=CORAL, fontsize=9)
    ax.set_xlabel(r"长径比 $\alpha=L/(2R_A)$")
    ax.set_ylabel(r"理论阈值 $\varphi_c^{\mathrm{th}}$ / %")
    ax.set_xlim(10, 120)
    ax.set_ylim(0.4, 4.2)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_vex")


def fig_bootstrap():
    crit = np.load("/workspace/paper/results/crit_wrap.npy")
    rng = np.random.default_rng(20260810)
    Ms = np.array([30, 50, 80, 120, 160, 200, 250])
    Nstar = 601
    means, los, his = [], [], []
    for M in Ms:
        ps = []
        for _ in range(400):
            samp = rng.choice(crit, size=M, replace=True)
            ps.append(np.mean(samp <= Nstar))
        ps = np.array(ps)
        means.append(ps.mean())
        los.append(np.quantile(ps, 0.025))
        his.append(np.quantile(ps, 0.975))
    means, los, his = map(np.array, (means, los, his))
    fig, ax = plt.subplots(figsize=(6.3, 3.45))
    ax.fill_between(Ms, los, his, color=TEAL, alpha=0.18, label="Bootstrap 95% 带")
    ax.plot(Ms, means, color=NAVY, lw=1.8, marker="o", ms=5, label=r"$\hat P(601)$")
    ax.axhline(0.90, color=CORAL, ls="--", lw=1.1, label="约束 90%")
    ax.set_xlabel("样本量 $M$")
    ax.set_ylabel(r"$\hat P_{\mathrm{on}}(N_A=601)$")
    ax.set_ylim(0.78, 1.0)
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_bootstrap")


def fig_degree():
    P, Q = load_group("组3")
    ii, jj, left, right = _rod_contacts(P, Q, THRESH_AA ** 2, THRESH_AE)
    left = left.astype(bool)
    right = right.astype(bool)
    n = len(P)
    deg = np.zeros(n, dtype=int)
    for a, b in zip(ii, jj):
        deg[a] += 1
        deg[b] += 1
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.35))
    ax = axes[0]
    bins = np.arange(0, deg.max() + 2) - 0.5
    ax.hist(deg, bins=bins, color=NAVY, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("接触度 $k$")
    ax.set_ylabel("介质数")
    ax.set_title("(a) 组 3 接触度分布", loc="left", color=NAVY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax = axes[1]
    tags = ["仅触左", "仅触右", "双侧", "不触极"]
    vals = [
        int(np.sum(left & ~right)),
        int(np.sum(right & ~left)),
        int(np.sum(left & right)),
        int(np.sum(~left & ~right)),
    ]
    cols = [LEFT_C, RIGHT_C, SPAN_C, "#B0B7BE"]
    ax.bar(tags, vals, color=cols, width=0.62, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + 4, str(v), ha="center", fontsize=9, color=NAVY)
    ax.set_ylabel("介质数")
    ax.set_title("(b) 组 3 触极分类", loc="left", color=NAVY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_degree")


def fig_polar():
    """Theoretical electrode-hit probability vs |u_x| and wrap cartoon."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.45))
    ux = np.linspace(0, 1, 240)
    # rough geometric: a rod hits a given face if its half-projection exceeds remaining gap
    # P(hit left or right | ux) rises with |ux|
    p_hit = np.clip(np.abs(ux) * H_A / (2 * HALF) + (R_A + DELTA) / HALF * 0.15, 0, 1)
    ax = axes[0]
    ax.plot(ux, p_hit, color=NAVY, lw=1.9)
    ax.fill_between(ux, 0, p_hit, color=TEAL, alpha=0.15)
    ax.set_xlabel(r"$|u_x|$（取向在 $X$ 上的投影）")
    ax.set_ylabel("触达任一电极的几何倾向")
    ax.set_title("(a) 取向越沿 $X$，越易触极", loc="left", color=NAVY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    # polarization cascade schematic
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.3, 2.1), 1.5, 2.0, boxstyle="round,pad=0.08",
                                facecolor=LEFT_C, edgecolor=NAVY, alpha=0.85))
    ax.text(1.05, 3.1, "左电极", ha="center", va="center", color="white", fontsize=9)
    xs = [2.4, 4.3, 6.2]
    labs = ["A1\n已带电", "A2\n极化", "A3\n待判"]
    cols = [TEAL, GOLD, "#D7D2C8"]
    for x, lab, c in zip(xs, labs, cols):
        ax.add_patch(FancyBboxPatch((x, 2.25), 1.55, 1.7, boxstyle="round,pad=0.08",
                                    facecolor=c, edgecolor=NAVY, alpha=0.9))
        ax.text(x + 0.78, 3.1, lab, ha="center", va="center", fontsize=8.5, color=NAVY)
    ax.add_patch(FancyBboxPatch((8.2, 2.1), 1.5, 2.0, boxstyle="round,pad=0.08",
                                facecolor=RIGHT_C, edgecolor=NAVY, alpha=0.85))
    ax.text(8.95, 3.1, "右电极", ha="center", va="center", color="white", fontsize=9)
    for x1, x2 in [(1.8, 2.4), (3.95, 4.3), (5.85, 6.2), (7.75, 8.2)]:
        ax.annotate("", xy=(x2, 3.1), xytext=(x1, 3.1),
                    arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.15))
    ax.text(5, 5.2, r"表面距 $\leq 1.8\,\mathrm{nm}$ 即整体带电", ha="center",
            color=NAVY, fontsize=10)
    ax.text(5, 0.85, "极化沿接触图做广度优先传播，等价于无向连通", ha="center",
            color=SLATE, fontsize=8.5)
    ax.set_title("(b) 极化传递与图连通等价", loc="left", color=NAVY)
    fig.tight_layout()
    save(fig, "fig_polar")


def fig_costcurve():
    rows = json.load(open("/workspace/paper/results/problem4_stage2.json"))
    fig, ax = plt.subplots(figsize=(6.5, 3.55))
    # group by nB
    nBs = sorted({r["nB"] for r in rows})
    cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(nBs)))
    for c, nB in zip(cmap, nBs):
        sub = sorted([r for r in rows if r["nB"] == nB], key=lambda r: r["nA"])
        ax.plot([r["cost"] for r in sub], [r["P"] for r in sub],
                "-o", color=c, ms=3.5, lw=1.15, label=fr"$N_B={nB}$")
    ax.axhline(0.90, color=CORAL, ls="--", lw=1.1)
    ax.axvline(8.92, color=NAVY, ls=":", lw=1.0)
    ax.text(8.95, 0.42, "纯 A\n8.92 元", color=NAVY, fontsize=8)
    ax.set_xlabel("成本 / 元")
    ax.set_ylabel(r"$\hat P_{\mathrm{on}}$")
    ax.set_ylim(0.35, 1.02)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_costcurve")


if __name__ == "__main__":
    fig_lenhist()
    fig_graph()
    fig_ecdf()
    fig_mixheat()
    fig_vex()
    fig_bootstrap()
    fig_degree()
    fig_polar()
    fig_costcurve()
    print("all extra figures done")
