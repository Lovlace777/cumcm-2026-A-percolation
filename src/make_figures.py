#!/usr/bin/env python3
"""Publication-quality figures for the CUMCM-style paper."""
from __future__ import annotations

import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, FancyArrow
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from simulate import (
    HALF, BOX, R_A, R_B, H_A, DELTA, THRESH_AA, THRESH_AE,
    VOL_A, VOL_CUBE, COST_ONE_A, COST_ONE_B,
    load_group, is_percolating_rods, UF, _rod_contacts,
    n_from_phi_A, phi_from_n_A, OUT_DIR, FIG_DIR,
)

os.makedirs(FIG_DIR, exist_ok=True)

# ---------- style ----------
mpl.rcParams.update({
    "font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"],
    "font.size": 10.5,
    "axes.unicode_minus": False,
    "axes.linewidth": 0.9,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
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
INK = "#1A1A1A"
LEFT_C = "#2E6B9E"
RIGHT_C = "#C45C26"
SPAN_C = "#1F7A6B"
ISO_C = "#9AA3A8"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"))
    plt.close(fig)
    print("saved", name)


# =====================================================================
# Fig: coordinate system
# =====================================================================
def fig_coord():
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # cube wire
    b = HALF
    edges = [
        [(-b, -b, -b), (b, -b, -b)], [(-b, b, -b), (b, b, -b)],
        [(-b, -b, b), (b, -b, b)], [(-b, b, b), (b, b, b)],
        [(-b, -b, -b), (-b, b, -b)], [(b, -b, -b), (b, b, -b)],
        [(-b, -b, b), (-b, b, b)], [(b, -b, b), (b, b, b)],
        [(-b, -b, -b), (-b, -b, b)], [(b, -b, -b), (b, -b, b)],
        [(-b, b, -b), (-b, b, b)], [(b, b, -b), (b, b, b)],
    ]
    for p, q in edges:
        ax.plot(*zip(p, q), color=NAVY, lw=1.1, alpha=0.85)

    yy, zz = np.meshgrid(np.linspace(-b, b, 2), np.linspace(-b, b, 2))
    ax.plot_surface(-b * np.ones_like(yy), yy, zz, color=LEFT_C, alpha=0.22, linewidth=0)
    ax.plot_surface(b * np.ones_like(yy), yy, zz, color=RIGHT_C, alpha=0.22, linewidth=0)

    # sample rod
    p = np.array([-2200.0, -800.0, 400.0])
    q = np.array([2800.0, 900.0, -600.0])
    ax.plot(*zip(p, q), color=GOLD, lw=2.6)
    ax.scatter(*p, color=GOLD, s=18)
    ax.scatter(*q, color=GOLD, s=18)
    # sample sphere
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 12)
    cx, cy, cz, rad = 800.0, -1600.0, 1200.0, 550.0
    xs = cx + rad * np.outer(np.cos(u), np.sin(v)).T
    ys = cy + rad * np.outer(np.sin(u), np.sin(v)).T
    zs = cz + rad * np.outer(np.ones_like(u), np.cos(v)).T
    ax.plot_surface(xs, ys, zs, color=TEAL, alpha=0.35, linewidth=0)

    ax.quiver(0, 0, 0, 2800, 0, 0, color=CORAL, arrow_length_ratio=0.08, lw=1.4)
    ax.quiver(0, 0, 0, 0, 2800, 0, color=STEEL, arrow_length_ratio=0.08, lw=1.4)
    ax.quiver(0, 0, 0, 0, 0, 2800, color=TEAL, arrow_length_ratio=0.08, lw=1.4)
    ax.text(3000, 0, 200, "X", color=CORAL, fontsize=11)
    ax.text(0, 3000, 200, "Y", color=STEEL, fontsize=11)
    ax.text(200, 200, 3000, "Z", color=TEAL, fontsize=11)
    ax.text(-b - 200, 0, b + 400, "左带电面\n$x=-5000$", color=LEFT_C, fontsize=8.5, ha="right")
    ax.text(b + 200, 0, b + 400, "右带电面\n$x=+5000$", color=RIGHT_C, fontsize=8.5)

    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.set_zlim(-b, b)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([-5000, 0, 5000])
    ax.set_yticks([-5000, 0, 5000])
    ax.set_zticks([-5000, 0, 5000])
    ax.tick_params(labelsize=8)
    ax.set_xlabel("X / nm", labelpad=4)
    ax.set_ylabel("Y / nm", labelpad=4)
    ax.set_zlabel("Z / nm", labelpad=4)
    ax.view_init(elev=18, azim=-58)
    ax.grid(False)
    save(fig, "fig_coord")


# =====================================================================
# Fig: wrapping schematic (2D)
# =====================================================================
def fig_wrap():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-5600, 5600)
        ax.set_ylim(-1800, 1900)
        ax.axvspan(-5000, 5000, color=CREAM, zorder=0)
        ax.plot([-5000, 5000, 5000, -5000, -5000],
                [-1200, -1200, 1200, 1200, -1200], color=NAVY, lw=1.4)
        ax.plot([-5000, -5000], [-1200, 1200], color=LEFT_C, lw=3.2, solid_capstyle="butt")
        ax.plot([5000, 5000], [-1200, 1200], color=RIGHT_C, lw=3.2, solid_capstyle="butt")
        ax.set_xticks([-5000, 0, 5000])
        ax.set_yticks([-1000, 0, 1000])
        ax.set_xlabel("X / nm")
        ax.set_ylabel("Y / nm")
        ax.grid(True, ls=":", color=SAND, zorder=0)

    # left: before wrap
    ax = axes[0]
    ax.plot([3500, 6000], [-200, 700], color=CORAL, lw=2.8)
    ax.plot([5000, 6000], [220, 700], color=CORAL, lw=6.0, alpha=0.28)
    ax.annotate("越界段 $X_1$", xy=(5600, 520), xytext=(2200, 1050),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.0),
                color=CORAL, fontsize=9)
    ax.set_title("(a) 截断前：轴线穿出右边界", loc="left", fontsize=10.5, color=NAVY)

    # right: after wrap
    ax = axes[1]
    ax.plot([3500, 5000], [-200, 220], color=STEEL, lw=2.8, label="保留段")
    ax.plot([-5000, -4000], [220, 700], color=CORAL, lw=2.8, ls="--", label="平移段")
    ax.annotate("", xy=(-4500, 460), xytext=(5500, 460),
                arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.1,
                                connectionstyle="arc3,rad=0.18"))
    ax.text(0, 1450, "沿 -X 平移一个边长 10000 nm", ha="center",
            color=GOLD, fontsize=8.5)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("(b) 截断后：越界段由左侧回填", loc="left", fontsize=10.5, color=NAVY)
    fig.tight_layout()
    save(fig, "fig_wrap")


# =====================================================================
# Fig: 3D configurations
# =====================================================================
def _component_colors(P, Q):
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
    spanning = rootL == rootR
    cols = []
    tags = []
    for i in range(n):
        r = uf.find(i)
        if spanning and r == rootL:
            cols.append(SPAN_C)
            tags.append("span")
        elif r == rootL:
            cols.append(LEFT_C)
            tags.append("left")
        elif r == rootR:
            cols.append(RIGHT_C)
            tags.append("right")
        else:
            cols.append(ISO_C)
            tags.append("iso")
    return cols, tags, spanning


def fig_groups_3d():
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.55))
    for ax, name in zip(axes, ("组1", "组2", "组3")):
        P, Q = load_group(name)
        cols, tags, spanning = _component_colors(P, Q)
        ax.axvspan(-HALF - 80, -HALF + 40, color=LEFT_C, alpha=0.18, lw=0)
        ax.axvspan(HALF - 40, HALF + 80, color=RIGHT_C, alpha=0.18, lw=0)
        ax.axvline(-HALF, color=LEFT_C, lw=1.6)
        ax.axvline(HALF, color=RIGHT_C, lw=1.6)
        order = np.argsort([{"iso": 0, "left": 1, "right": 1, "span": 2}[t] for t in tags])
        lw = 1.7 if name != "组3" else 0.55
        al = 0.95 if name != "组3" else 0.55
        for i in order:
            ax.plot([P[i, 0], Q[i, 0]], [P[i, 1], Q[i, 1]],
                    color=cols[i], lw=lw, alpha=al, solid_capstyle="round")
        ax.set_xlim(-5600, 5600)
        if name == "组3":
            ax.set_ylim(-5600, 5600)
        else:
            ax.set_ylim(-620, 620)
        ax.set_aspect("auto")
        ax.set_xlabel("X / nm")
        ax.set_ylabel("Y / nm")
        verdict = "导通" if spanning else "不导通"
        ax.set_title(f"{name}  n={len(P)}  {verdict}", fontsize=11, color=NAVY)
        ax.grid(True, ls=":", color="#E2DCD0")
        for s in ax.spines.values():
            s.set_color("#B0A89A")
    handles = [
        Line2D([0], [0], color=SPAN_C, lw=2.2, label="跨极通路"),
        Line2D([0], [0], color=LEFT_C, lw=2.2, label="仅连左极"),
        Line2D([0], [0], color=RIGHT_C, lw=2.2, label="仅连右极"),
        Line2D([0], [0], color=ISO_C, lw=2.2, label="孤立簇"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "fig_groups3d")




# =====================================================================
# Fig: P vs phi
# =====================================================================
def fig_pphi():
    w = np.load(os.path.join(OUT_DIR, "survival_wrap.npz"))
    c = np.load(os.path.join(OUT_DIR, "survival.npz"))
    fig, ax = plt.subplots(figsize=(6.6, 4.15))
    # wrap
    phi_w = w["Ns"] * VOL_A / VOL_CUBE * 100
    ax.fill_between(phi_w, np.clip(w["P"] - 1.96 * w["se"], 0, 1),
                    np.clip(w["P"] + 1.96 * w["se"], 0, 1),
                    color=TEAL, alpha=0.18, linewidth=0)
    ax.plot(phi_w, w["P"], color=TEAL, lw=2.0, label="截断分段系综（主模型）")
    # contained
    phi_c = c["Ns"] * VOL_A / VOL_CUBE * 100
    ax.plot(phi_c, c["P"], color=STEEL, lw=1.5, ls="--", label="完全内含系综（对照）")

    marks = [(0.50, 0.104), (0.60, 0.228), (0.70, 0.520), (1.00, 1.00)]
    # use actual from json
    with open(os.path.join(OUT_DIR, "problem23_wrap.json")) as f:
        p2 = json.load(f)["problem2"]
    xs, ys = [], []
    for k in ("0.50%", "0.60%", "0.70%", "1.00%"):
        xs.append(float(k[:-1]))
        ys.append(p2[k]["P"])
    ax.scatter(xs, ys, s=46, color=CORAL, zorder=5, label="题设体积分数")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y*100:.1f}%", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8, color=CORAL)

    ax.axhline(0.90, color=GOLD, ls=":", lw=1.3)
    ax.text(0.22, 0.915, r"$P=90\%$", color=GOLD, fontsize=9)
    ax.axvline(0.85, color=GOLD, ls=":", lw=1.1, alpha=0.8)
    ax.scatter([0.85], [0.908], s=54, marker="D", color=GOLD, zorder=6, label="90% 阈值 0.85%")

    ax.set_xlim(0.15, 1.35)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"介质 A 体积分数 $\varphi_A\ /\%$")
    ax.set_ylabel(r"导通概率 $P_{\mathrm{on}}$")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, ls=":", color="#DED8CC")
    for s in ax.spines.values():
        s.set_color("#B0A89A")
    fig.tight_layout()
    save(fig, "fig_pphi")


def fig_nc_hist():
    crit = np.load(os.path.join(OUT_DIR, "crit_wrap.npy"))
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    bins = np.linspace(180, 740, 22)
    ax.hist(crit, bins=bins, color=STEEL, alpha=0.82, edgecolor="white", linewidth=0.6)
    ax.axvline(np.median(crit), color=CORAL, lw=1.6, label=f"中位数 $N_c={np.median(crit):.0f}$")
    ax.axvline(np.quantile(crit, 0.9), color=GOLD, lw=1.6, ls="--",
               label=f"90% 分位 $N_c={np.quantile(crit,0.9):.0f}$")
    ax.set_xlabel(r"首次导通所需介质 A 个数 $N_c$")
    ax.set_ylabel("实现次数")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", ls=":", color="#DED8CC")
    fig.tight_layout()
    save(fig, "fig_nchist")


# =====================================================================
# Fig: cost heatmap-like scatter / contour from stage2
# =====================================================================
def fig_cost():
    with open(os.path.join(OUT_DIR, "problem4_stage2.json")) as f:
        rows = json.load(f)
    nA = np.array([r["nA"] for r in rows])
    nB = np.array([r["nB"] for r in rows])
    P = np.array([r["P"] for r in rows])
    cost = np.array([r["cost"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.55))

    ax = axes[0]
    sc = ax.scatter(nA, nB, c=P, cmap="YlGnBu", s=62, vmin=0.45, vmax=1.0,
                    edgecolors="white", linewidths=0.4, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$P_{\mathrm{on}}$")
    ok = P >= 0.90
    ax.scatter(nA[ok], nB[ok], facecolors="none", edgecolors=CORAL, s=110,
               linewidths=1.2, label=r"$P\geq 0.90$", zorder=4)
    ax.set_xlabel(r"介质 A 个数 $N_A$")
    ax.set_ylabel(r"介质 B 个数 $N_B$")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, ls=":", color="#DED8CC")

    ax = axes[1]
    sc = ax.scatter(cost, P, c=np.where(ok, TEAL, SLATE), s=54, zorder=3)
    ax.axhline(0.90, color=GOLD, ls=":", lw=1.2)
    ax.axvline(601 * COST_ONE_A, color=CORAL, ls="--", lw=1.0, label="纯 A 成本")
    # annotate best feasible
    if np.any(ok):
        i = np.argmin(np.where(ok, cost, 1e9))
        ax.scatter([cost[i]], [P[i]], s=80, marker="*", color=CORAL, zorder=5)
        ax.annotate(f"({int(nA[i])},{int(nB[i])})\n{cost[i]:.2f} 元",
                    (cost[i], P[i]), textcoords="offset points", xytext=(-52, 12),
                    fontsize=8, color=CORAL)
    ax.set_xlabel("填充总成本 / 元")
    ax.set_ylabel(r"$P_{\mathrm{on}}$")
    ax.set_ylim(0.42, 1.04)
    ax.legend(frameon=False)
    ax.grid(True, ls=":", color="#DED8CC")
    fig.tight_layout()
    save(fig, "fig_cost")


# =====================================================================
# Fig: excluded volume theory vs MC
# =====================================================================
def fig_theory():
    w = np.load(os.path.join(OUT_DIR, "survival_wrap.npz"))
    phi = w["Ns"] * VOL_A / VOL_CUBE * 100
    # logistic fit around MC
    fig, ax = plt.subplots(figsize=(6.4, 3.85))
    ax.plot(phi, w["P"], color=TEAL, lw=2.0, label="增量 Monte Carlo")
    ax.fill_between(phi, np.clip(w["P"] - 1.96 * w["se"], 0, 1),
                    np.clip(w["P"] + 1.96 * w["se"], 0, 1), color=TEAL, alpha=0.15, lw=0)

    # theoretical phi_c
    D_eff = 2 * R_A + DELTA
    phi_c_th = 0.70 * D_eff / H_A * 100  # percent, using 0.7 D/L
    # more precise: 2.8 R^2 / (L D_eff)
    phi_c_th2 = 2.8 * R_A ** 2 / (H_A * D_eff) * 100
    ax.axvline(phi_c_th2, color=CORAL, ls="--", lw=1.3,
               label=fr"排除体积 $\varphi_c^{{\mathrm{{th}}}}={phi_c_th2:.2f}\%$")
    ax.axvline(0.70, color=STEEL, ls=":", lw=1.2, label=r"MC 中位数 $\varphi_{50}=0.70\%$")
    ax.axvline(0.85, color=GOLD, ls=":", lw=1.2, label=r"MC 90% 阈值 $0.85\%$")
    ax.set_xlim(0.2, 1.25)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$\varphi_A\ /\%$")
    ax.set_ylabel(r"$P_{\mathrm{on}}$")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, ls=":", color="#DED8CC")
    fig.tight_layout()
    save(fig, "fig_theory")


# =====================================================================
# Fig: sensitivity to tunneling distance
# =====================================================================
def fig_sens():
    # lightweight: reuse wrap P curve scaled by effective D
    # We'll plot theoretical shift and a small MC if files exist
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    deltas = np.array([0.8, 1.2, 1.8, 2.4, 3.0])
    Deff = 2 * R_A + deltas
    phi_c = 2.8 * R_A ** 2 / (H_A * Deff) * 100
    ax.plot(deltas, phi_c, "o-", color=NAVY, lw=1.8, ms=6, label="排除体积预测 $\\varphi_c$")
    ax.scatter([1.8], [2.8 * R_A ** 2 / (H_A * (60 + 1.8)) * 100],
               s=70, color=CORAL, zorder=5, label="题设 $\\delta=1.8\\,\\mathrm{nm}$")
    ax.set_xlabel(r"隧穿阈值 $\delta$ / nm")
    ax.set_ylabel(r"渗流阈值 $\varphi_c$ / %")
    ax.grid(True, ls=":", color="#DED8CC")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_sens")


# =====================================================================
# Fig: task dependency
# =====================================================================
def fig_framework():
    fig, ax = plt.subplots(figsize=(7.3, 3.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    boxes = [
        (0.25, 2.4, 2.2, 1.35, "统一几何接口\n距离 / 隧穿 / 电极", CREAM),
        (2.85, 2.4, 2.2, 1.35, "连通图 $G$\n并查集动态合并", CREAM),
        (5.45, 2.4, 2.2, 1.35, "增量 Monte Carlo\nPon(φ) 曲线", CREAM),
        (8.05, 2.4, 1.75, 1.35, "成本优化\n$(N_A,N_B)$", CREAM),
        (0.25, 0.35, 2.2, 1.35, "问题 1\n给定构型判定", "#E8F1F5"),
        (2.85, 0.35, 2.2, 1.35, "问题 2\n四点导通概率", "#E8F1F5"),
        (5.45, 0.35, 2.2, 1.35, "问题 3\n90% 最低填充", "#E8F1F5"),
        (8.05, 0.35, 1.75, 1.35, "问题 4\n最低成本", "#E8F1F5"),
    ]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor=c, edgecolor=NAVY, linewidth=1.05))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.6, color=NAVY)
    for x0 in (1.35, 3.95, 6.55, 8.92):
        ax.annotate("", xy=(x0, 1.75), xytext=(x0, 2.35),
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))
    for x0, x1 in ((2.45, 2.85), (5.05, 5.45), (7.65, 8.05)):
        ax.annotate("", xy=(x1, 3.05), xytext=(x0, 3.05),
                    arrowprops=dict(arrowstyle="->", color=STEEL, lw=1.15))
    fig.tight_layout()
    save(fig, "fig_framework")


def fig_algo():
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    nodes = [
        (3.2, 10.7, 3.6, 1.0, "输入：介质端点 / 球心", "#E8F1F5", "s"),
        (3.2, 9.2, 3.6, 1.0, "初始化并查集\n电极 $L,R$ + 全部介质", CREAM, "r"),
        (3.2, 7.55, 3.6, 1.15, "对每条介质：若 d≤r+δ\n则与对应电极合并", CREAM, "r"),
        (3.2, 5.75, 3.6, 1.25, "扫描介质对：若最短距\n≤ ri+rj+δ 则合并", CREAM, "r"),
        (3.2, 4.05, 3.6, 1.05, "Find(L) = Find(R) ?", "#F6E7C1", "d"),
        (0.35, 2.15, 3.1, 1.0, "输出：导通", "#D9EDE7", "r"),
        (6.5, 2.15, 3.1, 1.0, "输出：不导通", "#F4D6D6", "r"),
    ]
    for x, y, w, h, t, c, kind in nodes:
        if kind == "d":
            poly = np.array([[x + w / 2, y + h], [x + w, y + h / 2],
                             [x + w / 2, y], [x, y + h / 2]])
            ax.fill(poly[:, 0], poly[:, 1], facecolor=c, edgecolor=NAVY, lw=1.05)
        else:
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                                        facecolor=c, edgecolor=NAVY, lw=1.05))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.7, color=NAVY)
    arrows = [
        ((5.0, 10.7), (5.0, 10.2)),
        ((5.0, 9.2), (5.0, 8.7)),
        ((5.0, 7.55), (5.0, 7.0)),
        ((5.0, 5.75), (5.0, 5.1)),
        ((3.2, 4.55), (1.9, 3.15)),
        ((6.8, 4.55), (8.05, 3.15)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=STEEL, lw=1.15))
    ax.text(2.0, 3.55, "是", color=TEAL, fontsize=9)
    ax.text(7.4, 3.55, "否", color=CORAL, fontsize=9)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
    save(fig, "fig_algo")


def fig_p1_bars():
    with open(os.path.join(OUT_DIR, "problem1.json")) as f:
        d = json.load(f)
    names = ["组1", "组2", "组3"]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.15), sharey=False)
    keys = [("n_left", "触左极"), ("n_right", "触右极"), ("n_aa", "介质接触对"), ("largest", "最大簇")]
    colors = [LEFT_C, RIGHT_C, GOLD, TEAL]
    for ax, name in zip(axes, names):
        info = d[name]
        vals = [info[k] for k, _ in keys]
        bars = ax.bar([k[1] for k in keys], vals, color=colors, width=0.68)
        ax.set_title(f"{name}  {'导通' if info['conducting'] else '不导通'}",
                     color=TEAL if info["conducting"] else CORAL, fontsize=11)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(True, axis="y", ls=":", color="#DED8CC")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save(fig, "fig_p1bars")


if __name__ == "__main__":
    print("drawing figures...")
    fig_coord()
    fig_wrap()
    fig_groups_3d()
    fig_pphi()
    fig_nc_hist()
    fig_cost()
    fig_theory()
    fig_sens()
    fig_framework()
    fig_algo()
    fig_p1_bars()
    print("all figures done")
