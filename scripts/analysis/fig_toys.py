"""Figures F1 (w -> 2w) and F2 (Baird) from results/E1 and results/E2 only.

    python scripts/analysis/fig_toys.py  -> report/figures/F1_w2w.{pdf,png}, F2_baird.{pdf,png}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIG = f"{ROOT}/report/figures"
PAL = sns.color_palette("colorblind")
# Same colour per variant in every figure.
STYLE = {"plain": (PAL[3], "-", "plain TD"),
         "clip": (PAL[0], "-", "clip only"),
         "clip_proj": (PAL[2], "-", "clip + projection"),
         "clip_wd": (PAL[1], "-", "clip + decay (decoupled)"),
         "clip_wd_coupled": (PAL[4], ":", "clip + decay (coupled)")}
SCHED = {"const": r"constant $\alpha=\alpha_0$", "decay": r"$\alpha_n=\alpha_0/n^{0.6}$"}


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIG}/{name}.{ext}", dpi=300, bbox_inches="tight")


def f1():
    df = pd.read_csv(f"{ROOT}/results/E1/w_trace.csv")
    summ = pd.read_csv(f"{ROOT}/results/E1/summary.csv")
    cfg = json.load(open(f"{ROOT}/results/E1/config.json"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, sch in zip(axes, SCHED):
        for v, (c, ls, lab) in STYLE.items():
            d = df[(df.schedule == sch) & (df.variant == v)]
            ax.plot(d.step, d.w.abs(), color=c, ls=ls, lw=1.6, label=lab)
            ov = summ[(summ.schedule == sch) & (summ.variant == v)].overflow_step.iloc[0]
            if pd.notna(ov):  # |w| > 1e300 (then inf): mark the step
                ax.axvline(ov, color=c, ls="--", lw=0.8)
                ax.text(ov, 0.5, f" overflow at n={int(ov)}", color=c, fontsize=7, rotation=90,
                        transform=ax.get_xaxis_transform(), va="center")
        d = df[(df.schedule == sch) & (df.variant == "clip")]
        ax.plot(d.step, d.p1_bound, color="k", ls=(0, (2, 3)), lw=1.0, label=r"P1 bound $|w_N|+\lambda\sum\alpha_m$")
        # plain TD grows exponentially; cap the axis so the slow (P1) divergence stays visible
        top = 30 * df[df.variant != "plain"].w.abs().max()
        fin = df[(df.schedule == sch) & (df.variant == "plain")].w.abs()
        ax.text(0.98, 0.97, f"plain TD leaves the axis; max logged |w| = {fin.max():.2g}", ha="right", va="top",
                fontsize=7, color=STYLE["plain"][0], transform=ax.transAxes)
        ax.set(xscale="log", yscale="log", xlabel="step $n$", title=SCHED[sch], ylim=(0.5, top))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("$|w_n|$")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(rf"E1 'w$\to$2w': $\gamma$={cfg['gamma']}, $\alpha_0$={cfg['alpha0']}, $\lambda$={cfg['clip_lambda']}, "
                 rf"R={cfg['proj_R']}, wd={cfg['weight_decay']}, $w_0$={cfg['w0']}", fontsize=10)
    fig.tight_layout()
    save(fig, "F1_w2w")


def f2():
    df = pd.read_csv(f"{ROOT}/results/E2/norms.csv.gz")
    cfg = json.load(open(f"{ROOT}/results/E2/config.json"))
    rows = [("linear", "out_norm", "linear: $\\|w\\|$"),
            ("net", "out_norm", "net: output-layer norm"),
            ("net", "hidden_norm", "net: hidden-layer norm")]
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
    for i, (fc, col, ylab) in enumerate(rows):
        for j, sch in enumerate(SCHED):
            ax = axes[i, j]
            for v, (c, ls, lab) in STYLE.items():
                d = df[(df.fclass == fc) & (df.schedule == sch) & (df.variant == v)]
                g = d.replace([np.inf, -np.inf], np.nan).groupby("step")[col]
                med, lo, hi = g.median(), g.min(), g.max()
                ax.plot(med.index, med, color=c, ls=ls, lw=1.5, label=lab)
                ax.fill_between(med.index, lo, hi, color=c, alpha=0.15, lw=0)
            ax.set(xscale="log", yscale="log")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(SCHED[sch])
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 2:
                ax.set_xlabel("step")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"E2 Baird: median and min-max over {cfg['seeds']} seeds; $\\alpha_0$={cfg['alpha0']}, "
                 f"$\\lambda$={cfg['clip_lambda']}, R={cfg['R_mult']:g}x init norm, wd={cfg['weight_decay']}"
                 "\n(missing curve segments = overflow to inf/nan)", fontsize=10)
    fig.tight_layout()
    save(fig, "F2_baird")


if __name__ == "__main__":
    f1()
    f2()
