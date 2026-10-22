"""Result P2 sweep (unit test 3 extended): how well does the natural trust-region step reproduce
Delta mu(s_i) = eta q_i as damping grows (width 1024) or as width shrinks (zero damping, pinv)?

    python scripts/p2_figure.py  -> results/unit_tests/p2_sweep.csv, report/figures/p2_check.{pdf,png}

Setup: Actor(obs=3, act=2, one hidden ReLU layer of `width`), M = 32 random states, random q_i ~ N(0, I),
eps = 1e-3, float64, 5 seeds. F = J^T J / M, g = J^T q / M, x = (F + damp I)^+ g, Delta theta = beta x with
beta = sqrt(eps / x^T F x). Agreement is measured on the linearised action change J Delta theta.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.func import functional_call, jacrev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdrl.networks import Actor  # noqa: E402

torch.set_default_dtype(torch.float64)
OBS, D, M, EPS, SEEDS = 3, 2, 32, 1e-3, 5
DAMPS = [0.0] + [10.0 ** k for k in range(-6, 1)]
WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256, 1024]


def jacobian(actor, states):
    p = {n: v.detach() for n, v in actor.named_parameters()}
    jac = jacrev(lambda q: functional_call(actor, q, (states,)))(p)
    return torch.cat([jac[n].reshape(states.shape[0] * D, -1) for n in p], 1)


def metrics(J, q, F_eig, damp):
    lam, V = F_eig
    g = J.T @ q.reshape(-1) / M
    if damp == 0:  # Moore-Penrose pseudo-inverse with numpy's default cutoff
        keep = lam > lam.max() * lam.numel() * torch.finfo(lam.dtype).eps
        inv = torch.where(keep, 1 / lam, torch.zeros_like(lam))
    else:
        inv = 1 / (lam.clamp_min(0) + damp)
    x = V @ (inv * (V.T @ g))
    beta = math.sqrt(EPS / (x @ (J.T @ (J @ x)) / M).item())
    a, b = J @ (beta * x), q.reshape(-1)
    eta = math.sqrt(EPS / q.pow(2).sum(-1).mean().item())
    return dict(cosine=torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
                scale=(a @ b / (eta * b @ b)).item(), rel_err=((a - eta * b).norm() / (eta * b).norm()).item())


def main():
    rows = []
    for seed in range(SEEDS):
        for width in WIDTHS:
            torch.manual_seed(seed)
            actor = Actor(OBS, D, hidden=(width,))
            states, q = torch.randn(M, OBS), torch.randn(M, D)
            J = jacobian(actor, states)
            F_eig = torch.linalg.eigh(J.T @ J / M)
            info = dict(seed=seed, width=width, n_params=J.shape[1], Md=M * D,
                        rank_J=torch.linalg.matrix_rank(J).item())
            for damp in (DAMPS if width == 1024 else [0.0]):
                rows.append(dict(info, damp=damp, **metrics(J, q, F_eig, damp)))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(f"{ROOT}/results/unit_tests", exist_ok=True)
    df.to_csv(f"{ROOT}/results/unit_tests/p2_sweep.csv", index=False)
    plot(df)


def plot(df):
    pal = sns.color_palette("colorblind")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    panels = [(df[df.width == 1024], "damp", r"damping $\lambda_{damp}$ (width 1024)"),
              (df[df.damp == 0], "width", "hidden width (zero damping, pinv)")]
    for ax, (d, x, label) in zip(axes, panels):
        agg = d.groupby(x).agg(["mean", "std"])
        xs = agg.index.values.astype(float)
        if x == "damp":
            xs = xs.copy()
            xs[0] = 1e-7  # draw damp = 0 at the left edge of the log axis
        for i, (m, name) in enumerate([("cosine", r"cosine$(J\Delta\theta, q)$"),
                                       ("scale", r"$\langle J\Delta\theta, q\rangle / (\eta\|q\|^2)$")]):
            ax.errorbar(xs, agg[m]["mean"], yerr=agg[m]["std"], marker="os"[i], color=pal[i], label=name, capsize=2)
        ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        if x == "damp":
            ax.set_xticks(xs)
            ax.set_xticklabels(["0"] + [f"1e{k}" for k in range(-6, 1)], fontsize=7)
        else:
            # full row rank of J needs M*d <= #params; mark the smallest width where it holds
            full = d.groupby("width").apply(lambda g: (g.rank_J == g.Md).all())
            w0 = full[full].index.min()
            ax.axvline(w0, color="grey", ls="--", lw=1)
            ax.text(w0 * 1.1, 0.05, f"J full row rank\nfor width $\\geq${w0}", fontsize=7, color="grey")
    axes[0].set_ylabel("agreement with P2 prediction")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"P2 check: natural step vs $\\Delta\\mu(s_i) = \\eta q_i$  (M={M}, d={D}, {SEEDS} seeds)", fontsize=10)
    fig.tight_layout()
    os.makedirs(f"{ROOT}/report/figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{ROOT}/report/figures/p2_check.{ext}", dpi=300)


if __name__ == "__main__":
    main()
