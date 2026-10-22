"""Diagnostic panel for any experiment (works on partial runs): one row per env; columns = return,
target drift |Δy| (log), critic output-layer norm, clipped-update fraction, actor change at s′, Q bias.
Thick line = mean over seeds, thin lines = individual seeds.   python scripts/analysis/monitor.py pilot E0
"""
import sys

import matplotlib.pyplot as plt

from common import color, runs, save, stacked

PANELS = [("eval", "ret_mean", "return", False),
          ("metrics", "drift_mean", "target drift |Δy|", True),
          ("metrics", "wnorm_out0", "critic output-layer norm", True),
          ("metrics", "clip_frac", "clipped-update fraction", False),
          ("metrics", "dmu_probe", "‖Δμ(s′)‖² per update (probe)", True),
          ("over", "q_bias", "Q − MC return", False)]


def main(exp):
    R = runs(exp)
    envs = sorted(R.env.unique())
    fig, axes = plt.subplots(len(envs), len(PANELS), figsize=(3.3 * len(PANELS), 2.6 * len(envs)), squeeze=False)
    for j, (what, col, title, logy) in enumerate(PANELS):
        D = stacked(R, what, col)
        for i, env in enumerate(envs):
            ax, E = axes[i, j], D[D.env == env]
            for tag, g in E.groupby("tag"):
                for _, s in g.groupby("seed"):
                    ax.plot(s.step, s.value, color=color(tag), lw=0.4, alpha=0.35)
                m = g.groupby("step").value.mean()
                ax.plot(m.index, m.values, color=color(tag), lw=1.6, label=tag)
            if logy and len(E) and (E.value > 0).any():
                ax.set_yscale("log")
            ax.set_title(f"{env} — {title}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.set_xlabel("env steps", fontsize=6)
    handles = {l: h for ax in axes.flat for h, l in zip(*ax.get_legend_handles_labels())}
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=len(handles), fontsize=8,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save(fig, f"monitor_{exp}")
    print(f"{exp}: {R.done.sum()}/{len(R)} done, {R.diverged.sum()} diverged -> report/figures/monitor_{exp}.png")


if __name__ == "__main__":
    for e in sys.argv[1:]:
        main(e)
