"""Shared loading + styling for every analysis script. Reads only results/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "report/figures"
METHODS = ["ddpg", "ddpg_notarget", "td3", "sac", "crossq", "ln_critic", "ours", "ours_exact",
           "ours_adam_actor", "ours_twin", "sac_kl"]
_pal = sns.color_palette("colorblind", 10) + [(0.3, 0.3, 0.3)]
COLOR = dict(zip(METHODS, _pal))  # same colour for the same method in every figure


def color(tag):
    """Colour of the longest method name that prefixes the run tag (e.g. 'ddpg_tau0.05' -> ddpg)."""
    base = next((m for m in sorted(METHODS, key=len, reverse=True) if tag.startswith(m)), None)
    return COLOR.get(base, "k")


def runs(exp):
    """DataFrame of run folders: exp, tag, env, seed, path, done, diverged."""
    rows = []
    for p in sorted((ROOT / "results" / exp).glob("*/*/seed_*")):
        rows.append(dict(exp=exp, tag=p.parts[-3], env=p.parts[-2], seed=int(p.name[5:]), path=p,
                         done=(p / "DONE").exists(), diverged=(p / "DIVERGED").exists()))
    return pd.DataFrame(rows)


def load(path, what):
    f = {"eval": "eval.csv", "metrics": "metrics.parquet", "over": "overestimation.csv"}[what]
    p = Path(path) / f
    try:
        return pd.read_parquet(p) if f.endswith("parquet") else pd.read_csv(p)
    except Exception:  # missing or empty (run just started)
        return pd.DataFrame()


def stacked(R, what, col):
    """Long DataFrame (tag, env, seed, step, value) of one logged column across runs."""
    out = []
    for r in R.itertuples():
        d = load(r.path, what)
        if col in d:
            out.append(pd.DataFrame(dict(tag=r.tag, env=r.env, seed=r.seed, step=d.step, value=d[col])))
    return pd.concat(out) if out else pd.DataFrame(columns=["tag", "env", "seed", "step", "value"])


def final_perf(path, last=5):
    """Per-seed final performance: mean evaluation return over the last 5 evaluation points."""
    d = load(path, "eval")
    return d.ret_mean.iloc[-last:].mean() if len(d) else np.nan


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300 if ext == "pdf" else 110, bbox_inches="tight")
    plt.close(fig)
