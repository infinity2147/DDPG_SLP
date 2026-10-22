"""Phase-1 toy problems (float64 numpy, semi-gradient TD with SGD).

    python scripts/toys.py e1   # "w -> 2w"           -> results/E1/
    python scripts/toys.py e2   # Baird's counterexample -> results/E2/

Variants (all clip the OUTPUT-layer gradient g to clip(g, lam) = g * min(1, lam / ||g||)):
    plain       no clipping, no stabiliser
    clip        clipping only
    clip_proj   clipping + projection of the output layer onto ||w|| <= R (A1)
    clip_wd     clipping + DECOUPLED decay after the step: w <- (1 - alpha*wd) w (as tdrl/stabilisers.py)
    clip_wd_coupled  extra: decay inside the clipped gradient, clip(g + wd*w, lam)
Schedules: const (alpha = alpha0) and decay (alpha_n = alpha0 / n^0.6), n = 1, 2, ...
All variants of one seed share the same random stream (common random numbers).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = ["plain", "clip", "clip_proj", "clip_wd", "clip_wd_coupled"]
SCHEDULES = ["const", "decay"]
OVERFLOW = 1e300


def alphas(alpha0, n_steps):
    n = np.arange(1, n_steps + 1, dtype=np.float64)
    return {"const": np.full(n_steps, alpha0), "decay": alpha0 / n ** 0.6}


def log_steps(n_steps, k=300):
    return set(np.unique(np.logspace(0, np.log10(n_steps), k).astype(int)).tolist())


def clip(g, lam):
    """Row-wise norm clipping; g has shape (..., p)."""
    n = np.linalg.norm(g, axis=-1, keepdims=True)
    return g * np.minimum(1.0, lam / np.maximum(n, 1e-300))


def stabilised_update(w, g, alpha, v, lam, R, wd):
    """One output-layer update for variant v. w, g: (..., p); alpha scalar; R broadcastable to w[..., :1]."""
    if v == "plain":
        return w - alpha * g
    if v == "clip_wd_coupled":
        return w - alpha * clip(g + wd * w, lam)
    w = w - alpha * clip(g, lam)
    if v == "clip_proj":
        n = np.linalg.norm(w, axis=-1, keepdims=True)
        w = w * np.minimum(1.0, R / np.maximum(n, 1e-300))
    elif v == "clip_wd":
        w = (1 - alpha * wd) * w
    return w


# ----------------------------------------------------------------------------- E1: w -> 2w
def run_e1(n_steps=10**6, alpha0=0.1, gamma=0.99, lam=1.0, R=10.0, wd=1e-2, w0=1.0):
    cfg = dict(experiment="E1 w->2w", phi_s1=1.0, phi_s2=2.0, r=0.0, gamma=gamma, w0=w0, n_steps=n_steps,
               alpha0=alpha0, schedules={"const": "alpha0", "decay": "alpha0 / n^0.6"}, clip_lambda=lam,
               proj_R=R, weight_decay=wd, overflow_threshold=OVERFLOW, variants=VARIANTS, dtype="float64")
    out = f"{ROOT}/results/E1"
    os.makedirs(out, exist_ok=True)
    al, logs = alphas(alpha0, n_steps), log_steps(n_steps)
    rows, summary = [], []
    for sch in SCHEDULES:
        for v in VARIANTS:
            w, a, overflow, N = np.array([w0]), al[sch], None, None
            # P1: after the first step N with (2 gamma - 1)|w| > lam, clipped TD adds exactly alpha_n * lam,
            # so |w_n| = |w_N| + lam * sum_{m=N}^{n-1} alpha_m (a lower bound in general).
            bound_base, cum = None, 0.0
            with np.errstate(over="ignore", invalid="ignore"):
                for n in range(1, n_steps + 1):
                    if N is None and (2 * gamma - 1) * abs(w[0]) > lam:
                        N, bound_base, cum = n, abs(w[0]), 0.0
                    g = -(0.0 + gamma * w * 2.0 - w * 1.0) * 1.0  # semi-gradient TD on s1 -> s2
                    w = stabilised_update(w, g, a[n - 1], v, lam, R, wd)
                    if N is not None:
                        cum += a[n - 1] * lam
                    if overflow is None and not abs(w[0]) <= OVERFLOW:
                        overflow = n
                    if n in logs:
                        rows.append(dict(schedule=sch, variant=v, step=n, w=w[0],
                                         p1_bound=bound_base + cum if N is not None else np.nan))
            summary.append(dict(schedule=sch, variant=v, final_abs_w=abs(w[0]), overflow_step=overflow, p1_N=N))
            print(summary[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/w_trace.csv", index=False)
    # P1 check: for clipping-only the realised |w_n| should equal the bound (up to rounding).
    p1 = df[(df.variant == "clip") & df.p1_bound.notna()].copy()
    p1["rel_gap"] = (p1.w.abs() - p1.p1_bound) / p1.p1_bound
    for s in summary:
        if s["variant"] == "clip":
            g = p1[p1.schedule == s["schedule"]]
            s["p1_min_rel_gap"], s["p1_max_abs_rel_gap"] = g.rel_gap.min(), g.rel_gap.abs().max()
    pd.DataFrame(summary).to_csv(f"{out}/summary.csv", index=False)
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)


# ----------------------------------------------------------------------------- E2: Baird
def baird_features():
    """Sutton & Barto (2nd ed.) Example 11.1: states 1-6 -> 2 e_i + e_8, state 7 -> e_7 + 2 e_8."""
    X = np.zeros((7, 8))
    for i in range(6):
        X[i, i], X[i, 7] = 2, 1
    X[6, 6], X[6, 7] = 1, 2
    return X


def run_e2(n_steps=10**6, alpha0=0.01, gamma=0.99, lam=1.0, R_mult=10.0, wd=1e-2, seeds=10, hidden=32):
    cfg = dict(experiment="E2 Baird", gamma=gamma, r=0.0, n_steps=n_steps, alpha0=alpha0,
               schedules={"const": "alpha0", "decay": "alpha0 / n^0.6"}, clip_lambda=lam,
               proj_R="R_mult x output-layer norm at init", R_mult=R_mult, weight_decay=wd, seeds=seeds,
               behaviour="dashed w.p. 6/7 (-> uniform over states 1-6), solid w.p. 1/7 (-> state 7)",
               target="always solid; rho = 7 (solid) or 0 (dashed)", state_dist="uniform over 7 states",
               linear=dict(features="S&B Ex. 11.1", w0=[1, 1, 1, 1, 1, 1, 10, 1]),
               net=dict(input="one-hot(7)", hidden=hidden, act="tanh", init="PyTorch default U(+-1/sqrt(fan_in))",
                        hidden_update="plain SGD, unclipped, same alpha", output_update="variant"),
               variants=VARIANTS, overflow_threshold=OVERFLOW, dtype="float64")
    out = f"{ROOT}/results/E2"
    os.makedirs(out, exist_ok=True)
    X, logs = baird_features(), log_steps(n_steps, 200)
    al = alphas(alpha0, n_steps)
    configs = [(v, s) for s in SCHEDULES for v in VARIANTS]  # leading axis C of every array
    C, S, H = len(configs), seeds, hidden
    rng_init = np.random.default_rng(12345)

    # ---- linear: all 8 weights are the "output layer"
    w = np.tile(np.array([1, 1, 1, 1, 1, 1, 10, 1.0]), (C, S, 1))
    R_lin = R_mult * np.linalg.norm(w[0, 0])
    # ---- net: one-hot -> tanh(W1 x + b1) -> w2 . h + b2. Per seed init, shared across configs.
    k1, k2 = 1 / np.sqrt(7), 1 / np.sqrt(H)
    W1 = np.tile(rng_init.uniform(-k1, k1, (1, S, H, 7)), (C, 1, 1, 1))
    b1 = np.tile(rng_init.uniform(-k1, k1, (1, S, H)), (C, 1, 1))
    out_p = np.tile(rng_init.uniform(-k2, k2, (1, S, H + 1)), (C, 1, 1))  # [w2, b2]
    R_net = R_mult * np.linalg.norm(out_p[0], axis=-1)[:, None]  # (S, 1): per-seed radius
    cfg["R_linear"], cfg["R_net_per_seed"] = R_lin, R_net[:, 0].tolist()

    rngs = [np.random.default_rng(s) for s in range(S)]
    ci, si = np.arange(C)[:, None], np.arange(S)[None, :]
    rows, overflow_lin, overflow_net = [], np.full((C, S), -1), np.full((C, S), -1)
    chunk = 10_000
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, n_steps, chunk):
            st = np.stack([r.integers(0, 7, chunk) for r in rngs], 1)  # (chunk, S) states ~ uniform
            solid = np.stack([r.random(chunk) < 1 / 7 for r in rngs], 1)
            for t in range(min(chunk, n_steps - start)):
                n = start + t + 1
                s, rho = st[t], np.where(solid[t], 7.0, 0.0)  # next state is 7 (index 6) under solid
                # ---------- linear
                x, x2 = X[s], X[6]
                delta = 0.0 + gamma * w @ x2 - (w * x).sum(-1)  # (C, S)
                g = -(rho * delta)[..., None] * x  # semi-gradient of the off-policy TD(0) update
                for c, (v, sch) in enumerate(configs):
                    w[c] = stabilised_update(w[c], g[c], al[sch][n - 1], v, lam, R_lin, wd)
                # ---------- net
                h = np.tanh(W1[ci, si, :, s] + b1)  # (C, S, H); one-hot input selects column s
                h2 = np.tanh(W1[..., 6] + b1)
                q = (out_p[..., :H] * h).sum(-1) + out_p[..., H]
                q2 = (out_p[..., :H] * h2).sum(-1) + out_p[..., H]  # stop-gradient target
                coef = -(rho * (0.0 + gamma * q2 - q))  # dLoss/dQ(s) for the semi-gradient
                g_out = coef[..., None] * np.concatenate([h, np.ones((C, S, 1))], -1)
                g_pre = coef[..., None] * out_p[..., :H] * (1 - h ** 2)  # (C,S,H) grad wrt W1[:, s] and b1
                for c, (v, sch) in enumerate(configs):
                    a = al[sch][n - 1]
                    W1[c, np.arange(S), :, s] -= a * g_pre[c]  # hidden layer: plain SGD
                    b1[c] -= a * g_pre[c]
                    out_p[c] = stabilised_update(out_p[c], g_out[c], a, v, lam, R_net, wd)
                # ---------- bookkeeping
                nl = np.linalg.norm(w, axis=-1)
                no = np.linalg.norm(out_p, axis=-1)
                nh = np.sqrt((W1 ** 2).sum((-1, -2)) + (b1 ** 2).sum(-1))
                overflow_lin[(overflow_lin < 0) & ~(nl <= OVERFLOW)] = n
                overflow_net[(overflow_net < 0) & ~(np.sqrt(no ** 2 + nh ** 2) <= OVERFLOW)] = n
                if n in logs:
                    for c, (v, sch) in enumerate(configs):
                        for k in range(S):
                            rows.append(dict(fclass="linear", variant=v, schedule=sch, seed=k, step=n,
                                             out_norm=nl[c, k], hidden_norm=0.0, total_norm=nl[c, k]))
                            rows.append(dict(fclass="net", variant=v, schedule=sch, seed=k, step=n,
                                             out_norm=no[c, k], hidden_norm=nh[c, k],
                                             total_norm=np.sqrt(no[c, k] ** 2 + nh[c, k] ** 2)))
            print(f"step {min(start + chunk, n_steps)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/norms.csv.gz", index=False)
    summ = []
    for fc, ov in (("linear", overflow_lin), ("net", overflow_net)):
        fin = df[(df.fclass == fc) & (df.step == df.step.max())]
        for c, (v, sch) in enumerate(configs):
            f = fin[(fin.variant == v) & (fin.schedule == sch)]
            summ.append(dict(fclass=fc, variant=v, schedule=sch,
                             out_mean=f.out_norm.mean(), out_std=f.out_norm.std(),
                             hidden_mean=f.hidden_norm.mean(), hidden_std=f.hidden_norm.std(),
                             total_mean=f.total_norm.mean(), total_std=f.total_norm.std(),
                             n_overflow=int((ov[c] > 0).sum()),
                             first_overflow=int(ov[c][ov[c] > 0].min()) if (ov[c] > 0).any() else None))
    pd.DataFrame(summ).to_csv(f"{out}/summary.csv", index=False)
    print(pd.DataFrame(summ).to_string())
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)


if __name__ == "__main__":
    which = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10**6
    run_e1(n) if which == "e1" else run_e2(n)
