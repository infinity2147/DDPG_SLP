"""One run = (method preset, env, seed, overrides) -> results dir with config.yaml, metrics.parquet, eval.csv,
overestimation.csv, ckpt/. Resumes from ckpt/ckpt.pt (saved every 100k steps); writes DONE when finished.

python -m tdrl.train --out results/E0/td3/Hopper-v5/seed_0 --method td3 --env Hopper-v5 --seed 0 [--set k=v ...]
"""
import argparse
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .agents import make_agent
from .buffer import ReplayBuffer
from .envs import make_env
from .evaluate import evaluate, overestimation, safe

ROOT = Path(__file__).resolve().parents[1]
RUN_DEFAULTS = dict(steps=1_000_000, learning_starts=10_000, expl_noise=0.1, utd=1, buffer_size=1_000_000,
                    eval_every=5_000, eval_episodes=10, log_every=1_000, probe_step=50_000, probe_size=1_024,
                    overest_every=50_000, ckpt_every=100_000, cosine_every=0, device="cpu", threads=1)


def resolve_config(method, env, seed, overrides):
    presets = yaml.safe_load(open(ROOT / "configs/methods.yaml"))
    cfg = {**RUN_DEFAULTS, **presets[method], "method": method, "env": env, "seed": seed}
    if env.startswith("Pendulum"):  # short task: 50k steps, probe set fixed earlier
        cfg.update(steps=50_000, probe_step=20_000, overest_every=10_000, ckpt_every=50_000)
    cfg.update(overrides)
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run(out, cfg):
    out = Path(out)
    if (out / "DONE").exists():
        return
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cfg["threads"])
    git = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    yaml.safe_dump({**cfg, "git_commit": git or "none"}, open(out / "config.yaml", "w"), sort_keys=False)

    seed, dev = cfg["seed"], cfg["device"]
    seed_all(seed)
    env, eval_env, mc_env = (make_env(cfg["env"], seed + k) for k in (0, 10_000, 20_000))
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = make_agent(obs_dim, act_dim, cfg, dev)
    buf = ReplayBuffer(obs_dim, act_dim, min(cfg["buffer_size"], cfg["steps"]), dev)
    rng = np.random.default_rng(seed)
    stochastic = cfg["algo"] in ("sac", "crossq")

    t0, updates, probe_idx, metrics, evals, overs = 0, 0, None, [], [], []
    ck = out / "ckpt/ckpt.pt"
    if ck.exists():  # resume; the interrupted episode is not restored, a fresh one starts
        d = torch.load(ck, weights_only=False)
        agent.load_state_dict(d["agent"])
        buf.load_state_dict(d["buffer"])
        t0, updates, probe_idx, metrics, evals, overs = (d[k] for k in ("t", "updates", "probe_idx", "metrics", "evals", "overs"))
        rng.bit_generator.state = d["np_rng"]
        torch.set_rng_state(d["torch_rng"])
        random.setstate(d["py_rng"])
        print(f"resumed at step {t0}", flush=True)

    def flush():
        pd.DataFrame(metrics).to_parquet(out / "metrics.parquet")
        pd.DataFrame(evals).to_csv(out / "eval.csv", index=False)
        pd.DataFrame(overs).to_csv(out / "overestimation.csv", index=False)

    acc, cnt, upd_time, n_timed = defaultdict(float), defaultdict(int), 0.0, 0
    t_wall, diverged = time.time(), False
    s, _ = env.reset(seed=seed + t0)
    for t in range(t0, cfg["steps"]):
        # ---- act ----
        if t < cfg["learning_starts"]:
            a = env.action_space.sample()
        elif stochastic:
            a = agent.act(s, deterministic=False)
        else:
            a = (agent.act(s) + cfg["expl_noise"] * rng.standard_normal(act_dim)).clip(-1, 1)
        s2, r, term, trunc, _ = env.step(safe(a))
        buf.add(s, a, r, s2, term, trunc)
        s = env.reset()[0] if (term or trunc) else s2
        if t + 1 == cfg["probe_step"]:  # fixed probe set for the target-drift diagnostic
            probe_idx = rng.choice(len(buf), cfg["probe_size"], replace=False)

        # ---- learn ----
        if t >= cfg["learning_starts"]:
            for _ in range(cfg["utd"]):
                updates += 1
                log = updates % cfg["log_every"] == 0
                b = buf.sample(agent.c["batch_size"], rng)
                probe = buf.get(probe_idx) if (log and probe_idx is not None) else None
                tic = time.perf_counter()
                st = agent.update(b, probe, log)
                if not log:  # wall-clock per update, excluding diagnostic (logging) updates
                    upd_time += time.perf_counter() - tic
                    n_timed += 1
                for k, v in st.items():
                    acc[k] += v
                    cnt[k] += 1
                if cfg["cosine_every"] and updates % cfg["cosine_every"] == 0:
                    acc["cos_exact_cheap"] += agent.compare_directions(buf.sample(256, rng))
                    cnt["cos_exact_cheap"] += 1
                if log:
                    row = {k: acc[k] / cnt[k] for k in acc}
                    row.update(step=t + 1, updates=updates, time_per_update=upd_time / max(n_timed, 1),
                               wall=time.time() - t_wall)
                    metrics.append(row)
                    acc, cnt, upd_time, n_timed = defaultdict(float), defaultdict(int), 0.0, 0
                    if not np.isfinite(row.get("critic_loss", 0.0)) or abs(row.get("q_mean", 0.0)) > 1e8:
                        diverged = True

        # ---- evaluate / diagnose / checkpoint ----
        if (t + 1) % cfg["eval_every"] == 0 or diverged:
            rets, _ = evaluate(eval_env, agent.act, cfg["eval_episodes"], agent.c["gamma"])
            evals.append(dict(step=t + 1, ret_mean=rets.mean(), ret_std=rets.std(), returns=list(rets.round(2))))
            print(f"{out} step {t + 1} return {rets.mean():.1f}", flush=True)
            flush()  # keep logs on disk current so partial runs can be monitored
        if t >= cfg["learning_starts"] and ((t + 1) % cfg["overest_every"] == 0 or diverged):
            overs.append(dict(step=t + 1, **overestimation(mc_env, agent, rng=rng)))
        if diverged:  # a diverged run is a result: keep it and mark it (see DECISIONS.md)
            (out / "DIVERGED").write_text(f"step {t + 1}\n")
            break
        if (t + 1) % cfg["ckpt_every"] == 0 and t + 1 < cfg["steps"]:
            flush()
            atomic_save(dict(agent=agent.state_dict(), buffer=buf.state_dict(), t=t + 1, updates=updates,
                             probe_idx=probe_idx, metrics=metrics, evals=evals, overs=overs,
                             np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                             py_rng=random.getstate()), ck)
    flush()
    torch.save(dict(agent=agent.state_dict(), n_params=agent.n_params()), out / "ckpt/final.pt")
    ck.unlink(missing_ok=True)
    (out / "DONE").write_text(f"{time.time() - t_wall:.0f} s\n")


def parse(v):
    """YAML value, but '1e-3' (a string in YAML 1.1) becomes a float."""
    v = yaml.safe_load(v)
    try:
        return float(v) if isinstance(v, str) else v
    except ValueError:
        return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], help="overrides k=v (values parsed as YAML)")
    a = p.parse_args()
    overrides = {k: parse(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    run(a.out, resolve_config(a.method, a.env, a.seed, overrides))


if __name__ == "__main__":
    main()
