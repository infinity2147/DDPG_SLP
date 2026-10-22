"""Job lists for every experiment. `python scripts/experiments.py E0 E4` writes queue/E0.jsonl, queue/E4.jsonl.
Runs shared between experiments are defined once and reused by the analysis (see DECISIONS.md)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS5, SEEDS10 = list(range(5)), list(range(10))
TUNE = ["HalfCheetah-v5", "Hopper-v5"]
E5_ENVS = TUNE + ["Walker2d-v5"]


def jobs(exp, tag, method, envs, seeds, **overrides):
    return [dict(out=f"results/{exp}/{tag}/{e}/seed_{s}", method=method, env=e, seed=s, set=overrides)
            for e in envs for s in seeds]


def E0():  # baselines + DDPG tau sweep (tau = 1.0 is the no-target preset)
    J = jobs("E0", "td3", "td3", TUNE, SEEDS5) + jobs("E0", "sac", "sac", TUNE, SEEDS5)
    J += jobs("E0", "ddpg_tau0.005", "ddpg", TUNE, SEEDS5)
    J += jobs("E0", "ddpg_tau1.0", "ddpg_notarget", TUNE, SEEDS5)
    for tau in (0.001, 0.05):
        J += jobs("E0", f"ddpg_tau{tau}", "ddpg", TUNE, SEEDS5, tau=tau)
    return J


def pilot():  # exploratory short runs to look at curves/diagnostics early; not used for any claim
    J = []
    for m in ("ddpg", "ddpg_notarget", "td3", "ours", "ours_exact", "ours_adam_actor"):
        J += jobs("pilot", m, m, ["Pendulum-v1"], [0, 1, 2])
    for m in ("ddpg", "ddpg_notarget", "td3", "ours"):
        J += jobs("pilot", m, m, ["Hopper-v5"], [0, 1, 2], steps=60_000, probe_step=20_000,
                  overest_every=20_000, ckpt_every=30_000)
    return J


def G1R():  # gate G1: projection radius R in {3,10,30,100} x init output-layer norm, Hopper 300k
    return [j for R in (3, 10, 30, 100)
            for j in jobs("G1R", f"ours_R{R}", "ours", ["Hopper-v5"], SEEDS5, steps=300_000, R_mult=float(R))]


def E4():  # first full runs, 500k (Pendulum 50k). DDPG on HalfCheetah/Hopper = E0 runs truncated at 500k.
    J = jobs("E4", "ours", "ours", TUNE + ["Pendulum-v1"], SEEDS5, steps=500_000, threads=2)
    for m in ("ddpg", "ddpg_notarget"):
        J += jobs("E4", m, m, ["Pendulum-v1"], SEEDS5)
    return J


def E5():  # target drift (key). DDPG with/without targets on HalfCheetah/Hopper = E0 runs.
    J = jobs("E5", "ddpg", "ddpg", ["Walker2d-v5"], SEEDS5) + jobs("E5", "ddpg_notarget", "ddpg_notarget", ["Walker2d-v5"], SEEDS5)
    J += jobs("E5", "ours_adam_actor", "ours_adam_actor", E5_ENVS, SEEDS5, threads=2)
    J += jobs("E5", "ours", "ours", E5_ENVS, SEEDS5, threads=2, cosine_every=10_000)  # also E6 cosine
    J += jobs("E5", "ours_exact", "ours_exact", E5_ENVS, SEEDS5, threads=2)  # also E6
    return J


def E7():  # constraint set; C = both is E5 'ours'
    return [j for c in ("s", "sprime") for j in jobs("E7", f"ours_C{c}", "ours", E5_ENVS, SEEDS5, threads=2, cset=c)]


EXPERIMENTS ={k: v for k, v in globals().items() if (k[:1] in "EGX" or k == "pilot") and callable(v)}

if __name__ == "__main__":
    (ROOT / "queue").mkdir(exist_ok=True)
    for name in sys.argv[1:]:
        J = EXPERIMENTS[name]()
        with open(ROOT / f"queue/{name}.jsonl", "w") as f:
            f.writelines(json.dumps(j) + "\n" for j in J)
        print(name, len(J), "jobs")
