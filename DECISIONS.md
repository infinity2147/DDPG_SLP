# DECISIONS

Every gate outcome and design choice, with the evidence (files under `results/`).

## Setup / design choices (2026-10-22)

- **Package versions**: Python 3.11, torch 2.6.0, gymnasium 1.1.1 (MuJoCo **v5** tasks everywhere), mujoco 3.16.0,
  dm_control 1.0.49 via shimmy 2.0.1. Pinned in `requirements.txt`.
- **Compute**: GPUs on the cluster are fully allocated by other users, so runs execute as single-thread CPU
  processes on DGX/A40 nodes (QoS: ≤4 dgx + ≤3 a40 jobs per user). Measured cost per update on CPU
  (smoke runs, Hopper): DDPG 14 ms, TD3 12 ms, no-target DDPG 8.5 ms, LN critic 10 ms, SAC 24 ms,
  ours/B-cheap 68 ms, ours/B-exact 62 ms, CrossQ 518 ms (2048-wide BN critic).
- **Fewer files than the suggested layout**: all agents live in `tdrl/agents.py` (two classes switched by config),
  evaluation + overestimation diagnostic in `tdrl/evaluate.py`, drift diagnostic inside `Base.update`.
  Method presets: `configs/methods.yaml`; experiment job lists: `scripts/experiments.py` (Python, not YAML, so grids stay short).
- **A2 decay is decoupled and applied after clipping** (θ ← (1 − lr·wd)θ). If the decay were inside the clipped
  update it would itself be clipped away; E1 confirms that coupled decay diverges exactly like clipping-only
  (`results/E1/summary.csv`, variant `clip_wd_coupled`).
- **Update clipping (Adam)** clips the realised output-layer delta (weights+bias as one vector) to ‖Δ‖ ≤ lr·λ.
- **Drift diagnostic** uses the noise-free target (no TD3 smoothing noise; SAC: fixed noise draw) so drift is not
  dominated by sampling noise. Twin critics: min over heads. P3 bound is logged with L_a estimated (i) locally
  (‖∇_a Q‖ at μ_k(s′)) and (ii) as the max over the probe set.
- **Overestimation**: Q1(s, μ(s)) vs discounted MC return (horizon 700, γ^700 < 1e-3) from restored simulator state
  for MuJoCo/Pendulum; DMC falls back to episode start states (task state such as targets is not in physics state).
- **Diverged runs** (non-finite critic loss or |Q| > 1e8) stop, are marked `DIVERGED`, get one evaluation with the
  NaN policy acting as zero action, and are counted in statistics with that result.
- **Pendulum** (50k steps): probe set fixed at 20k steps instead of 50k.
- **Shared runs**: E0's DDPG τ=0.005 / τ=1.0 runs are reused as DDPG (targets) / DDPG (no targets) wherever an
  experiment needs the identical configuration.

## Unit tests (Section 5) — PASS (13/13)

`tests/test_core.py`, log `slurm_logs/tests_360471.out`. FVP rel. err < 1e-5; CG solves SPD; P2 with exact pinv on a
1024-wide actor: cosine(JΔθ, q) = 1.0, scale = 1.0 (`results/unit_tests/p2.json`). Agreement degrades with damping
(cos 0.97 at λ_damp = 1e-2, 0.63 at 1) and when width is too small for full row rank (`results/unit_tests/p2_sweep.csv`).
Note: B-exact with 10 CG iterations and λ_damp = 1e-6 reaches only cos 0.74 (50 iters: 0.998); with the default
λ_damp = 1e-2 the system is better conditioned.

## Phase 1 — E1, E2 (premise check)

- **E1 (w→2w, α0 = 0.1, 10⁶ steps)**: clipping-only diverges: |w| = 1.0·10⁵ (constant step) and 63.6 and still
  growing (α0/n^0.6); realised |w_n| equals the P1 closed-form bound to relative error ≤ 1e-14. Plain TD overflows
  (const, step 7389) / reaches 4·10²⁶ (decaying). Projection holds |w| = R = 10; decoupled decay bounded at λ/wd ≈ 100.
  **Premise confirmed: clipping alone is not stable.**
- **E2 (Baird, 10 seeds, 10⁶ steps)**: linear plain TD overflows in 10/10 seeds (constant step). Clipping-only
  wanders to 59 ± 29 (max 107) with no bound; projection caps at R (=10×‖w0‖ = 103); decoupled decay keeps
  2.5 ± 1.1 (true solution w = 0). With the tanh net all variants stayed small (output norm < 1.5).

## Gate G1 — PROVISIONAL: stabiliser A1 (projection)

Both A1 and A2 keep weights bounded in E1/E2; the spec's tie-break prefers A1. E3 (stalling check) needs a trained
TD3 actor from E0 and will confirm or overturn this. R check (`G1R`, Hopper 300k, R ∈ {3,10,30,100}×init norm,
5 seeds) is running.
