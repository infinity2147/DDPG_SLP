# Research Execution Prompt: Targetless DDPG with a Trust-Region Actor

You are acting as a research engineer and scientist. Your job is to implement, run and analyse a full set of reinforcement-learning experiments on this machine's GPUs, and then write a rigorous research report from the results. Work autonomously through the phases below. Stop and ask me only when you hit a blocker described in Section 12.

Read this whole document before writing any code.

---

## 1. Research context (what we are testing and why)

### 1.1 The problem
DDPG trains a critic Q_φ(s, a) by temporal-difference (TD) learning toward a target

  y = r + γ · Q(s′, μ(s′))

that is built from the network's own predictions. Combined with neural-network function approximation and off-policy replay data (the "deadly triad"), this can diverge. Standard DDPG/TD3/SAC stabilise it with **target networks**: slow Polyak-averaged copies φ⁻, θ⁻ (τ ≈ 0.005) used only to compute y. Target networks slow learning, add memory and a hyperparameter (τ), and give no guarantee.

### 1.2 Prior work this builds on
- **Output-layer clipping** (Ramaswamy, Bhatnagar & Saxena, arXiv 2305.12125): clip the gradient of only the critic's last layer, train the hidden layers with plain SGD, use squashing activations, and drop the target network. They claim this guarantees bounded weights.
  - **Known gap:** bounded *steps* do not imply bounded *weights* when Σ step sizes = ∞. In the "w → 2w" example (Section 6, E1), clipped TD still diverges, just slowly. **Do not assume clipping alone is stable. Experiment E1 tests this directly.**
- **LayerNorm + ℓ2 critic** (Gallici et al., "Simplifying Deep Temporal Difference Learning", arXiv 2407.04811): provably stable target-free TD for value-based RL.
- **CrossQ** (Bhatt et al., ICLR 2024): BatchNorm critic, no target networks.
- **Natural deterministic policy gradient** (Silver et al., 2014): the metric E[∇θμ ∇θμᵀ], which our trust region uses.

### 1.3 Our method (the thing under test)
We remove **both** target networks and control the moving target from two sides.

**Component A: a target-free, stabilised critic.**
- Target: y_i = r_i + γ(1 − done_i) · sg[Q_φ(s′_i, μ_θ(s′_i))], where sg = stop-gradient (semi-gradient TD). Use the live networks, not copies.
- Hidden layers: normal update.
- Output layer (final linear layer: weights + bias): norm-clipped update, clip(v, λ) = v · min(1, λ/‖v‖₂).
- Plus one **stabiliser** (chosen in Phase 1):
  - **A1, projection:** after each step, if the output-layer weights' norm exceeds R, rescale them onto the ball of radius R. Variant A1-all projects every layer onto its own ball.
  - **A2, regularisation:** LayerNorm after every hidden layer, plus ℓ2 weight decay on the output layer.

**Important implementation subtlety.** Adam normalises gradient scale, so clipping the gradient *before* Adam barely bounds the step. Implement clipping on the **update** (the parameter delta Δ = θ_new − θ_old produced by the optimiser), not only on the raw gradient. Provide both optimisers:
- `critic_opt=sgd`: SGD with momentum 0.9; clip the gradient (faithful to the theory).
- `critic_opt=adam`: Adam; clip the output-layer update Δ to norm ≤ lr·λ.

Default to Adam with update clipping for benchmark runs. Use the SGD version in toy and theory-validation experiments.

**Component B: a trust-region actor.**
Each actor update solves, on the *updated* critic Q_{φ_{k+1}}:

  maximise (1/M) Σ_i Q(s_i, μ_θ(s_i))  subject to  (1/M) Σ_{s ∈ C} ‖μ_θ(s) − μ_{θ_k}(s)‖² ≤ ε

where C is the constraint state set (Section 1.4). Actions are in the normalised space [−1, 1]^d, so ε is a mean squared action change. For example, ε = 1e-3 means an RMS change of about 0.032 per action.

Two implementations:
- **B-exact (natural gradient + CG).**
  - Let J_i = ∂μ(s_i)/∂θ and q_i = ∇_a Q(s_i, a) at a = μ(s_i).
  - Gradient: g = (1/M) Σ J_iᵀ q_i.
  - Metric: F = (1/M) Σ J_iᵀ J_i (rank ≤ M·d, so it is singular). Use F + λ_damp·I.
  - Solve (F + λ_damp I) x = g with 10–20 conjugate-gradient iterations. Compute F·v via `torch.func.jvp` followed by `vjp` (J v, then Jᵀ(J v)); never materialise F.
  - Step size: β = sqrt(ε / (xᵀ F x)).
  - Backtracking line search: θ_new = θ_k + 0.5^m β x for m = 0…10, accepting the first m where both (i) the batch-mean Q improves and (ii) the true (non-linearised) constraint holds. If none is accepted, skip the actor update and log it.
- **B-cheap (action-space regression).** For wide networks the minimum-norm natural step moves each action along its own critic slope.
  - Compute target actions a_i⁺ = μ_k(s_i) + η q_i, with η = sqrt(ε / ((1/M) Σ‖q_i‖²)). Do not clip a_i⁺ to the action bounds; let tanh handle it and log how often a_i⁺ falls outside [−1, 1].
  - Take K (default 10) SGD/Adam steps minimising (1/M) Σ ‖μ_θ(s_i) − a_i⁺‖² starting from θ_k.
  - Run the same backtracking check by interpolating parameters θ_k + 0.5^m (θ_reg − θ_k).

**Component C: constraint on next states.** The TD target uses μ(s′), so the actor's change at s′ is what moves the target. Constraint set options:
- `C=s` (current states)
- `C=sprime` (next states)
- `C=both`

When C includes s′, also include those states in the objective batch, so q is available for them in B-cheap.

**Component D: overestimation.** Optional twin critics with min(Q1, Q2) in the target, both target-free, each with Component A applied.

### 1.4 The central thesis to test
The per-step change in the TD target satisfies

  |Δy_i| ≤ γ ( |ΔQ at s′_i from the critic step| + L_a · ‖Δμ(s′_i)‖ ).

Component A bounds the first term; Component B (with s′ in C) bounds the second. **Claim: the actor's trust region plus a bounded critic step controls target drift about as well as a target network does, so the target network becomes unnecessary.** Experiment E5 is the key test. Report its outcome honestly whether or not it supports the claim.

### 1.5 Research questions
| # | Question |
| --- | --- |
| RQ1 | Does output-layer clipping alone keep critic weights bounded? (Expected: no.) |
| RQ2 | Can target-free DDPG with Components A+B match DDPG/TD3 with targets? |
| RQ3 | Does the trust-region actor reduce target drift to target-network levels? |
| RQ4 | Does B-cheap match B-exact at lower cost? |
| RQ5 | Is the method less sensitive to hyperparameters and seeds than DDPG? |
| RQ6 | Does constraining s′ matter compared with constraining only s? |

---

## 2. Ground rules (non-negotiable)

1. **Never fabricate or estimate a result.** Every number, plot and claim in the report must come from logged data on disk, produced by scripts in the repo. If a run did not finish, say so.
2. **Report negative results.** If the hypothesis fails, the report says so clearly and analyses why. A clean negative result is a valid outcome.
3. **Fair comparisons.** Every method gets the same tuning budget, the same network sizes (unless the method defines its own architecture, e.g. CrossQ), the same environment steps, the same evaluation protocol and the same seeds.
4. **Tune only on tuning tasks.** Use Hopper and HalfCheetah for tuning. Freeze hyperparameters before running the remaining tasks, and record the freeze in `DECISIONS.md`.
5. **Reproducibility.** Seed everything (python, numpy, torch, environment). Save the resolved config of every run next to its logs. Pin package versions in `requirements.txt` / `environment.yml`. Record the git commit hash in every run directory.
6. **Validate before scaling.** No large sweep starts until its baseline passes the relevant gate.
7. **Keep a written trail.** Maintain `PROGRESS.md` (status, what is running, what is done) and `DECISIONS.md` (every gate outcome, every design choice, with the evidence).

---

## 3. Environment and tooling setup

1. Inspect the machine: GPUs (`nvidia-smi`), CPU count, RAM and disk. Write a short summary to `PROGRESS.md`. Plan concurrency from it: typically 4–8 single-seed runs per GPU for 256-unit MLPs; measure, don't guess.
2. Create a Python environment with:
   - PyTorch (CUDA)
   - `gymnasium[mujoco]` (MuJoCo v4 or v5 tasks; pick one version and use it everywhere)
   - `dm_control` with a Gymnasium wrapper (`shimmy` or your own thin wrapper)
   - numpy, pandas, pyarrow, matplotlib, seaborn, `rliable`, scipy, tqdm, pyyaml or hydra
   - optionally `wandb`, but **always** also log locally
3. Verify that MuJoCo renders headless and that every target environment can be stepped.
4. Run unit tests (Section 5) before any training.

---

## 4. Codebase structure

Build one clean, single-package codebase so all methods share everything except their defining differences.

```
tdrl/
  envs.py            # make_env(name, seed): MuJoCo + DMC, action normalised to [-1,1], observation float32
  buffer.py          # replay buffer (1M), stores done/truncated separately (bootstrap on truncation)
  networks.py        # MLP actor (tanh output), MLP critic(s); options: LayerNorm, BatchNorm (CrossQ), activation
  stabilisers.py     # output-layer update clipping, projection (A1/A1-all), weight decay (A2)
  trust_region.py    # B-exact (jvp/vjp Fisher-vector product, CG, line search) and B-cheap
  agents/
    ddpg.py          # flags: use_target_nets, tau
    td3.py
    sac.py
    crossq.py
    ln_critic.py     # DDPG-style actor + LayerNorm/l2 critic, no targets (PQN-style adaptation)
    ours.py          # Components A-D, all switchable by config
  diagnostics.py     # target drift, Q-vs-MC gap, weight norms, action change, timing
  train.py           # one run = (method, env, seed, config) -> logs + checkpoints
  evaluate.py
configs/             # yaml per method and per experiment
scripts/
  launch.py          # job queue: runs N jobs per GPU, resumes unfinished runs, skips finished ones
  toy_w2w.py
  toy_baird.py
  policy_eval.py     # E3
  analysis/          # one script per figure/table; reads logs only
results/             # results/<experiment>/<method>/<env>/seed_<k>/{config.yaml, metrics.parquet, eval.csv, ckpt/}
report/
tests/
PROGRESS.md
DECISIONS.md
```

**Default hyperparameters** (shared unless an experiment says otherwise):

| Setting | Value |
| --- | --- |
| Hidden layers | 2 × 256, ReLU (squashing variant: tanh) |
| Batch size | 256 |
| Replay buffer | 1,000,000 |
| γ | 0.99 |
| Learning rates | 3e-4 (actor and critic, Adam) |
| Random steps before learning | 10,000 |
| Updates per environment step | 1 |
| Exploration | Gaussian noise σ = 0.1 (DDPG/TD3/ours) |
| Target-network τ (where used) | 0.005 |
| Training length | 1M environment steps (Pendulum: 50k) |
| Evaluation | every 5k steps, 10 deterministic episodes |

Defaults for ours: λ (clip) = 1.0, R chosen in Phase 1, ε = 1e-3, λ_damp = 1e-2, K = 10 regression steps, C = both.

**Log every 1,000 updates** (to `metrics.parquet`):
- critic loss
- mean Q on the batch
- output-layer and total critic weight norms (A1-all: per layer)
- actor and critic gradient/update norms
- fraction of clipped critic updates
- trust-region stats: accepted backtracking step m, skipped updates, realised mean ‖Δμ‖² on s and on s′, the linearised vs actual constraint value
- B-exact only: CG residual
- B-cheap only: fraction of a⁺ outside [−1, 1]
- **target drift** (below)
- wall-clock time per update

**Target-drift diagnostic.** Fix a probe set of 1,024 transitions sampled from the buffer at step 50k and kept for the whole run. At each log step, compute y on the probe set immediately before and after one full update (critic + actor), and log mean and 95th percentile of |Δy|. For target-network methods, y uses the target networks, which is the quantity that actually feeds the critic loss.

**Overestimation diagnostic.** Every 50k steps, take 20 states from fresh evaluation episodes. Compare Q(s, μ(s)) with the Monte Carlo discounted return of the deterministic policy rolled out from that exact state, using environment state restore where the simulator supports it. If restore is not supported, use episode start states and discounted returns from the first step. Log mean bias and mean absolute error.

---

## 5. Unit tests (must pass before Phase 1)

1. Fisher-vector product: on a tiny actor, build F explicitly from full Jacobians and check that the jvp/vjp product matches F·v (relative error < 1e-5, float64).
2. CG: solves a random SPD system to tolerance.
3. **Result P2 numerical check.** On a wide (1024-unit) actor with full-row-rank Jacobian (M·d < number of parameters), confirm that the exact minimum-norm natural step (pseudo-inverse, float64) produces Δμ(s_i) ∝ q_i with the predicted scale η. Report the cosine similarity between JΔθ and q (expected > 0.999). Then show how agreement degrades as damping grows or width shrinks. Save this as a small figure for the report.
4. Line search: accepts the full step on a quadratic toy and rejects a step that violates the constraint.
5. Projection keeps norm ≤ R; update clipping keeps ‖Δ‖ ≤ lr·λ.
6. Stop-gradient: the critic loss gradient has no contribution through y.
7. Buffer: bootstraps through time-limit truncation and not through terminal states.

---

## 6. Experiments (run in this order; each phase ends with a gate)

Seeds: 5 per configuration in Phases 0–3, 5 and 6; **10** in Phase 4. Use the same seed list for every method.

### Phase 0 — Baselines and "seeing the holes"

**E0. Reproduce baselines and measure DDPG's holes.**
- Train DDPG, TD3 and SAC on HalfCheetah and Hopper for 1M steps.
- Run DDPG with τ ∈ {0.001, 0.005, 0.05, 1.0}; τ = 1.0 means no target networks.
- Record overestimation bias, target drift, action-change spikes, and the seed spread across 5 seeds.
- **Gate G0:** TD3 and SAC final returns fall within the range of public reference results (CleanRL benchmark curves, original papers) on both tasks. If not, debug the implementation and do not proceed. Record the comparison in `DECISIONS.md`.

### Phase 1 — Toy problems: which critic stabiliser?

**E1. "w → 2w".**
- Setup: two states with features φ(s₁) = 1 and φ(s₂) = 2; Q = w·φ; train only on s₁ → s₂, r = 0, γ = 0.99.
- Variants: plain semi-gradient TD; clipping only; clipping + projection (R = 10); clipping + weight decay (1e-2).
- Step sizes α_n = α₀ / n^0.6, plus a constant-step version; 10⁶ steps; w₀ = 1.
- Plot |w| (log scale).
- Also prove the divergence of clipping-only in closed form for the report (Result P1): once (2γ−1)|w| > λ, every step adds exactly α_n·λ, so |w_n| ≥ |w_N| + λ Σ_{m=N}^{n} α_m → ∞.

**E2. Baird's counterexample.**
- Use the standard 7-state star MDP and features (Sutton & Barto, 2nd ed., Example 11.1).
- Same four variants as E1, both with linear features and with a 2-layer tanh network (32 units) that takes one-hot states.
- Plot the weight norm over 10⁵–10⁶ steps; 10 seeds.

- **Gate G1:** choose the stabiliser for all later phases. It must keep weights bounded in E1 and E2 without stalling learning in E3. Prefer A1 if A1 and A2 tie, since A1 is simpler to prove. Also choose R:
  - Run a short check over R ∈ {3, 10, 30, 100} × the output-layer norm at initialisation, on Hopper for 300k steps.
  - Choose the smallest R that does not reduce return.

### Phase 2 — The critic in isolation

**E3. Target-free policy evaluation.**
- Freeze a trained TD3 actor from E0 (HalfCheetah and Hopper).
- Fill a 1M buffer with its behaviour (action noise σ = 0.1, so the data is off-policy relative to the deterministic policy).
- Learn Q^μ with:
  - target network τ = 0.005
  - no target network
  - no target + clipping only
  - no target + chosen stabiliser
  - no target + A2 (whichever stabiliser was not chosen)
- Ground truth: Monte Carlo discounted returns of μ from 200 held-out states.
- Plot the relative Q error over 500k updates.

**E4. First full runs.**
- Pendulum, HalfCheetah and Hopper, 500k steps (Pendulum 50k).
- Compare DDPG (targets), DDPG (no targets), and ours (A + B-cheap + C = both).
- **Gate G2:** ours beats DDPG-no-targets on at least 2 of the 3 tasks (IQM, non-overlapping 95% CIs or a probability of improvement > 0.75).
  - If it fails, debug in this order: clip λ too small, R too small, ε too small, actor regression K, learning rates.
  - Log each attempt in `DECISIONS.md`.

### Phase 3 — The actor and the central thesis

**E5. Target drift (the key experiment).**
- HalfCheetah, Hopper and Walker2d; 1M steps.
- Methods: DDPG (targets); DDPG (no targets); stabilised critic + Adam actor; stabilised critic + trust-region actor (B-cheap); the same with B-exact.
- Plot mean |Δy| over training (log y-axis) and its relation to return.
- Also plot drift split into its critic part and actor part. Compute the critic part by applying only the critic step and recomputing y.

**E6. B-exact vs B-cheap.**
- Same tasks.
- Measure: returns; time per update; cosine similarity between the two parameter steps on a shared batch, logged every 10k updates by running both from the same θ_k.
- Also measure realised constraint satisfaction for each.

**E7. Constraint set.**
- C ∈ {s, sprime, both}, with B-cheap, on HalfCheetah, Hopper and Walker2d.
- Compare drift and return.

- **Gate G3:** freeze the final method.
  - Use B-cheap if its IQM is within B-exact's CI and it is at least 2× faster.
  - Choose C with the lowest drift among options without a significant return loss.
  - Then run a fixed-budget hyperparameter tuning of ours **and** of each baseline on Hopper + HalfCheetah (same number of configurations each, e.g. 12) and freeze everything.

### Phase 4 — Main benchmark

**E8. Main comparison**, 1M steps, 10 seeds.
- **Tasks:**
  - MuJoCo: HalfCheetah, Hopper, Walker2d, Ant, Humanoid.
  - DMC: cheetah-run, walker-walk, walker-run, quadruped-run, finger-spin, reacher-hard, hopper-hop.
  - If compute runs short, drop DMC tasks before MuJoCo ones and record this.
- **Methods:**
  - DDPG (targets)
  - TD3
  - SAC
  - DDPG (no targets)
  - LN+ℓ2 critic, no targets
  - CrossQ
  - ours (frozen)
  - ours + twin critics
- **Outputs:**
  - learning curves per task (mean ± 95% bootstrap CI)
  - final-return table
  - aggregate metrics with `rliable`: IQM, optimality gap, performance profiles, probability of improvement of ours vs each baseline
- **Normalisation:** use raw returns for DMC (0–1000). For MuJoCo, use (score − random) / (TD3-final − random), with random-policy scores measured by you.

**E13. Compute cost** (runs alongside E8). For every method on HalfCheetah, on identical hardware with no other jobs on that GPU, measure:
- time per update
- wall-clock time to 1M steps
- peak GPU memory
- parameter count, including target copies

### Phase 5 — Ablations (HalfCheetah, Hopper, Walker2d, Ant; 5 seeds)

**E9. Component removal.**
- Full method
- No stabiliser (clipping only)
- No clipping (stabiliser only)
- No trust region (Adam actor)
- No critic fix
- Trust region with C = s only

**E10. Overestimation.** Single vs twin critics, both target-free. Report Q bias and MAE over training.

**E11. Targets back.** Ours with target networks added (τ = 0.005), compared with ours without them.

### Phase 6 — Robustness (Hopper, Walker2d; 5 seeds)

**E12. Sensitivity sweeps**, one at a time around the frozen config:
- λ ∈ {0.1, 0.3, 1, 3, 10, 100}
- ε ∈ {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}
- critic lr ∈ {1e-4, 3e-4, 1e-3}
- R multipliers ∈ {3, 10, 30, 100}

Run the same lr sweep for DDPG and TD3. Report the final IQM vs each value. Also report seed spread (std and interquartile range of final return across the 10 seeds from E8) for ours vs DDPG vs TD3.

### Phase 7 — Pushing the research direction forward

Phases 0–6 test the method as specified. This phase tries to **advance** it. Do this phase only after the Phase 4 runs are complete or queued. Treat it as exploratory: 3 tasks (Hopper, Walker2d, Ant), 5 seeds, 500k–1M steps. Clearly label every result from it as exploratory in the report.

Run the extensions below in priority order. For each, write a one-paragraph hypothesis in `DECISIONS.md` before running it and a one-paragraph verdict after.

**X1. Drift-targeted adaptive trust region.** The thesis says drift is what matters, so control it directly.
- Pick a drift budget D, for example the median |Δy| that DDPG-with-targets showed in E5.
- After each update, adapt ε multiplicatively: ε ← ε·1.05 if measured drift < D, else ε ← ε/1.05, clipped to [1e-6, 1e-1].
- Hypothesis: this removes ε as a hyperparameter and matches or beats the best fixed ε from E12.

**X2. Trust region on the critic too (symmetric version).**
- Constrain the critic step so that the mean squared change of Q at the next-state batch, (1/M) Σ (Q_new(s′, μ(s′)) − Q_old(s′, μ(s′)))², stays ≤ ε_Q.
- Implement it with the same line-search machinery: scale the critic update back until the constraint holds.
- Compare against Component A (clipping + stabiliser) and against A + X2.
- Hypothesis: this bounds the critic term in the drift bound directly and may make the stabiliser unnecessary in practice. (Keep the stabiliser for the theory.)

**X3. High replay ratio.**
- Target networks are especially costly when there are many updates per environment step, because the targets lag further behind.
- Run updates-per-step ∈ {1, 4, 8} for ours, TD3 and CrossQ at 300k steps.
- Hypothesis: our advantage over target-network methods grows with the replay ratio.

**X4. Transfer to a stochastic actor (SAC-style).**
- Replace the squared-action-change constraint with the KL divergence between the old and new Gaussian policies, averaged over s and s′, and remove SAC's target networks.
- Hypothesis: the same "actor trust region replaces the target network" principle holds beyond deterministic policies, which would make the contribution more general.

**X5. Theory progress on P5.**
- Attempt the two-timescale convergence argument for ours with A1-all projection, the critic on the fast timescale and the actor on the slow one, using Borkar's two-timescale framework.
- Write down exactly which assumption fails or is unverifiable if the proof does not close.
- Run a numerical check on a small linear-quadratic control problem where Q is known in closed form: confirm the iterates approach the predicted limit.

**X6. Your own ideas (optional).** If the earlier phases reveal a clear failure mode or a promising signal, design one follow-up experiment to probe it. State the hypothesis first, keep it small, and label it as exploratory.

---

## 7. Theory work (in parallel; goes in the report)

Write clean proofs, each verified numerically where possible:
- **P1:** clipped output-layer TD diverges on w → 2w (closed form, verified by E1).
- **P2:** under full row rank and zero damping, the minimum-norm natural trust-region step gives Δμ(s_i) = η q_i with η = sqrt(ε / mean‖q‖²). Also describe the damped case, which interpolates towards plain gradient ascent. Verified by unit test 3.
- **P3:** the target-drift bound in Section 1.4, with explicit constants under Lipschitz assumptions. Verified against E5's measured drift: plot the bound against the measurement.
- **P4 (sketch, clearly labelled as such):** boundedness of all weights under A1-all projection, and the conditions needed under A2.
- **P5:** leave as a stated conjecture (two-timescale convergence) unless you can prove it rigorously. Do not present a sketch as a theorem.

---

## 8. Compute management

- Use `scripts/launch.py` as a persistent job queue. Write the full job list per phase to a file, run K jobs per GPU, mark completion with a sentinel file, and make reruns skip completed jobs and resume crashed ones from checkpoints (checkpoint every 100k steps).
- After the first 3 runs of E8, measure throughput and project total wall-clock time for all remaining phases. Write the projection to `PROGRESS.md`.
- If the projection exceeds the available time, cut in this order:
  1. DMC tasks beyond 4
  2. E12 sweep breadth
  3. ablation tasks to 3
  4. Humanoid
  - Never cut seeds in E8 below 10 or drop baselines. Record every cut.
- Monitor for NaNs/divergence. A diverged run is a **result**, not an error: keep it, mark it, and count it in statistics (as its final return, with divergence recorded). Do not silently rerun diverged seeds.

---

## 9. Analysis rules

- All statistics use per-seed final performance, defined as the average evaluation return over the last 5 evaluation points.
- Aggregates use `rliable` with stratified bootstrap (≥ 2,000 resamples) and 95% CIs.
- When claiming "A beats B", report the probability of improvement and the CI. Avoid p-value fishing.
- Every figure is produced by a script in `scripts/analysis/` that reads only `results/`. Save figures as PDF + PNG, 300 dpi, colour-blind-safe palette, with the same colour for the same method across all figures.

---

## 10. The report

Produce `report/report.md` and build `report/report.pdf` (pandoc + LaTeX, or write LaTeX directly). Target 12–20 pages plus appendix. Structure:

1. **Abstract** (≤ 200 words), with the headline numbers.
2. **Introduction:** the moving-target problem, the cost of target networks, the gap in the clipping approach, our thesis, and a contribution list. Each contribution must be backed by a section.
3. **Background:** TD learning, the deadly triad, DDPG, target networks, trust regions, the natural deterministic policy gradient. Keep it concise.
4. **Related work:** target-free methods (clipping, LN+ℓ2, CrossQ, DeepMellow, constrained TD, NGDQN); trust-region and natural-gradient off-policy methods. Say clearly what is new here.
5. **Method:** Components A–D, the full pseudo-code (algorithm box), and implementation details (update clipping with Adam, CG, B-cheap, line search).
6. **Theory:** P1–P3 with proofs, P4 sketch, P5 as a conjecture. Each is linked to its numerical verification.
7. **Experimental setup:** environments, baselines, tuning protocol, seeds, evaluation, statistics, hardware, total GPU-hours actually used.
8. **Results**, one subsection per research question, each opening with a one-sentence answer:
   - RQ1: Figures F1 (w → 2w) and F2 (Baird).
   - Critic in isolation: Figure F3 (E3 Q-error).
   - RQ3, the key result: Figure F4 (target drift curves), and F4b (drift split into critic and actor parts, with the P3 bound overlaid).
   - RQ4: Figure F5 (exact vs cheap: returns, cosine similarity, time).
   - RQ6: Figure F6 (constraint set).
   - RQ2: Figure F7 (learning-curve grid for all tasks), Figure F8 (IQM / optimality gap / performance profiles / probability of improvement), Table T2 (final returns, mean ± CI).
   - Ablations: Figure F9 (ablation bars), Figure F10 (overestimation bias over training).
   - RQ5: Figure F11 (sensitivity curves), Figure F12 (seed spread).
   - Cost: Table T3 (compute and memory).
9. **Discussion:** what worked, what did not, why, and when to use the method.
9b. **Research progress and next directions (Phase 7):**
    - X1–X6 results, each with hypothesis, outcome and a verdict: promising / inconclusive / negative.
    - A ranked list of the 3 most promising next steps for the research direction, each with the evidence that motivates it and the first experiment to run.
    - A short paragraph on what the strongest paper from this work would claim, given the results actually obtained.
10. **Limitations and negative results:** all failed gates, diverged runs, cut experiments, and caveats on the theory.
11. **Conclusion.**
12. **Appendix:**
    - Table T1 (all hyperparameters, per method)
    - tuning grids and chosen values
    - per-task tables
    - all unit-test results including the P2 figure
    - the `DECISIONS.md` log, summarised
    - reproduction commands

**Writing rules:**
- Every number in the text appears in a figure or table, or is traceable to a file in `results/`.
- Keep the voice neutral; no hype words.
- If a hypothesis failed, the abstract says so.
- Include a short "How to reproduce" section with exact commands.

Also produce `report/summary.md`: a one-page plain-language summary of the findings for a non-specialist reader. It should cover what was tested, what was found, the key figure, and what it means.

---

## 11. Working style

- Start by writing a task list covering all phases, and update it as you go.
- Commit code to git after each working milestone, with clear messages.
- At each gate, write a short gate note in `DECISIONS.md` covering evidence, decision and next step. Then continue without waiting for me, unless Section 12 applies.
- Prefer simple, readable code over clever code. Comment the non-obvious parts, especially update clipping, CG, line search and the drift diagnostic.
- When an experiment surprises you, look at per-seed curves and diagnostics before drawing conclusions. Write the explanation into the report.

## 12. When to stop and ask me

- The GPU or environment setup fails in a way you cannot fix; for example, MuJoCo or DMC will not install.
- Gate G0 still fails after reasonable debugging (baselines do not reproduce).
- Gate G2 still fails after working through the full debug list. Summarise what you tried and propose options. One of those options is to proceed and report a negative result.
- The projected compute exceeds the available time even after the cuts in Section 8.
- You find evidence that a premise of this prompt is wrong; for example, clipping alone is stable in E1, or the P2 identity fails in unit test 3. Report it before building on it.

Begin with Section 3 (environment check) and Section 5 (unit tests). Then proceed through Phases 0–6 in order, run Phase 7 to push the direction forward, and finish with the report.