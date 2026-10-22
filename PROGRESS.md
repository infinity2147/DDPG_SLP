# PROGRESS

## Machine
Slurm cluster (login node has no GPU/Python). Partitions: dgx (9 nodes × 8 GPU, 256 CPU), a40 (19 × 4 GPU, 64 CPU),
l40 (6 × 8 GPU). Per-user QoS: dgx ≤ 4 running jobs, a40 ≤ 3. On 2026-10-22 all GPUs were allocated by other users,
while DGX nodes had 100–200 idle CPUs each → runs use 1 CPU thread per run. /home: Lustre, no quota.

## Status (2026-10-22 13:35)
| Item | State |
| --- | --- |
| Env + all 13 tasks step | done |
| Unit tests (13) | pass |
| E1, E2 toys | done (`results/E1`, `results/E2`) |
| pilot (Pendulum 50k ×6 methods, Hopper 60k ×4 methods, 3 seeds) | running (job 360484) |
| E0 (60 runs, 1M) + G1R (20 runs, 300k) | running (job 360485, 80 CPU slots) |
| E3–E13, X1–X6 | queued after their gates |

## Throughput / projection
CPU cost per update (ms): DDPG 14, TD3 12, SAC 24, ours 68, CrossQ 518. E0 needs ≈ 3–7 h per run; ours ≈ 19 h per
1M-step run. Full E8 on CPU is infeasible in days without GPUs → see cuts in DECISIONS.md once measured on E0.

## Commands
    python scripts/experiments.py <EXP...>                     # write queue/<EXP>.jsonl
    python scripts/launch.py submit --queues <EXP...> --partition dgx --gpus 0 --cpu-slots 80 --cpus 82 --mem 240G
    python scripts/launch.py status --queues <EXP>
