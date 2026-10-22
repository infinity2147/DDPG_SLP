"""Persistent job queue over a shared filesystem.

A job is one line of queue/<name>.jsonl: {"out": ..., "method": ..., "env": ..., "seed": ..., "set": {...}}.
Workers (one per Slurm allocation) claim jobs atomically by creating <out>/CLAIM with O_EXCL, run up to
--per-gpu jobs per GPU (or --cpu-slots CPU jobs), and skip jobs whose <out>/DONE exists. A claim whose Slurm job is
no longer running is stale and gets reclaimed; the run then resumes from its checkpoint.

  python scripts/launch.py worker --queues E0 E5 --per-gpu 8          # inside an allocation
  python scripts/launch.py submit --queues E0 --partition l40 --gpus 8  # sbatch a worker
  python scripts/launch.py status --queues E0
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
JOB_ID = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")


def load(queues):
    jobs = []
    for q in queues:
        jobs += [json.loads(l) for l in open(ROOT / f"queue/{q}.jsonl") if l.strip()]
    return jobs


def running_slurm_jobs():
    out = subprocess.run(["squeue", "-h", "-u", os.environ["USER"], "-o", "%i"], capture_output=True, text=True).stdout
    return set(out.split())


def try_claim(job):
    """Atomically claim a job; returns True on success. Stale claims (dead Slurm job) are removed first."""
    out = ROOT / job["out"]
    if (out / "DONE").exists() or (out / "FAILED").exists():
        return False
    out.mkdir(parents=True, exist_ok=True)
    claim = out / "CLAIM"
    if claim.exists():
        owner = claim.read_text().strip()
        if owner == JOB_ID or owner.startswith("local") or owner in running_slurm_jobs():
            return False
        claim.unlink(missing_ok=True)  # stale: its worker died; the run will resume from ckpt
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, JOB_ID.encode())
    os.close(fd)
    return True


def start(job, device_env):
    out = ROOT / job["out"]
    sets = [f"{k}={json.dumps(v) if isinstance(v, (list, dict)) else v}" for k, v in job.get("set", {}).items()]
    cmd = [PY, "-m", "tdrl.train", "--out", job["out"], "--method", job["method"], "--env", job["env"],
           "--seed", str(job["seed"]), "--set", *sets]
    log = open(out / "train.log", "a")
    return subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **device_env})


def worker(a):
    gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", "0") or 0) if a.gpus is None else a.gpus
    slots = [{"CUDA_VISIBLE_DEVICES": str(g)} for g in range(gpus) for _ in range(a.per_gpu)] or \
            [{"CUDA_VISIBLE_DEVICES": ""} for _ in range(a.cpu_slots)]
    running = {}  # slot index -> (Popen, job)
    while True:
        for i, (p, job) in list(running.items()):
            if p.poll() is not None:  # finished or crashed
                out = ROOT / job["out"]
                if p.returncode != 0:  # count failures; give up after 3 (the run is then marked FAILED)
                    n = int((out / "FAILS").read_text()) + 1 if (out / "FAILS").exists() else 1
                    (out / "FAILS").write_text(str(n))
                    if n >= 3:
                        (out / "FAILED").write_text(f"returncode {p.returncode}\n")
                (out / "CLAIM").unlink(missing_ok=True)
                del running[i]
        free = [i for i in range(len(slots)) if i not in running]
        if free:
            for job in load(a.queues):  # reread each round so newly queued jobs get picked up
                if not free:
                    break
                if try_claim(job):
                    i = free.pop(0)
                    job = {**job, "set": {**job.get("set", {}), "device": "cuda" if gpus else "cpu"}}
                    running[i] = (start(job, slots[i]), job)
                    print(time.strftime("%H:%M:%S"), "start", job["out"], flush=True)
        if not running:  # nothing left that this worker can claim (rest is done or owned by live workers)
            print("no claimable jobs left; exiting", flush=True)
            return
        time.sleep(20)


def status(a):
    jobs = load(a.queues)
    st = {"done": 0, "running": 0, "failed": 0, "diverged": 0, "todo": 0}
    for j in jobs:
        o = ROOT / j["out"]
        k = "done" if (o / "DONE").exists() else "failed" if (o / "FAILED").exists() else \
            "running" if (o / "CLAIM").exists() else "todo"
        st[k] += 1
        st["diverged"] += (o / "DIVERGED").exists()
    print(" ".join(a.queues), st)


def submit(a):
    res = f"--gres=gpu:{a.gpus}" if a.gpus else ""
    cmd = (f"sbatch -p {a.partition} --qos={a.partition} {res} -c {a.cpus} --mem={a.mem} -t {a.time} "
           f"-J tdrl -o {ROOT}/slurm_logs/worker_%j.out --wrap \"cd {ROOT} && {PY} scripts/launch.py worker "
           f"--queues {' '.join(a.queues)} --per-gpu {a.per_gpu} --cpu-slots {a.cpu_slots}\"")
    print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["worker", "status", "submit"])
    p.add_argument("--queues", nargs="+", required=True)
    p.add_argument("--per-gpu", type=int, default=8)
    p.add_argument("--cpu-slots", type=int, default=0)
    p.add_argument("--gpus", type=int, default=None)
    p.add_argument("--partition", default="l40")
    p.add_argument("--cpus", type=int, default=64)
    p.add_argument("--mem", default="256G")
    p.add_argument("--time", default="2-00:00:00")
    a = p.parse_args()
    {"worker": worker, "status": status, "submit": submit}[a.cmd](a)
