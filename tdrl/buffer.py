"""Uniform replay buffer. `terminated` and `truncated` are stored separately: the TD target
bootstraps through time-limit truncation and stops only at true terminal states."""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=1_000_000, device="cpu"):
        self.s = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros((size, 1), np.float32)
        self.s2 = np.zeros((size, obs_dim), np.float32)
        self.term = np.zeros((size, 1), np.float32)
        self.trunc = np.zeros((size, 1), np.float32)
        self.size, self.ptr, self.full, self.device = size, 0, False, device

    def __len__(self):
        return self.size if self.full else self.ptr

    def add(self, s, a, r, s2, terminated, truncated):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i] = s, a, r, s2
        self.term[i], self.trunc[i] = terminated, truncated
        self.ptr = (i + 1) % self.size
        self.full |= self.ptr == 0

    def sample(self, n, rng):
        return self.get(rng.integers(0, len(self), n))

    def get(self, idx):
        """Batch dict of tensors. done = terminated only (truncation still bootstraps)."""
        t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return dict(s=t(self.s), a=t(self.a), r=t(self.r), s2=t(self.s2), done=t(self.term))

    def state_dict(self):
        n = len(self)
        return dict(ptr=self.ptr, full=self.full, **{k: getattr(self, k)[:n].copy()
                                                       for k in ("s", "a", "r", "s2", "term", "trunc")})

    def load_state_dict(self, d):
        n = len(d["s"])
        for k in ("s", "a", "r", "s2", "term", "trunc"):
            getattr(self, k)[:n] = d[k]
        self.ptr, self.full = d["ptr"], d["full"]
