"""All agents, switchable by config (see configs/methods.yaml for the presets).

DetAgent : DDPG, TD3, DDPG without targets, LN+l2 critic, and ours (Components A-D, X1, X2).
SACAgent : SAC, CrossQ, and target-free SAC with a KL trust region (X4).

Without target networks, `critic_t` / `actor_t` are the live networks themselves, so the TD target
is built from the current networks; it is always computed under no_grad (semi-gradient TD).
"""
import copy
import math
import torch
from .networks import Actor, GaussianActor, Critic, linear_layers
from .stabilisers import CriticStabiliser, layer_norm
from .trust_region import TrustRegionActor

DEFAULTS = dict(
    gamma=0.99, actor_lr=3e-4, critic_lr=3e-4, hidden=[256, 256], batch_size=256,
    use_target=True, tau=0.005, n_critics=1, policy_delay=1, target_noise=0.0, noise_clip=0.5,
    critic_hidden=None, critic_norm=None, critic_act="relu", critic_opt="adam",
    clip=None, stab="none", R_mult=10.0, wd=0.0,                 # Component A
    actor_update="adam", eps=1e-3, damp=1e-2, cg_iters=10, K=10, cset="both",  # Components B, C
    drift_budget=None,                                           # X1: adaptive eps
    critic_eps=None,                                             # X2: critic trust region
    alpha_lr=None, crossq=False, kl_eps=None, adam_beta1=0.9,    # SAC / CrossQ / X4
)


@torch.no_grad()
def polyak(src, dst, tau):
    torch._foreach_lerp_(list(dst.parameters()), list(src.parameters()), tau)


def norm(ts):
    return torch.sqrt(sum(t.pow(2).sum() for t in ts)).item()


def n_params(*nets):
    return sum(p.numel() for n in nets for p in n.parameters())


class Base:
    """Shared drift diagnostic and helpers. Subclasses implement target(), _critic_step(), _actor_step()."""

    def update(self, b, probe=None, log=False):
        """One full update (critic then actor). If `probe` is given, measure target drift:
        y on the probe set before the update, after the critic step only, and after the full update."""
        self.it += 1
        st = {}
        if probe is not None:
            with torch.no_grad():
                y0, mu0 = self.target(probe, return_action=True)
        self._critic_step(b, st, log)
        if probe is not None:
            with torch.no_grad():
                y1 = self.target(probe)
            La0 = self.action_grad_norm(probe, mu0)  # ||grad_a Q_{k+1}(s', a)|| at mu_k(s')
        if self.it % self.c["policy_delay"] == 0:
            self._actor_step(b, st, log)
        if probe is not None:
            with torch.no_grad():
                y2, mu2 = self.target(probe, return_action=True)
            La2 = self.action_grad_norm(probe, mu2)
            disc = self.c["gamma"] * (1 - probe["done"]).squeeze(-1)
            dy, dyc = (y2 - y0).abs().squeeze(-1), (y1 - y0).abs().squeeze(-1)
            dmu = (mu2 - mu0).norm(dim=-1)
            st.update(drift_mean=dy.mean().item(), drift_p95=dy.quantile(0.95).item(),
                      drift_critic=dyc.mean().item(), drift_actor=(y2 - y1).abs().mean().item(),
                      dmu_probe=dmu.pow(2).mean().item(), La_mean=La0.mean().item(),
                      # P3 bound |dy| <= |dy_critic| + gamma * L_a * ||dmu(s')||, with L_a estimated locally
                      # (gradient norm at mu_k) and conservatively (max gradient norm over the probe set)
                      bound_local=(dyc + disc * La0 * dmu).mean().item(),
                      bound_max=(dyc + disc * torch.maximum(La0.max(), La2.max()) * dmu).mean().item())
        return st

    def action_grad_norm(self, p, a):
        a = a.detach().requires_grad_(True)
        self.critic_t.eval()  # BatchNorm critics: do not touch running statistics
        q = self.critic_t(p["s2"], a).min(0).values
        self.critic_t.train()
        return torch.autograd.grad(q.sum(), a)[0].norm(dim=-1).detach()

    def log_norms(self, st):
        for h, q in enumerate(self.critic.qs):
            ls = linear_layers(q)
            st[f"wnorm_out{h}"] = layer_norm(ls[-1]).item()
            st[f"wnorm_total{h}"] = norm(q.parameters())
            if self.c["stab"] == "proj_all":
                for i, l in enumerate(ls):
                    st[f"wnorm_l{i}_{h}"] = layer_norm(l).item()
        st["actor_wnorm"] = norm(self.actor.parameters())

    def n_params(self):
        nets = [self.actor, self.critic]
        if self.c["use_target"]:
            nets += [self.critic_t] + ([self.actor_t] if self.actor_t is not self.actor else [])
        return n_params(*nets)


class DetAgent(Base):
    def __init__(self, obs_dim, act_dim, cfg, device="cpu"):
        c = self.c = {**DEFAULTS, **cfg}
        self.it, self.device = 0, device
        self.actor = Actor(obs_dim, act_dim, c["hidden"]).to(device)
        self.critic = Critic(obs_dim, act_dim, c["n_critics"], c["critic_hidden"] or c["hidden"],
                             c["critic_act"], c["critic_norm"]).to(device)
        if c["use_target"]:
            self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        else:  # target-free: the TD target uses the live networks
            self.actor_t, self.critic_t = self.actor, self.critic
        sgd = c["critic_opt"] == "sgd"
        self.c_opt = (torch.optim.SGD(self.critic.parameters(), c["critic_lr"], momentum=0.9) if sgd
                      else torch.optim.Adam(self.critic.parameters(), c["critic_lr"]))
        self.stabiliser = CriticStabiliser(list(self.critic.qs), self.c_opt, c["critic_lr"], sgd,
                                           c["clip"], c["stab"], c["R_mult"], c["wd"])
        if c["actor_update"] == "adam":
            self.a_opt = torch.optim.Adam(self.actor.parameters(), c["actor_lr"])
        else:
            self.tr = TrustRegionActor(self.actor, c["actor_update"], c["eps"], c["damp"], c["cg_iters"],
                                       c["K"], c["actor_lr"], c["cset"])

    @torch.no_grad()
    def act(self, s):
        return self.actor(torch.as_tensor(s, device=self.device).unsqueeze(0))[0].cpu().numpy()

    def q_value(self, s):
        """Q1(s, mu(s)) used by the overestimation diagnostic."""
        with torch.no_grad():
            return self.critic(s, self.actor(s))[0]

    def target(self, b, smooth=False, return_action=False):
        """y = r + gamma (1 - done) min_h Q_t(s', mu_t(s')); call under no_grad (stop-gradient)."""
        a2 = mu = self.actor_t(b["s2"])
        if smooth and self.c["target_noise"] > 0:  # TD3 target policy smoothing
            n = (torch.randn_like(a2) * self.c["target_noise"]).clamp(-self.c["noise_clip"], self.c["noise_clip"])
            a2 = (a2 + n).clamp(-1, 1)
        y = b["r"] + self.c["gamma"] * (1 - b["done"]) * self.critic_t(b["s2"], a2).min(0).values
        return (y, mu) if return_action else y

    def _critic_step(self, b, st, log):
        c = self.c
        with torch.no_grad():
            y = self.target(b, smooth=True)
            if c["critic_eps"] is not None or c["drift_budget"] is not None:
                self._y_batch, a2 = y, self.actor(b["s2"])
                q_old = self.critic(b["s2"], a2)
        q = self.critic(b["s"], b["a"])
        loss = (q - y).pow(2).mean(dim=(1, 2)).sum()  # MSE per head, summed over heads
        self.c_opt.zero_grad(set_to_none=True)
        loss.backward()
        if log or c["critic_eps"] is not None:
            old = [p.detach().clone() for p in self.critic.parameters()]
        if log:
            st.update(critic_loss=loss.item(), q_mean=q[0].mean().item(),
                      critic_grad_norm=norm(p.grad for p in self.critic.parameters()))
        st["clip_frac"] = self.stabiliser.step()
        if c["critic_eps"] is not None:  # X2: backtrack the critic step until mean (dQ(s',mu(s')))^2 <= eps_Q
            new = [p.detach().clone() for p in self.critic.parameters()]
            st["critic_bt_m"] = -1
            with torch.no_grad():
                for m in range(11):
                    for p, o, n in zip(self.critic.parameters(), old, new):
                        p.copy_(o + 0.5 ** m * (n - o))
                    if (self.critic(b["s2"], a2) - q_old).pow(2).mean() <= c["critic_eps"]:
                        st["critic_bt_m"] = m
                        break
                else:
                    for p, o in zip(self.critic.parameters(), old):
                        p.copy_(o)
        if log:
            st["critic_update_norm"] = norm(p.detach() - o for p, o in zip(self.critic.parameters(), old))
            self.log_norms(st)
        if c["use_target"]:
            polyak(self.critic, self.critic_t, c["tau"])

    def _actor_step(self, b, st, log):
        c = self.c
        if log:
            old = [p.detach().clone() for p in self.actor.parameters()]
        if c["actor_update"] == "adam":
            loss = -self.critic(b["s"], self.actor(b["s"]))[0].mean()
            self.a_opt.zero_grad(set_to_none=True)
            loss.backward()
            if log:
                st["actor_grad_norm"] = norm(p.grad for p in self.actor.parameters())
            self.a_opt.step()
        else:
            st.update(self.tr.step(b["s"], b["s2"], lambda s, a: self.critic(s, a)[0]))
        if log:
            st["actor_update_norm"] = norm(p.detach() - o for p, o in zip(self.actor.parameters(), old))
        if c["drift_budget"] is not None:  # X1: steer eps so that the batch target drift tracks the budget
            with torch.no_grad():
                d = (self.target(b) - self._y_batch).abs().mean().item()
            self.tr.eps = min(max(self.tr.eps * (1.05 if d < c["drift_budget"] else 1 / 1.05), 1e-6), 1e-1)
            st.update(eps=self.tr.eps, batch_drift=d)
        if c["use_target"]:
            polyak(self.actor, self.actor_t, c["tau"])

    def compare_directions(self, b):
        return self.tr.compare_directions(b["s"], b["s2"], lambda s, a: self.critic(s, a)[0])

    def state_dict(self):
        d = {k: getattr(self, k).state_dict() for k in ("actor", "critic", "actor_t", "critic_t", "c_opt")}
        d["a_opt"] = (self.a_opt if hasattr(self, "a_opt") else self.tr.reg_opt).state_dict()
        d["it"], d["radii"] = self.it, self.stabiliser.radii
        d["eps"] = self.tr.eps if hasattr(self, "tr") else None
        return d

    def load_state_dict(self, d):
        for k in ("actor", "critic", "actor_t", "critic_t", "c_opt"):
            getattr(self, k).load_state_dict(d[k])
        (self.a_opt if hasattr(self, "a_opt") else self.tr.reg_opt).load_state_dict(d["a_opt"])
        self.it, self.stabiliser.radii = d["it"], d["radii"]
        if d["eps"] is not None:
            self.tr.eps = d["eps"]


class SACAgent(Base):
    """SAC (twin critics, auto-entropy). crossq=True: BatchNorm critic, no targets, joint s/s' forward pass.
    kl_eps (X4): target-free SAC whose actor step is backtracked until KL(old || new) <= kl_eps on s and s'."""

    def __init__(self, obs_dim, act_dim, cfg, device="cpu"):
        c = self.c = {**DEFAULTS, "n_critics": 2, **cfg}
        self.it, self.device = 0, device
        self.actor = GaussianActor(obs_dim, act_dim, c["hidden"]).to(device)
        bn = c["crossq"]
        self.critic = Critic(obs_dim, act_dim, 2, c["critic_hidden"] or c["hidden"], "relu",
                             "bn" if bn else None, input_norm=bn).to(device)
        self.critic_t = copy.deepcopy(self.critic) if c["use_target"] else self.critic
        self.actor_t = self.actor
        betas = (c["adam_beta1"], 0.999)
        self.c_opt = torch.optim.Adam(self.critic.parameters(), c["critic_lr"], betas=betas)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), c["actor_lr"], betas=betas)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], c["alpha_lr"] or c["actor_lr"], betas=betas)
        self.target_ent = -act_dim
        self.stabiliser = CriticStabiliser(list(self.critic.qs), self.c_opt, c["critic_lr"], False)  # no-op, for radii

    @torch.no_grad()
    def act(self, s, deterministic=True):
        s = torch.as_tensor(s, device=self.device).unsqueeze(0)
        return (self.actor(s) if deterministic else self.actor.sample(s)[0])[0].cpu().numpy()

    def q_value(self, s):
        self.critic.eval()
        with torch.no_grad():
            q = self.critic(s, self.actor(s))[0]
        self.critic.train()
        return q

    def target(self, b, return_action=False):
        """Soft TD target with a fixed noise draw per probe (so drift is not polluted by sampling noise).
        Called under no_grad. BatchNorm critics are evaluated in eval mode here."""
        g = torch.Generator(device=self.device).manual_seed(0)
        mu, std = self.actor.dist(b["s2"])
        u = mu + std * torch.randn(mu.shape, generator=g, device=self.device)
        a2 = torch.tanh(u)
        logp = (torch.distributions.Normal(mu, std).log_prob(u) - torch.log(1 - a2.pow(2) + 1e-6)).sum(-1, keepdim=True)
        self.critic_t.eval()
        q2 = self.critic_t(b["s2"], a2).min(0).values
        self.critic_t.train()
        y = b["r"] + self.c["gamma"] * (1 - b["done"]) * (q2 - self.log_alpha.exp() * logp)
        return (y, a2) if return_action else y

    def _critic_step(self, b, st, log):
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self.actor.sample(b["s2"])
        if self.c["crossq"]:  # one joint forward pass so BatchNorm sees both s and s' statistics
            q_all = self.critic(torch.cat([b["s"], b["s2"]]), torch.cat([b["a"], a2]))
            q, q2 = q_all.chunk(2, dim=1)
            q2 = q2.detach()
        else:
            q = self.critic(b["s"], b["a"])
            with torch.no_grad():
                q2 = self.critic_t(b["s2"], a2)
        y = (b["r"] + self.c["gamma"] * (1 - b["done"]) * (q2.min(0).values - alpha * logp2)).detach()
        loss = (q - y).pow(2).mean(dim=(1, 2)).sum()
        self.c_opt.zero_grad(set_to_none=True)
        loss.backward()
        if log:
            old = [p.detach().clone() for p in self.critic.parameters()]
            st.update(critic_loss=loss.item(), q_mean=q[0].mean().item(), alpha=alpha.item(),
                      critic_grad_norm=norm(p.grad for p in self.critic.parameters()))
        self.c_opt.step()
        if log:
            st["critic_update_norm"] = norm(p.detach() - o for p, o in zip(self.critic.parameters(), old))
            self.log_norms(st)
        if self.c["use_target"]:
            polyak(self.critic, self.critic_t, self.c["tau"])

    def _actor_step(self, b, st, log):
        s = b["s"] if self.c["kl_eps"] is None else torch.cat([b["s"], b["s2"]])
        if self.c["kl_eps"] is not None:
            with torch.no_grad():
                mu0, std0 = self.actor.dist(s)
            old = [p.detach().clone() for p in self.actor.parameters()]
        self.critic.eval()  # CrossQ: actor loss uses running BN statistics
        a, logp = self.actor.sample(b["s"])
        loss = (self.log_alpha.exp().detach() * logp - self.critic(b["s"], a).min(0).values).mean()
        self.a_opt.zero_grad(set_to_none=True)
        loss.backward()
        if log:
            st["actor_grad_norm"] = norm(p.grad for p in self.actor.parameters())
        self.a_opt.step()
        self.critic.train()
        if self.c["kl_eps"] is not None:  # X4: backtrack until mean KL(pi_old || pi_new) over s and s' <= kl_eps
            new = [p.detach().clone() for p in self.actor.parameters()]
            st["bt_m"], st["skipped"] = -1, 1.0
            with torch.no_grad():
                for m in range(11):
                    for p, o, n in zip(self.actor.parameters(), old, new):
                        p.copy_(o + 0.5 ** m * (n - o))
                    mu1, std1 = self.actor.dist(s)
                    kl = torch.distributions.kl_divergence(torch.distributions.Normal(mu0, std0),
                                                           torch.distributions.Normal(mu1, std1)).sum(-1).mean()
                    if kl <= self.c["kl_eps"]:
                        st.update(bt_m=m, skipped=0.0, kl=kl.item())
                        break
                else:
                    for p, o in zip(self.actor.parameters(), old):
                        p.copy_(o)
        # temperature
        alpha_loss = -(self.log_alpha.exp() * (logp.detach() + self.target_ent)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

    def state_dict(self):
        d = {k: getattr(self, k).state_dict() for k in ("actor", "critic", "critic_t", "c_opt", "a_opt", "alpha_opt")}
        d.update(it=self.it, log_alpha=self.log_alpha.detach().clone())
        return d

    def load_state_dict(self, d):
        for k in ("actor", "critic", "critic_t", "c_opt", "a_opt", "alpha_opt"):
            getattr(self, k).load_state_dict(d[k])
        self.it = d["it"]
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"])


def make_agent(obs_dim, act_dim, cfg, device="cpu"):
    cls = SACAgent if cfg.get("algo") in ("sac", "crossq") else DetAgent
    return cls(obs_dim, act_dim, {k: v for k, v in cfg.items() if k in DEFAULTS}, device)
