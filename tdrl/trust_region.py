"""Component B: trust-region actor update.

    maximise  mean_i Q(s_i, mu_theta(s_i))   s.t.   mean_{s in C} ||mu_theta(s) - mu_k(s)||^2 <= eps

B-exact: natural-gradient direction from CG on (F + damp I) x = g, with Fisher-vector products
         F v = (1/|C|) J^T (J v) computed by jvp then vjp (F is never materialised).
B-cheap: regress mu_theta onto a+ = mu_k + eta * grad_a Q, eta = sqrt(eps / mean||grad_a Q||^2).
Both finish with a backtracking line search on the TRUE (non-linearised) constraint.
"""
import math
import torch
from torch.func import functional_call, jvp, vjp


def cg(Av, b, iters, tol=1e-10):
    """Conjugate gradient for A x = b with A symmetric positive definite (given as a mat-vec).
    Returns x and the relative residual ||b - A x|| / ||b||."""
    x = torch.zeros_like(b)
    r, p = b.clone(), b.clone()
    rr = r @ r
    for _ in range(iters):
        if rr < tol:
            break
        Ap = Av(p)
        alpha = rr / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rr_new = r @ r
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x, (torch.sqrt(rr) / b.norm().clamp_min(1e-12)).item()


def fisher_vector_product(actor, params, states, damp=0.0):
    """Returns v -> (1/|C|) J^T J v + damp v for the actor outputs on `states`, v a flat vector."""
    names, shapes = list(params), [p.shape for p in params.values()]
    sizes = [p.numel() for p in params.values()]
    f = lambda p: functional_call(actor, p, (states,))
    _, vjp_fn = vjp(f, params)  # one forward pass, reused for every CG iteration

    def unflat(v):
        return {n: c.view(s) for n, c, s in zip(names, v.split(sizes), shapes)}

    def Fv(v):
        Jv = jvp(f, (params,), (unflat(v),))[1]  # J v (forward mode)
        JtJv = vjp_fn(Jv)[0]  # J^T (J v) (reverse mode)
        return torch.cat([JtJv[n].reshape(-1) for n in names]) / states.shape[0] + damp * v

    return Fv


class TrustRegionActor:
    def __init__(self, actor, mode="cheap", eps=1e-3, damp=1e-2, cg_iters=10, K=10, lr=3e-4,
                 cset="both", max_backtracks=10):
        self.actor, self.mode, self.eps, self.damp = actor, mode, eps, damp
        self.cg_iters, self.K, self.cset, self.max_bt = cg_iters, K, cset, max_backtracks
        self.reg_opt = torch.optim.Adam(actor.parameters(), lr=lr)  # B-cheap regression optimiser

    def params(self):
        return {n: p.detach().clone() for n, p in self.actor.named_parameters()}

    @torch.no_grad()
    def set_params(self, base, direction=None, scale=1.0):
        for n, p in self.actor.named_parameters():
            p.copy_(base[n] if direction is None else base[n] + scale * direction[n])

    def sets(self, s, s2):
        """Objective and constraint state sets. If C contains s', s' also joins the objective."""
        obj = s if self.cset == "s" else torch.cat([s, s2])
        con = {"s": s, "sprime": s2, "both": torch.cat([s, s2])}[self.cset]
        return obj, con

    def propose(self, mode, obj, con, qfn, base):
        """Full (unscaled) step direction from theta_k = base. Leaves the actor at base."""
        info = {}
        if mode == "cheap":
            a = self.actor(obj).detach().requires_grad_(True)
            q = torch.autograd.grad(qfn(obj, a).sum(), a)[0]  # per-state grad_a Q
            eta = math.sqrt(self.eps / q.pow(2).sum(-1).mean().clamp_min(1e-12).item())
            target = a.detach() + eta * q  # not clipped: tanh output simply cannot reach |a+| > 1
            info["aplus_out_frac"] = (target.abs() > 1).float().mean().item()
            for _ in range(self.K):
                loss = (self.actor(obj) - target).pow(2).sum(-1).mean()
                self.reg_opt.zero_grad()
                loss.backward()
                self.reg_opt.step()
            direction = {n: p.detach() - base[n] for n, p in self.actor.named_parameters()}
            self.set_params(base)
            info["lin_con0"] = None  # filled by caller (measured at the full step)
        else:
            obj_q = qfn(obj, self.actor(obj)).mean()
            g = torch.autograd.grad(obj_q, list(self.actor.parameters()))
            g = torch.cat([t.reshape(-1) for t in g])
            Fv = fisher_vector_product(self.actor, base, con, self.damp)
            x, info["cg_residual"] = cg(Fv, g, self.cg_iters)
            xFx = (x @ (Fv(x) - self.damp * x)).clamp_min(1e-12)  # undamped quadratic form
            beta = math.sqrt(self.eps / xFx.item())
            info["lin_con0"] = self.eps  # beta^2 x^T F x = eps by construction
            sizes = [p.numel() for p in base.values()]
            direction = {n: c.view(base[n].shape) for n, c in zip(base, (beta * x).split(sizes))}
        return direction, info

    def step(self, s, s2, qfn):
        """One trust-region actor update on the (already updated) critic qfn(s, a) -> (B, 1)."""
        obj, con = self.sets(s, s2)
        base = self.params()
        with torch.no_grad():
            mu_con0, mu_s0, mu_s20 = self.actor(con), self.actor(s), self.actor(s2)
            q0 = qfn(obj, self.actor(obj)).mean()
        direction, info = self.propose(self.mode, obj, con, qfn, base)
        info.update(bt_m=-1, skipped=1.0, q_gain=0.0)
        with torch.no_grad():
            # Backtracking: accept the first m where Q improves AND the true constraint holds.
            for m in range(self.max_bt + 1):
                self.set_params(base, direction, 0.5 ** m)
                con_val = (self.actor(con) - mu_con0).pow(2).sum(-1).mean().item()
                if m == 0 and info["lin_con0"] is None:
                    info["lin_con0"] = con_val  # B-cheap: the full regression step's constraint
                q1 = qfn(obj, self.actor(obj)).mean()
                if q1 > q0 and con_val <= self.eps:
                    info.update(bt_m=m, skipped=0.0, q_gain=(q1 - q0).item(), con_actual=con_val,
                                con_lin=info["lin_con0"] * 0.25 ** m)  # quadratic model: scale^2
                    break
            else:
                self.set_params(base)
                info.update(con_actual=0.0, con_lin=0.0)
            info["dmu_s"] = (self.actor(s) - mu_s0).pow(2).sum(-1).mean().item()
            info["dmu_sprime"] = (self.actor(s2) - mu_s20).pow(2).sum(-1).mean().item()
        del info["lin_con0"]
        return info

    def compare_directions(self, s, s2, qfn):
        """E6 diagnostic: cosine similarity between the B-exact and B-cheap steps from the same theta_k.
        Restores the actor and the regression optimiser state afterwards."""
        obj, con = self.sets(s, s2)
        base, opt_state = self.params(), {k: v for k, v in self.reg_opt.state_dict().items()}
        opt_state = torch.utils._pytree.tree_map(lambda t: t.clone() if torch.is_tensor(t) else t, opt_state)
        d = {m: torch.cat([v.reshape(-1) for v in self.propose(m, obj, con, qfn, base)[0].values()])
             for m in ("exact", "cheap")}
        self.reg_opt.load_state_dict(opt_state)
        self.set_params(base)
        return torch.nn.functional.cosine_similarity(d["exact"], d["cheap"], dim=0).item()
