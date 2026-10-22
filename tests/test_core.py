"""Unit tests 1, 2, 3, 4, 5, 7 of Section 5 (run: python -m pytest tests/test_core.py -q)."""
import json
import math
import os
import sys

import numpy as np
import pytest
import torch
from torch.func import functional_call, jacrev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tdrl.networks import Actor, Critic, linear_layers  # noqa: E402
from tdrl.stabilisers import CriticStabiliser, layer_norm, project  # noqa: E402
from tdrl.trust_region import TrustRegionActor, cg, fisher_vector_product  # noqa: E402
from tdrl.buffer import ReplayBuffer  # noqa: E402


@pytest.fixture(autouse=True)
def float64():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(old)


def params_of(actor):
    return {n: p.detach().clone() for n, p in actor.named_parameters()}


def flat(d):
    return torch.cat([v.reshape(-1) for v in d.values()])


def full_jacobian(actor, states):
    """Explicit J of shape (M*d, P): row (i, j) = d mu_j(s_i) / d theta."""
    p = params_of(actor)
    jac = jacrev(lambda q: functional_call(actor, q, (states,)))(p)  # dict of (M, d, *shape)
    M, d = states.shape[0], actor(states).shape[1]
    return torch.cat([jac[n].reshape(M * d, -1) for n in p], 1)


# ---------------------------------------------------------------- 1. Fisher-vector product
@pytest.mark.parametrize("damp", [0.0, 1e-2])
def test_fisher_vector_product(damp):
    actor = Actor(3, 2, hidden=(6, 5))
    states = torch.randn(7, 3)
    J = full_jacobian(actor, states)
    F = J.T @ J / states.shape[0] + damp * torch.eye(J.shape[1])
    Fv = fisher_vector_product(actor, params_of(actor), states, damp)
    for _ in range(5):
        v = torch.randn(J.shape[1])
        rel = ((Fv(v) - F @ v).norm() / (F @ v).norm()).item()
        assert rel < 1e-5, rel


# ---------------------------------------------------------------- 2. CG
def test_cg():
    n = 50
    A = torch.randn(n, n)
    A = A @ A.T + 0.1 * torch.eye(n)  # SPD
    b = torch.randn(n)
    x, res = cg(lambda v: A @ v, b, iters=500, tol=1e-24)
    assert res < 1e-8
    assert torch.allclose(x, torch.linalg.solve(A, b), rtol=1e-6, atol=1e-8)


# ---------------------------------------------------------------- 3. Result P2
def p2_step(J, q, M, eps, damp=0.0):
    """Natural trust-region step from the explicit metric. damp = 0 -> exact min-norm pinv(F) g."""
    F = J.T @ J / M
    g = J.T @ q.reshape(-1) / M
    if damp == 0:
        x = torch.linalg.pinv(F, hermitian=True) @ g
    else:
        x = torch.linalg.solve(F + damp * torch.eye(F.shape[0]), g)
    beta = math.sqrt(eps / (x @ F @ x).item())
    return beta * x


def p2_metrics(Jdtheta, q, eps):
    """cosine(J dtheta, q), scale <J dtheta, q>/(eta ||q||^2) and per-state relative error vs eta q."""
    eta = math.sqrt(eps / q.pow(2).sum(-1).mean().item())
    a, b = Jdtheta.reshape(-1), q.reshape(-1)
    return dict(cosine=torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
                scale=(a @ b / (eta * b @ b)).item(),
                rel_err=((a - eta * b).norm() / (eta * b).norm()).item(), eta=eta)


def test_p2_identity():
    obs, d, M, eps = 3, 2, 32, 1e-3
    actor = Actor(obs, d, hidden=(1024,))  # one hidden layer of 1024 units
    states, q = torch.randn(M, obs), torch.randn(M, d)
    J = full_jacobian(actor, states)
    P = J.shape[1]
    assert M * d < P and torch.linalg.matrix_rank(J).item() == M * d  # full row rank
    exact = p2_metrics(J @ p2_step(J, q, M, eps), q, eps)

    # The CG implementation used by TrustRegionActor (B-exact), small damping, on a linear critic
    # qfn(s, a) = <a, q(s)> so grad_a Q = q exactly.
    cg_res = {}
    for iters in (10, 50):
        tr = TrustRegionActor(actor, mode="exact", eps=eps, damp=1e-6, cg_iters=iters, cset="s")
        qfn = lambda s, a: (a * q).sum(-1, keepdim=True)
        direction, info = tr.propose("exact", states, states, qfn, params_of(actor))
        cg_res[iters] = dict(p2_metrics(J @ flat(direction), q, eps), cg_residual=info["cg_residual"])

    out = dict(width=1024, obs_dim=obs, act_dim=d, M=M, n_params=P, eps=eps, exact_pinv=exact,
               cg_damp_1e_6={f"iters_{k}": v for k, v in cg_res.items()})
    os.makedirs(f"{ROOT}/results/unit_tests", exist_ok=True)
    with open(f"{ROOT}/results/unit_tests/p2.json", "w") as f:
        json.dump(out, f, indent=2)

    assert exact["cosine"] > 0.999 and abs(exact["scale"] - 1) < 1e-3 and exact["rel_err"] < 1e-3
    assert cg_res[50]["cosine"] > 0.99


# ---------------------------------------------------------------- 4. Line search
def toy(mode, eps=1e-3, **kw):
    actor = Actor(3, 2, hidden=(32, 32))
    s, s2 = torch.randn(64, 3), torch.randn(64, 3)
    target = torch.full((1, 2), 0.5)
    qfn = lambda s_, a: -(a - target).pow(2).sum(-1, keepdim=True)  # quadratic critic, max at a* = 0.5
    tr = TrustRegionActor(actor, mode=mode, eps=eps, lr=1e-3, **kw)
    return actor, tr, s, s2, qfn


def true_con(actor, base, con):
    after = actor(con)
    before = functional_call(actor, base, (con,))
    return (after - before).pow(2).sum(-1).mean().item()


@pytest.mark.parametrize("mode", ["exact", "cheap"])
def test_line_search_accepts(mode):
    """Both modes take an accepted step that improves Q and satisfies the TRUE constraint."""
    actor, tr, s, s2, qfn = toy(mode, damp=1e-4, cg_iters=20)
    c = torch.cat([s, s2])
    base = params_of(actor)
    q0 = qfn(c, actor(c)).mean().item()
    info = tr.step(s, s2, qfn)
    with torch.no_grad():
        q1, con = qfn(c, actor(c)).mean().item(), true_con(actor, base, c)
    assert info["skipped"] == 0.0 and q1 > q0 and abs(q1 - q0 - info["q_gain"]) < 1e-12
    assert con <= tr.eps and abs(con - info["con_actual"]) < 1e-12
    # B-exact lands exactly on the LINEARISED boundary (beta^2 x^T F x = eps); the curved tanh actor
    # overshoots it by O(0.1-1%) in the true constraint, so m = 1 is also a legitimate outcome.
    assert info["bt_m"] in ((0, 1) if mode == "exact" else (0,)), info


def test_line_search_full_step_quadratic():
    """Linear actor + quadratic critic: true constraint == linearised one, so a step inside the trust
    region is accepted at m = 0 (full step)."""
    actor = torch.nn.Linear(3, 2)
    s, s2 = torch.randn(64, 3), torch.randn(64, 3)
    qfn = lambda s_, a: -(a - 0.5).pow(2).sum(-1, keepdim=True)
    tr = TrustRegionActor(actor, mode="exact", eps=1e-3, damp=1e-8, cg_iters=50)
    c, base = torch.cat([s, s2]), params_of(actor)
    d, _ = tr.propose("exact", c, c, qfn, base)
    tr.set_params(base, d)
    with torch.no_grad():
        assert abs(true_con(actor, base, c) / tr.eps - 1) < 1e-9  # exactly on the boundary
    tr.set_params(base)
    orig = tr.propose
    tr.propose = lambda *a, **k: ({n: 0.999 * v for n, v in orig(*a, **k)[0].items()}, {"lin_con0": None})
    info = tr.step(s, s2, qfn)
    assert info["bt_m"] == 0 and info["skipped"] == 0.0 and info["con_actual"] <= tr.eps


@pytest.mark.parametrize("mode", ["exact", "cheap"])
def test_line_search_rejects_violation(mode):
    actor, tr, s, s2, qfn = toy(mode, damp=1e-4, cg_iters=20)
    orig = tr.propose

    def huge(*a, **k):  # same direction, 1000x too long: violates the constraint at m = 0
        d, info = orig(*a, **k)
        return {n: 1000 * v for n, v in d.items()}, info

    tr.propose = huge
    base = params_of(actor)
    info = tr.step(s, s2, qfn)
    with torch.no_grad():
        con = true_con(actor, base, torch.cat([s, s2]))
    if info["skipped"]:
        assert con == 0.0  # actor restored
    else:
        assert info["bt_m"] > 0 and con <= tr.eps

    # A direction that only lowers Q is never accepted: the update is skipped and the actor restored.
    base = params_of(actor)
    tr.propose = lambda *a, **k: ({n: -1e-3 * v for n, v in orig(*a, **k)[0].items()}, {"lin_con0": 0.0})
    info = tr.step(s, s2, qfn)
    assert info["skipped"] == 1.0 and info["bt_m"] == -1
    assert all(torch.equal(p, base[n]) for n, p in actor.named_parameters())


# ---------------------------------------------------------------- 5. Stabilisers
def test_projection():
    layer = torch.nn.Linear(10, 1)
    with torch.no_grad():
        layer.weight.mul_(100)
    project(layer, 3.0)
    assert layer_norm(layer).item() <= 3.0 + 1e-9
    inside = torch.nn.Linear(10, 1)
    before = layer_norm(inside).item()
    project(inside, 1e3)  # already inside: untouched
    assert layer_norm(inside).item() == before


def random_critic_steps(sgd, stab, n=300, lr=1e-2, lam=1.0):
    critic = Critic(4, 2, n=2, hidden=(16, 16))
    opt = (torch.optim.SGD(critic.parameters(), lr=lr, momentum=0.9) if sgd
           else torch.optim.Adam(critic.parameters(), lr=lr))
    st = CriticStabiliser(list(critic.qs), opt, lr, sgd, clip=lam, stab=stab, R_mult=1.5, wd=0.0)
    for _ in range(n):
        x, a, y = torch.randn(32, 4), torch.randn(32, 2), 100 * torch.randn(1, 32, 1)
        outs = [linear_layers(q)[-1] for q in critic.qs]
        old = [torch.cat([o.weight.detach().reshape(-1), o.bias.detach()]).clone() for o in outs]
        opt.zero_grad()
        (critic(x, a) - y).pow(2).mean().backward()
        st.step()
        yield critic, st, outs, old


def test_adam_update_clipping():
    lr, lam = 1e-2, 1.0
    for critic, st, outs, old in random_critic_steps(sgd=False, stab="none", lr=lr, lam=lam):
        for o, w0 in zip(outs, old):
            delta = torch.cat([o.weight.detach().reshape(-1), o.bias.detach()]) - w0
            assert delta.norm().item() <= lr * lam + 1e-7


def test_sgd_gradient_clipping_and_projection():
    lam = 1.0
    for critic, st, outs, old in random_critic_steps(sgd=True, stab="proj", lam=lam):
        for o, R in zip(outs, [r[-1] for r in st.radii]):
            g = torch.cat([o.weight.grad.reshape(-1), o.bias.grad])
            assert g.norm().item() <= lam + 1e-9  # gradient clipped before the SGD step
            assert layer_norm(o).item() <= R + 1e-9  # projection keeps the output layer in its ball


# ---------------------------------------------------------------- 7. Buffer
def test_buffer_done_is_terminated_only():
    buf = ReplayBuffer(2, 1, size=10)
    buf.add([0, 0], [0], 1.0, [1, 1], terminated=False, truncated=False)  # ordinary
    buf.add([1, 1], [0], 1.0, [2, 2], terminated=False, truncated=True)  # time limit: bootstrap
    buf.add([2, 2], [0], 1.0, [3, 3], terminated=True, truncated=False)  # true terminal: no bootstrap
    b = buf.get(np.arange(3))
    assert b["done"].reshape(-1).tolist() == [0.0, 0.0, 1.0]
    # The TD target y = r + gamma (1 - done) Q' therefore bootstraps through truncation only.
    y = b["r"] + 0.99 * (1 - b["done"]) * 5.0
    assert y.reshape(-1).tolist() == pytest.approx([1 + 0.99 * 5, 1 + 0.99 * 5, 1.0])
