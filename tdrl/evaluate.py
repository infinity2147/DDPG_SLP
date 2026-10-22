"""Evaluation and the overestimation (Q vs Monte-Carlo) diagnostic."""
import numpy as np
import torch
from .envs import get_state, set_state


def safe(a):
    return np.nan_to_num(a, nan=0.0).clip(-1, 1)  # a diverged (NaN) policy acts as zero action


def evaluate(env, policy, episodes=10, gamma=0.99):
    """Deterministic episodes. Returns undiscounted returns, plus (start obs, discounted return) pairs."""
    rets, starts = [], []
    for _ in range(episodes):
        o, _ = env.reset()
        s0, R, G, k, done = o, 0.0, 0.0, 0, False
        while not done:
            o, r, term, trunc, _ = env.step(safe(policy(o)))
            R += r
            G += gamma ** k * r
            k += 1
            done = term or trunc
        rets.append(R)
        starts.append((s0, G))
    return np.array(rets), starts


def overestimation(env, agent, n_states=20, gamma=0.99, horizon=700, rng=None):
    """Q(s, mu(s)) - MC return of the deterministic policy from exactly s, for n_states states taken
    from fresh evaluation episodes. Uses simulator state restore (gymnasium MuJoCo, Pendulum); for dm_control
    (whose task state is not fully in physics.get_state) it falls back to episode start states.
    horizon=700 truncates the infinite discounted sum with relative error gamma^700 < 1e-3."""
    rng = rng or np.random.default_rng(0)
    restore = not hasattr(env.unwrapped, "_env")
    policy = agent.act
    if restore:
        pool = []  # (obs, sim state) along fresh deterministic episodes
        while len(pool) < 5 * n_states:
            o, _ = env.reset()
            done = False
            while not done:
                pool.append((o, get_state(env)))
                o, _, term, trunc, _ = env.step(safe(policy(o)))
                done = term or trunc
        picks = [pool[i] for i in rng.choice(len(pool), n_states, replace=False)]
        obs, mc = [], []
        for o, st in picks:
            env.reset()
            set_state(env, st)
            G, k, ob = 0.0, 0, o
            for k in range(horizon):
                ob, r, term, _, _ = env.step(safe(policy(ob)))  # time-limit truncation ignored on purpose
                G += gamma ** k * r
                if term:
                    break
            obs.append(o)
            mc.append(G)
    else:
        _, starts = evaluate(env, policy, n_states, gamma)
        obs, mc = zip(*starts)
    s = torch.as_tensor(np.array(obs), device=agent.device)
    q = agent.q_value(s).squeeze(-1).cpu().numpy()
    err = q - np.array(mc)
    return dict(q_bias=float(err.mean()), q_mae=float(np.abs(err).mean()), q_est=float(q.mean()),
                mc_return=float(np.mean(mc)), restore=restore)
