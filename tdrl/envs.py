"""make_env(name, seed): Gymnasium MuJoCo (v5) and DeepMind Control (via shimmy).
Actions are rescaled to [-1, 1], observations are flat float32.
Names: 'HalfCheetah-v5', 'Pendulum-v1', ... or 'dmc:cheetah-run'."""
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RescaleAction, FlattenObservation, TransformObservation


def make_env(name, seed=0):
    if name.startswith("dmc:"):
        import shimmy  # noqa: F401  (registers dm_control/* envs)
        gym.register_envs(shimmy)
        env = FlattenObservation(gym.make(f"dm_control/{name[4:]}-v0"))
    else:
        env = gym.make(name)
    env = RescaleAction(env, -1.0, 1.0)
    env = TransformObservation(env, lambda o: o.astype(np.float32),
                               gym.spaces.Box(-np.inf, np.inf, env.observation_space.shape, np.float32))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def get_state(env):
    """Full simulator state (for Monte-Carlo rollouts from an exact state)."""
    u = env.unwrapped
    if hasattr(u, "_env"):  # dm_control via shimmy
        return u._env.physics.get_state().copy()
    if hasattr(u, "data") and hasattr(u, "set_state"):  # gymnasium MuJoCo
        return (u.data.qpos.copy(), u.data.qvel.copy())
    if hasattr(u, "state"):  # classic control (Pendulum)
        return np.array(u.state, copy=True)
    return None


def set_state(env, state):
    u = env.unwrapped
    if hasattr(u, "_env"):
        with u._env.physics.reset_context():
            u._env.physics.set_state(state)
    elif hasattr(u, "set_state"):
        u.set_state(*state)
    else:
        u.state = state.copy()
