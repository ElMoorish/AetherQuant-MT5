"""
PufferLib 3.0 Standard Integration.
Enforces Rule C: Emulation wrapper, flat array state space, and Stable-Baselines3 interoperability.
"""
from typing import Callable, Any, Optional
import gymnasium as gym

try:
    import pufferlib
    import pufferlib.emulation
    PUFFER_AVAILABLE = True
except ImportError:
    pufferlib = None
    PUFFER_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    PPO = None
    DummyVecEnv = None
    SB3_AVAILABLE = False


def make_puffer_v3_env(env_fn: Callable[[], gym.Env]) -> gym.Env:
    """
    Wraps standard Gymnasium environment with PufferLib 3.0 GymnasiumPufferEnv.
    """
    raw_env = env_fn()
    if not PUFFER_AVAILABLE:
        return raw_env

    # PufferLib 3.0 standard emulation
    if hasattr(pufferlib, "emulation") and hasattr(pufferlib.emulation, "GymnasiumPufferEnv"):
        return pufferlib.emulation.GymnasiumPufferEnv(env_creator=env_fn)
    return raw_env


def train_puffer_v3_sb3(
    env_fn: Callable[[], gym.Env],
    total_timesteps: int = 50000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    verbose: int = 1
) -> Any:
    """
    Trains Stable-Baselines3 PPO agent using PufferLib 3.0 wrapped environment.
    """
    if not SB3_AVAILABLE:
        raise RuntimeError("stable-baselines3 is not installed.")

    env = make_puffer_v3_env(env_fn)
    vec_env = DummyVecEnv([lambda: env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=verbose
    )

    model.learn(total_timesteps=total_timesteps)
    return model
