"""
Unit test suite for Gymnasium Trading Environment and PufferLib Integration.
"""
import pytest
import numpy as np
import pandas as pd

from skills.pufferlib_rl_trading.scripts.trading_gym_env import MT5TradingGymEnv
from skills.pufferlib_rl_trading.scripts.puffer_v3_standard import make_puffer_v3_env
from skills.pufferlib_rl_trading.scripts.puffer_v4_standard import PufferV4VectorizedRunner


@pytest.fixture
def synthetic_trading_df():
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    ret = np.random.normal(0.0001, 0.002, n)
    price = 100.0 * np.exp(np.cumsum(ret))

    return pd.DataFrame({
        "time": dates,
        "close": price,
        "log_return": ret,
        "volatility": np.abs(np.random.normal(0.01, 0.002, n)),
    })


def test_gym_env_reset_and_step(synthetic_trading_df):
    env = MT5TradingGymEnv(
        df=synthetic_trading_df,
        feature_cols=["log_return", "volatility"],
        price_col="close",
        window_size=10,
        action_mode="discrete"
    )

    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert info["equity"] == 100000.0

    # Step 1: BUY action (1)
    obs, reward, terminated, truncated, info = env.step(1)
    assert isinstance(reward, float)
    assert not terminated
    assert info["position"] == 1.0

    # Step 2: FLAT action (0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert info["position"] == 0.0


def test_puffer_v3_wrapper(synthetic_trading_df):
    def make_env():
        return MT5TradingGymEnv(
            df=synthetic_trading_df,
            feature_cols=["log_return", "volatility"],
            window_size=10
        )

    wrapped_env = make_puffer_v3_env(make_env)
    assert wrapped_env is not None


def test_puffer_v4_vectorized_runner(synthetic_trading_df):
    def make_env():
        return MT5TradingGymEnv(
            df=synthetic_trading_df,
            feature_cols=["log_return", "volatility"],
            window_size=10
        )

    runner = PufferV4VectorizedRunner(env_creator=make_env, num_envs=4)
    actions = np.array([1, 0, 2, 1])  # Actions for 4 envs
    step_out = runner.run_rollout_step(actions)

    assert "observations" in step_out
    assert "rewards" in step_out
    assert len(step_out["rewards"]) == 4
    runner.close()
