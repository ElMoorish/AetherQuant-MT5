"""
OpenAI Gymnasium Trading Environment for MT5 Data.
Simulates realistic trading dynamics including spreads, commissions, position tracking,
and drawdown-penalized rewards.
"""
from typing import Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class MT5TradingGymEnv(gym.Env):
    """
    Gymnasium environment simulating quantitative execution on MT5 financial series.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        price_col: str = "close",
        window_size: int = 30,
        initial_balance: float = 100000.0,
        transaction_cost_pct: float = 0.0002,  # 2 bps spread + commission
        drawdown_penalty_weight: float = 0.5,
        action_mode: str = "discrete",  # "discrete" or "continuous"
    ):
        super().__init__()
        self.df = df.copy().reset_index(drop=True)
        self.feature_cols = feature_cols
        self.price_col = price_col
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.drawdown_penalty_weight = drawdown_penalty_weight
        self.action_mode = action_mode

        self.num_features = len(feature_cols)
        self.total_steps = len(df) - 1

        # Action Space:
        # Discrete: 0 = Flat (0.0), 1 = Long (+1.0), 2 = Short (-1.0)
        # Continuous: Box[-1.0, 1.0]
        if self.action_mode == "discrete":
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space:
        # Flattened window [window_size * num_features] + 4 internal state features:
        # [current_position, unrealized_pnl_ratio, current_drawdown_pct, normalized_equity]
        obs_dim = (self.window_size * self.num_features) + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State Variables
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.position = 0.0  # -1.0 to +1.0
        self.entry_price = 0.0
        self.trades_count = 0

    def _get_observation(self) -> np.ndarray:
        """Constructs flat observation vector."""
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step
        feature_slice = self.df.iloc[start_idx:end_idx][self.feature_cols].values  # [W, D]
        flat_features = feature_slice.flatten().astype(np.float32)

        current_price = self.df.iloc[self.current_step][self.price_col]
        unrealized_pnl = 0.0
        if self.position != 0.0 and self.entry_price > 0:
            unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price

        dd_pct = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
        norm_equity = self.equity / self.initial_balance

        internal_state = np.array([
            self.position,
            unrealized_pnl,
            dd_pct,
            norm_equity
        ], dtype=np.float32)

        return np.concatenate([flat_features, internal_state])

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.trades_count = 0

        obs = self._get_observation()
        info = {"equity": self.equity, "balance": self.balance, "position": self.position}
        return obs, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Map action to target position
        if self.action_mode == "discrete":
            target_pos = 0.0 if action == 0 else (1.0 if action == 1 else -1.0)
        else:
            target_pos = float(np.clip(action[0] if isinstance(action, (list, np.ndarray)) else action, -1.0, 1.0))

        current_price = self.df.iloc[self.current_step][self.price_col]
        next_step = self.current_step + 1
        terminated = next_step >= self.total_steps
        truncated = False

        next_price = self.df.iloc[next_step][self.price_col] if not terminated else current_price
        price_return = (next_price - current_price) / current_price

        # Position change and transaction costs
        pos_delta = abs(target_pos - self.position)
        cost = pos_delta * self.equity * self.transaction_cost_pct
        if pos_delta > 0:
            self.trades_count += 1
            if target_pos != 0.0:
                self.entry_price = current_price

        # Update position
        self.position = target_pos

        # Compute step PnL and equity
        pnl = (self.position * price_return * self.equity) - cost
        self.equity += pnl
        self.equity = max(1.0, self.equity)  # prevent negative equity collapse

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Reward: return - drawdown penalty
        step_return = pnl / self.initial_balance
        drawdown = (self.peak_equity - self.equity) / self.peak_equity
        reward = float(step_return - (self.drawdown_penalty_weight * (drawdown ** 2)))

        self.current_step = next_step
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "equity": self.equity,
            "balance": self.balance,
            "position": self.position,
            "trades_count": self.trades_count,
            "drawdown": drawdown,
        }

        # Bankruptcy termination guard
        if self.equity <= self.initial_balance * 0.2:
            terminated = True

        return obs, reward, terminated, truncated, info
