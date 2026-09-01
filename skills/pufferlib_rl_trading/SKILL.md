---
name: pufferlib-rl-trading
description: >-
  Reinforcement Learning execution & portfolio optimization using Gymnasium,
  Stable-Baselines3, and version-isolated PufferLib 3.0 & 4.0 vectorization pipelines.
---

# PufferLib & SB3 Reinforcement Learning Trading Skill

This skill defines the architectural standards for high-throughput RL trading environments, vectorized rollouts, and policy optimization.

---

## 1. Version-Separated RL Architecture (Rule C Compliance)

PufferLib 3.0 and PufferLib 4.0 introduce distinct architectural paradigms. To prevent API signature cross-contamination, workflows are isolated into two dedicated modules:

### PufferLib 3.0 Standard (`scripts/puffer_v3_standard.py`)
- Emulation wrapper: `pufferlib.emulation.GymnasiumPufferEnv`
- High-throughput flat-array state space flattening
- Seamless interoperability with Stable-Baselines3 (`PPO`, `RecurrentPPO`, `SAC`)
- Standard multiprocessing rollout buffers

### PufferLib 4.0 Standard (`scripts/puffer_v4_standard.py`)
- Native zero-copy PyTorch tensor memory sharing
- Structured observation spaces (Dict/Tuple) without forced flattening
- Hardware-accelerated vectorization via `pufferlib.vector.Multiprocessing` or `pufferlib.vector.GPU`
- Explicit guardrails preventing legacy v3 call invocation

---

## 2. Trading Gymnasium Environment (`scripts/trading_gym_env.py`)

- **State Space**: Window of stationary features $[L, D]$ + internal agent state (position, unrealized PnL, drawdown fraction, margin usage).
- **Action Space**:
  - Discrete: $\{0: \text{FLAT}, 1: \text{LONG}, 2: \text{SHORT}\}$
  - Continuous: Target position weight $[-1.0, +1.0]$
- **Reward Function**: Differential Sharpe Ratio with Drawdown Penalization:
  $$R_t = \frac{\Delta \text{Equity}_t}{\text{Equity}_{t-1}} - c_{\text{trans}} \cdot |\Delta \text{Pos}| - \lambda_{\text{dd}} \cdot \max(0, \text{Peak} - \text{Equity}_t)$$

---

## 3. Standard Usage Workflow

```python
from skills.pufferlib_rl_trading.scripts.trading_gym_env import MT5TradingGymEnv
from skills.pufferlib_rl_trading.scripts.puffer_v3_standard import train_puffer_v3_sb3
from skills.pufferlib_rl_trading.scripts.puffer_v4_standard import PufferV4VectorizedRunner

# Option A: Train SB3 PPO with PufferLib 3.0 emulation
model = train_puffer_v3_sb3(env_fn=make_env, total_timesteps=100000)

# Option B: Run zero-copy multi-env batch with PufferLib 4.0
runner = PufferV4VectorizedRunner(env_creator=make_env, num_envs=16)
runner.run_benchmark(steps=10000)
```

---

## 4. Module Directory Structure

```text
skills/pufferlib_rl_trading/
├── SKILL.md
├── scripts/
│   ├── __init__.py
│   ├── trading_gym_env.py     # OpenAI Gymnasium MT5 trading simulation
│   ├── puffer_v3_standard.py  # PufferLib 3.0 Emulation + SB3 PPO
│   └── puffer_v4_standard.py  # PufferLib 4.0 Zero-Copy Vectorization
└── tests/
    ├── __init__.py
    └── test_rl_trading.py     # Pytest test suite for Gym, SB3, & vectorizers
```
