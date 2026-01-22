"""
Trading environment compatible with Gymnasium.
Provides the interface for agents to interact with market data.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from loguru import logger


@dataclass
class EnvironmentConfig:
    """Configuration for the trading environment."""
    initial_balance: float = 100_000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    max_position_size: float = 0.25  # Max 25% in single position
    lookback_window: int = 60  # Number of historical bars to observe
    reward_scaling: float = 1.0
    use_log_returns: bool = True


@dataclass
class Position:
    """Represents a position in an asset."""
    symbol: str
    shares: float
    entry_price: float
    entry_time: pd.Timestamp
    current_price: float = 0.0

    @property
    def value(self) -> float:
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        return self.value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis


@dataclass
class PortfolioState:
    """Current state of the portfolio."""
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_portfolio_value: float = 0.0

    @property
    def position_value(self) -> float:
        return sum(pos.value for pos in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.cash + self.position_value

    @property
    def drawdown(self) -> float:
        if self.max_portfolio_value == 0:
            return 0.0
        return (self.max_portfolio_value - self.total_value) / self.max_portfolio_value

    def update_max_value(self):
        self.max_portfolio_value = max(self.max_portfolio_value, self.total_value)


class TradingEnvironment(gym.Env):
    """
    Gymnasium-compatible trading environment.

    Observation space:
        - Market features (OHLCV + technical indicators)
        - Portfolio state (cash ratio, position ratios, pnl)

    Action space:
        - Discrete: [0: Hold, 1: Buy, 2: Sell] or
        - Continuous: Position target [-1, 1] where -1 is full short, 1 is full long

    Reward:
        - Portfolio return (with risk-adjusted options)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        symbol: str = "ASSET",
        config: Optional[EnvironmentConfig] = None,
        discrete_actions: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize trading environment.

        Args:
            data: DataFrame with OHLCV and features
            feature_columns: List of columns to use as features
            symbol: Asset symbol being traded
            config: Environment configuration
            discrete_actions: Use discrete (3) or continuous action space
            render_mode: Rendering mode
        """
        super().__init__()

        self.config = config or EnvironmentConfig()
        self.symbol = symbol
        self.feature_columns = feature_columns
        self.discrete_actions = discrete_actions
        self.render_mode = render_mode
        self.logger = logger.bind(env="trading")

        # Store data
        self._validate_data(data)
        self.data = data.reset_index(drop=True)
        self.n_steps = len(data) - self.config.lookback_window

        # Feature dimension
        self.n_features = len(feature_columns)
        self.n_portfolio_features = 4  # cash_ratio, position_ratio, unrealized_pnl, drawdown

        # Define spaces
        self._setup_spaces()

        # Initialize state
        self.portfolio: Optional[PortfolioState] = None
        self.current_step = 0
        self.done = False
        self.history: list[dict] = []

    def _validate_data(self, data: pd.DataFrame):
        """Validate that data has required columns."""
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in data.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        missing_features = [f for f in self.feature_columns if f not in data.columns]
        if missing_features:
            raise ValueError(f"Data missing feature columns: {missing_features}")

    def _setup_spaces(self):
        """Define observation and action spaces."""
        # Observation: (lookback_window, n_features) + portfolio state
        obs_dim = self.config.lookback_window * self.n_features + self.n_portfolio_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action space
        if self.discrete_actions:
            # 0: Hold, 1: Buy, 2: Sell
            self.action_space = spaces.Discrete(3)
        else:
            # Continuous: target position [-1, 1]
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)

        # Reset portfolio
        self.portfolio = PortfolioState(
            cash=self.config.initial_balance,
            max_portfolio_value=self.config.initial_balance,
        )

        # Reset step counter
        self.current_step = 0
        self.done = False
        self.history = []

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take action and advance environment.

        Args:
            action: Trading action

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        # Get current price
        idx = self.current_step + self.config.lookback_window
        current_price = self.data.loc[idx, "close"]

        # Store pre-action portfolio value
        pre_value = self.portfolio.total_value

        # Update position prices
        if self.symbol in self.portfolio.positions:
            self.portfolio.positions[self.symbol].current_price = current_price

        # Execute action
        self._execute_action(action, current_price)

        # Advance to next step
        self.current_step += 1

        # Check if done
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        # Get new price and update positions
        if not terminated:
            next_idx = self.current_step + self.config.lookback_window
            next_price = self.data.loc[next_idx, "close"]
            if self.symbol in self.portfolio.positions:
                self.portfolio.positions[self.symbol].current_price = next_price

        # Calculate reward
        post_value = self.portfolio.total_value
        reward = self._calculate_reward(pre_value, post_value)

        # Update max portfolio value
        self.portfolio.update_max_value()

        # Get observation
        obs = self._get_observation()
        info = self._get_info()

        # Record history
        self.history.append({
            "step": self.current_step,
            "action": action if isinstance(action, int) else action[0],
            "price": current_price,
            "portfolio_value": post_value,
            "cash": self.portfolio.cash,
            "position_value": self.portfolio.position_value,
            "reward": reward,
        })

        self.done = terminated

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int | np.ndarray, price: float):
        """Execute trading action."""
        if self.discrete_actions:
            self._execute_discrete_action(action, price)
        else:
            self._execute_continuous_action(action[0], price)

    def _execute_discrete_action(self, action: int, price: float):
        """Execute discrete action (Hold/Buy/Sell)."""
        position = self.portfolio.positions.get(self.symbol)

        if action == 1:  # Buy
            if position is None:
                # Calculate position size
                max_value = self.portfolio.cash * self.config.max_position_size
                # Account for transaction costs
                effective_price = price * (1 + self.config.transaction_cost + self.config.slippage)
                shares = max_value / effective_price

                if shares > 0 and self.portfolio.cash >= shares * effective_price:
                    cost = shares * effective_price
                    self.portfolio.cash -= cost
                    self.portfolio.positions[self.symbol] = Position(
                        symbol=self.symbol,
                        shares=shares,
                        entry_price=price,
                        entry_time=self.data.loc[
                            self.current_step + self.config.lookback_window, "timestamp"
                        ] if "timestamp" in self.data.columns else pd.Timestamp.now(),
                        current_price=price,
                    )
                    self.portfolio.total_trades += 1

        elif action == 2:  # Sell
            if position is not None:
                # Sell entire position
                effective_price = price * (1 - self.config.transaction_cost - self.config.slippage)
                proceeds = position.shares * effective_price
                pnl = proceeds - position.cost_basis

                self.portfolio.cash += proceeds
                self.portfolio.total_pnl += pnl

                if pnl > 0:
                    self.portfolio.winning_trades += 1

                del self.portfolio.positions[self.symbol]
                self.portfolio.total_trades += 1

        # Action 0 (Hold) does nothing

    def _execute_continuous_action(self, target: float, price: float):
        """Execute continuous action (target position)."""
        # Target is in [-1, 1], map to position size
        target_position_value = self.portfolio.total_value * abs(target) * self.config.max_position_size

        current_position = self.portfolio.positions.get(self.symbol)
        current_value = current_position.value if current_position else 0

        # Calculate required trade
        trade_value = target_position_value - current_value if target >= 0 else -current_value

        if abs(trade_value) > 100:  # Minimum trade threshold
            if trade_value > 0:  # Buy
                effective_price = price * (1 + self.config.transaction_cost + self.config.slippage)
                shares_to_buy = min(trade_value / effective_price, self.portfolio.cash / effective_price)

                if shares_to_buy > 0:
                    cost = shares_to_buy * effective_price
                    self.portfolio.cash -= cost

                    if current_position:
                        # Add to position
                        total_cost = current_position.cost_basis + cost
                        total_shares = current_position.shares + shares_to_buy
                        current_position.shares = total_shares
                        current_position.entry_price = total_cost / total_shares
                    else:
                        self.portfolio.positions[self.symbol] = Position(
                            symbol=self.symbol,
                            shares=shares_to_buy,
                            entry_price=price,
                            entry_time=pd.Timestamp.now(),
                            current_price=price,
                        )
                    self.portfolio.total_trades += 1

            elif trade_value < 0 and current_position:  # Sell
                shares_to_sell = min(abs(trade_value) / price, current_position.shares)
                effective_price = price * (1 - self.config.transaction_cost - self.config.slippage)
                proceeds = shares_to_sell * effective_price

                pnl = proceeds - (shares_to_sell * current_position.entry_price)
                self.portfolio.cash += proceeds
                self.portfolio.total_pnl += pnl

                if pnl > 0:
                    self.portfolio.winning_trades += 1

                current_position.shares -= shares_to_sell
                if current_position.shares <= 0:
                    del self.portfolio.positions[self.symbol]

                self.portfolio.total_trades += 1

    def _calculate_reward(self, pre_value: float, post_value: float) -> float:
        """Calculate reward for the step."""
        if self.config.use_log_returns:
            if pre_value > 0 and post_value > 0:
                reward = np.log(post_value / pre_value)
            else:
                reward = 0.0
        else:
            reward = (post_value - pre_value) / pre_value if pre_value > 0 else 0.0

        return reward * self.config.reward_scaling

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        idx = self.current_step + self.config.lookback_window

        # Get market features
        start_idx = idx - self.config.lookback_window + 1
        market_data = self.data.loc[start_idx:idx, self.feature_columns].values
        market_features = market_data.flatten()

        # Get portfolio features
        portfolio_value = self.portfolio.total_value
        cash_ratio = self.portfolio.cash / portfolio_value if portfolio_value > 0 else 1.0
        position_ratio = self.portfolio.position_value / portfolio_value if portfolio_value > 0 else 0.0

        position = self.portfolio.positions.get(self.symbol)
        unrealized_pnl = position.unrealized_pnl_pct if position else 0.0

        portfolio_features = np.array([
            cash_ratio,
            position_ratio,
            unrealized_pnl,
            self.portfolio.drawdown,
        ], dtype=np.float32)

        # Combine
        obs = np.concatenate([market_features, portfolio_features]).astype(np.float32)

        # Handle NaN
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return obs

    def _get_info(self) -> dict:
        """Get info dict."""
        return {
            "portfolio_value": self.portfolio.total_value,
            "cash": self.portfolio.cash,
            "position_value": self.portfolio.position_value,
            "total_trades": self.portfolio.total_trades,
            "winning_trades": self.portfolio.winning_trades,
            "total_pnl": self.portfolio.total_pnl,
            "drawdown": self.portfolio.drawdown,
            "step": self.current_step,
        }

    def render(self):
        """Render environment state."""
        if self.render_mode == "human":
            info = self._get_info()
            print(f"\nStep {self.current_step}/{self.n_steps}")
            print(f"Portfolio: ${info['portfolio_value']:,.2f}")
            print(f"Cash: ${info['cash']:,.2f}")
            print(f"Positions: ${info['position_value']:,.2f}")
            print(f"Drawdown: {info['drawdown']:.2%}")
            print(f"Trades: {info['total_trades']} (Win: {info['winning_trades']})")

    def get_episode_stats(self) -> dict:
        """Get statistics for completed episode."""
        if not self.history:
            return {}

        values = [h["portfolio_value"] for h in self.history]
        returns = pd.Series(values).pct_change().dropna()

        return {
            "total_return": (values[-1] - values[0]) / values[0] if values[0] > 0 else 0,
            "sharpe_ratio": returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            "max_drawdown": self.portfolio.drawdown,
            "total_trades": self.portfolio.total_trades,
            "win_rate": self.portfolio.winning_trades / self.portfolio.total_trades if self.portfolio.total_trades > 0 else 0,
            "final_value": values[-1],
        }


class MultiAssetEnvironment(TradingEnvironment):
    """
    Environment for trading multiple assets simultaneously.
    Extends TradingEnvironment for portfolio-level decisions.
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        feature_columns: list[str],
        config: Optional[EnvironmentConfig] = None,
        **kwargs,
    ):
        """
        Initialize multi-asset environment.

        Args:
            data: Dictionary mapping symbol to DataFrame
            feature_columns: Features to use
            config: Environment configuration
        """
        self.symbols = list(data.keys())
        self.n_assets = len(self.symbols)

        # Align all dataframes to common index
        self.multi_data = self._align_data(data)

        # Use first symbol's data as base for parent class
        first_symbol = self.symbols[0]
        super().__init__(
            data=data[first_symbol],
            feature_columns=feature_columns,
            symbol=first_symbol,
            config=config,
            **kwargs,
        )

        # Override observation space for multiple assets
        obs_dim = self.config.lookback_window * self.n_features * self.n_assets + self.n_portfolio_features * self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Override action space for multiple assets
        if self.discrete_actions:
            # One action per asset
            self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        else:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.n_assets,),
                dtype=np.float32,
            )

    def _align_data(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Align all dataframes to common timestamps."""
        # Find common index
        common_idx = None
        for symbol, df in data.items():
            if "timestamp" in df.columns:
                idx = pd.DatetimeIndex(df["timestamp"])
            else:
                idx = df.index
            common_idx = idx if common_idx is None else common_idx.intersection(idx)

        # Reindex all dataframes
        aligned = {}
        for symbol, df in data.items():
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            aligned[symbol] = df.loc[common_idx].reset_index()

        return aligned
