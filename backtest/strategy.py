"""
Strategy evaluation framework.
Provides tools for testing, optimizing, and validating trading strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Any
import numpy as np
import pandas as pd
from loguru import logger

from agents.base import BaseAgent, Signal, Action
from .engine import BacktestEngine, BacktestConfig
from .metrics import PerformanceMetrics, TradeMetrics, compare_strategies


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""
    train_period: int = 252  # Training window in bars
    test_period: int = 63  # Testing window in bars
    step_size: int = 21  # Step size between windows
    min_train_samples: int = 100
    retrain: bool = True  # Whether to retrain on each window


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 1000
    confidence_level: float = 0.95
    block_size: int = 5  # For block bootstrap
    preserve_autocorrelation: bool = True


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    Wraps an agent and provides additional strategy logic.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logger.bind(strategy=name)

    @abstractmethod
    def generate_signal(
        self,
        data: pd.DataFrame,
        current_bar: int,
        positions: dict,
    ) -> Optional[Signal]:
        """
        Generate trading signal.

        Args:
            data: Full historical data
            current_bar: Current bar index
            positions: Current positions

        Returns:
            Trading signal or None
        """
        pass

    @abstractmethod
    def on_bar(self, bar: pd.Series):
        """Called on each new bar."""
        pass


class AgentStrategy(Strategy):
    """
    Strategy wrapper for agent-based trading.
    """

    def __init__(
        self,
        agent: BaseAgent,
        risk_filter: Optional[Callable[[Signal], bool]] = None,
        position_sizer: Optional[Callable[[Signal, float], float]] = None,
    ):
        """
        Initialize agent strategy.

        Args:
            agent: Trading agent
            risk_filter: Optional function to filter signals
            position_sizer: Optional function to adjust position sizes
        """
        super().__init__(agent.name)
        self.agent = agent
        self.risk_filter = risk_filter
        self.position_sizer = position_sizer

        # Get agent's required lookback
        self.lookback = getattr(agent, "sequence_length", 60)

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_bar: int,
        positions: dict,
    ) -> Optional[Signal]:
        """Generate signal using the agent."""
        if current_bar < self.lookback:
            return None

        # Get features for agent
        feature_cols = self.agent.get_feature_names()
        available = [c for c in feature_cols if c in data.columns]

        start_idx = max(0, current_bar - self.lookback)
        features = data.iloc[start_idx:current_bar + 1][available]

        # Get signal from agent
        symbol = data.iloc[current_bar].get("symbol", "ASSET")
        signal = self.agent.predict(features, symbol)

        # Apply risk filter
        if self.risk_filter and not self.risk_filter(signal):
            return None

        # Apply position sizer
        if self.position_sizer:
            equity = sum(p.market_value for p in positions.values())
            signal.position_size = self.position_sizer(signal, equity)

        return signal

    def on_bar(self, bar: pd.Series):
        """Called on each new bar."""
        pass


class StrategyEvaluator:
    """
    Comprehensive strategy evaluation framework.

    Features:
    - Walk-forward analysis
    - Monte Carlo simulation
    - Parameter sensitivity analysis
    - Benchmark comparison
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: Optional[BacktestConfig] = None,
    ):
        """
        Initialize strategy evaluator.

        Args:
            data: Full historical data
            config: Backtest configuration
        """
        self.data = data
        self.config = config or BacktestConfig()
        self.logger = logger.bind(component="evaluator")

    def run_backtest(
        self,
        agent: BaseAgent,
        symbol: str = "ASSET",
    ) -> dict:
        """
        Run a simple backtest.

        Args:
            agent: Trading agent
            symbol: Asset symbol

        Returns:
            Backtest results
        """
        engine = BacktestEngine(self.config)
        return engine.run(self.data, agent, symbol)

    def walk_forward_analysis(
        self,
        agent: BaseAgent,
        wf_config: Optional[WalkForwardConfig] = None,
        train_fn: Optional[Callable[[BaseAgent, pd.DataFrame], None]] = None,
    ) -> dict:
        """
        Perform walk-forward analysis.

        Walk-forward analysis tests the strategy on out-of-sample data
        after training on in-sample data, simulating real trading.

        Args:
            agent: Trading agent
            wf_config: Walk-forward configuration
            train_fn: Optional function to train the agent

        Returns:
            Walk-forward results
        """
        wf_config = wf_config or WalkForwardConfig()
        n_bars = len(self.data)

        self.logger.info(
            f"Walk-forward analysis: train={wf_config.train_period}, "
            f"test={wf_config.test_period}, step={wf_config.step_size}"
        )

        windows = []
        all_equity = []
        all_returns = []

        # Generate windows
        start = 0
        while start + wf_config.train_period + wf_config.test_period <= n_bars:
            train_end = start + wf_config.train_period
            test_end = train_end + wf_config.test_period

            windows.append({
                "train_start": start,
                "train_end": train_end,
                "test_start": train_end,
                "test_end": test_end,
            })

            start += wf_config.step_size

        self.logger.info(f"Generated {len(windows)} walk-forward windows")

        # Run each window
        for i, window in enumerate(windows):
            train_data = self.data.iloc[window["train_start"]:window["train_end"]]
            test_data = self.data.iloc[window["test_start"]:window["test_end"]]

            # Train agent on in-sample data
            if wf_config.retrain and train_fn:
                self.logger.debug(f"Training on window {i + 1}/{len(windows)}")
                train_fn(agent, train_data)

            # Test on out-of-sample data
            engine = BacktestEngine(self.config)
            results = engine.run(test_data, agent)

            window["results"] = results
            window["return"] = results["total_return"]
            window["sharpe"] = results["sharpe_ratio"]

            # Collect equity and returns
            if "equity_curve" in results:
                all_equity.extend(results["equity_curve"]["equity"].tolist())
            if "daily_returns" in results:
                all_returns.extend(results["daily_returns"].tolist())

        # Calculate aggregate metrics
        returns = np.array(all_returns)
        equity = np.array(all_equity) if all_equity else np.array([self.config.initial_capital])

        metrics = PerformanceMetrics(returns, equity)

        return {
            "windows": windows,
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown,
            "window_returns": [w["return"] for w in windows],
            "window_sharpes": [w["sharpe"] for w in windows],
            "equity_curve": equity,
            "returns": returns,
        }

    def monte_carlo_simulation(
        self,
        agent: BaseAgent,
        mc_config: Optional[MonteCarloConfig] = None,
    ) -> dict:
        """
        Run Monte Carlo simulation.

        Uses bootstrap resampling of returns to estimate
        confidence intervals for strategy performance.

        Args:
            agent: Trading agent
            mc_config: Monte Carlo configuration

        Returns:
            Monte Carlo results
        """
        mc_config = mc_config or MonteCarloConfig()

        # First run backtest to get original returns
        base_results = self.run_backtest(agent)
        original_returns = base_results.get("daily_returns", np.array([]))

        if len(original_returns) < 30:
            self.logger.warning("Not enough returns for Monte Carlo simulation")
            return {"error": "Insufficient data"}

        self.logger.info(f"Running {mc_config.n_simulations} Monte Carlo simulations")

        simulated_returns = []
        simulated_sharpes = []
        simulated_drawdowns = []
        simulated_final_equity = []

        for i in range(mc_config.n_simulations):
            # Bootstrap resample returns
            if mc_config.preserve_autocorrelation:
                # Block bootstrap to preserve autocorrelation
                resampled = self._block_bootstrap(
                    original_returns,
                    mc_config.block_size,
                )
            else:
                # Simple bootstrap
                indices = np.random.randint(0, len(original_returns), len(original_returns))
                resampled = original_returns[indices]

            # Calculate metrics for simulated path
            equity = self.config.initial_capital * np.cumprod(1 + resampled)
            metrics = PerformanceMetrics(resampled, equity)

            simulated_returns.append(metrics.total_return)
            simulated_sharpes.append(metrics.sharpe_ratio)
            simulated_drawdowns.append(metrics.max_drawdown)
            simulated_final_equity.append(equity[-1])

        # Calculate confidence intervals
        ci_level = mc_config.confidence_level
        lower_pct = (1 - ci_level) / 2 * 100
        upper_pct = (1 + ci_level) / 2 * 100

        return {
            "n_simulations": mc_config.n_simulations,
            "original_return": base_results["total_return"],
            "original_sharpe": base_results["sharpe_ratio"],

            # Return statistics
            "return_mean": np.mean(simulated_returns),
            "return_std": np.std(simulated_returns),
            "return_ci": (
                np.percentile(simulated_returns, lower_pct),
                np.percentile(simulated_returns, upper_pct),
            ),

            # Sharpe statistics
            "sharpe_mean": np.mean(simulated_sharpes),
            "sharpe_std": np.std(simulated_sharpes),
            "sharpe_ci": (
                np.percentile(simulated_sharpes, lower_pct),
                np.percentile(simulated_sharpes, upper_pct),
            ),

            # Drawdown statistics
            "drawdown_mean": np.mean(simulated_drawdowns),
            "drawdown_worst": np.max(simulated_drawdowns),
            "drawdown_ci": (
                np.percentile(simulated_drawdowns, lower_pct),
                np.percentile(simulated_drawdowns, upper_pct),
            ),

            # Probability of loss
            "prob_loss": np.mean(np.array(simulated_returns) < 0),
            "prob_sharpe_positive": np.mean(np.array(simulated_sharpes) > 0),

            # Raw simulations
            "simulated_returns": simulated_returns,
            "simulated_sharpes": simulated_sharpes,
            "simulated_drawdowns": simulated_drawdowns,
        }

    def _block_bootstrap(
        self,
        data: np.ndarray,
        block_size: int,
    ) -> np.ndarray:
        """Perform block bootstrap to preserve autocorrelation."""
        n = len(data)
        n_blocks = int(np.ceil(n / block_size))

        resampled = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n - block_size + 1)
            resampled.extend(data[start:start + block_size])

        return np.array(resampled[:n])

    def parameter_sensitivity(
        self,
        agent_class: type,
        param_name: str,
        param_values: list,
        base_config: dict,
    ) -> dict:
        """
        Analyze sensitivity to a parameter.

        Args:
            agent_class: Agent class to instantiate
            param_name: Name of parameter to vary
            param_values: Values to test
            base_config: Base agent configuration

        Returns:
            Sensitivity analysis results
        """
        results = []

        self.logger.info(f"Parameter sensitivity: {param_name} with {len(param_values)} values")

        for value in param_values:
            config = base_config.copy()
            config[param_name] = value

            # Create agent with this config
            agent = agent_class(**config)

            # Run backtest
            bt_results = self.run_backtest(agent)

            results.append({
                "param_value": value,
                "total_return": bt_results["total_return"],
                "sharpe_ratio": bt_results["sharpe_ratio"],
                "max_drawdown": bt_results["max_drawdown"],
                "win_rate": bt_results["win_rate"],
            })

        return {
            "param_name": param_name,
            "results": pd.DataFrame(results),
        }

    def compare_agents(
        self,
        agents: list[BaseAgent],
        benchmark_returns: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple agents.

        Args:
            agents: List of agents to compare
            benchmark_returns: Optional benchmark returns

        Returns:
            Comparison DataFrame
        """
        results = {}

        for agent in agents:
            self.logger.info(f"Evaluating agent: {agent.name}")
            bt_results = self.run_backtest(agent)
            results[agent.name] = bt_results

        # Add benchmark if provided
        if benchmark_returns is not None:
            equity = self.config.initial_capital * np.cumprod(1 + benchmark_returns)
            metrics = PerformanceMetrics(benchmark_returns, equity)
            results["Benchmark"] = {
                "total_return": metrics.total_return,
                "annualized_return": metrics.annualized_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "volatility": metrics.volatility,
                "win_rate": metrics.win_rate,
                "total_trades": 0,
                "profit_factor": 0,
            }

        return compare_strategies(results)


class BenchmarkStrategy:
    """
    Benchmark strategies for comparison.
    """

    @staticmethod
    def buy_and_hold(data: pd.DataFrame) -> np.ndarray:
        """Calculate buy and hold returns."""
        if "close" not in data.columns:
            return np.array([0.0])

        prices = data["close"].values
        returns = np.diff(prices) / prices[:-1]
        return returns

    @staticmethod
    def equal_weight_rebalance(
        data: dict[str, pd.DataFrame],
        rebalance_freq: int = 21,
    ) -> np.ndarray:
        """
        Equal weight portfolio with periodic rebalancing.

        Args:
            data: Dictionary of symbol -> DataFrame
            rebalance_freq: Rebalancing frequency in bars

        Returns:
            Portfolio returns
        """
        # Align all data
        all_returns = {}
        for symbol, df in data.items():
            if "close" in df.columns:
                prices = df["close"].values
                all_returns[symbol] = np.diff(prices) / prices[:-1]

        if not all_returns:
            return np.array([0.0])

        # Find common length
        min_len = min(len(r) for r in all_returns.values())

        # Equal weight
        n_assets = len(all_returns)
        weight = 1.0 / n_assets

        portfolio_returns = np.zeros(min_len)
        for returns in all_returns.values():
            portfolio_returns += weight * returns[:min_len]

        return portfolio_returns


if __name__ == "__main__":
    # Test strategy evaluation
    print("Testing Strategy Evaluation...")
    print("=" * 50)

    # Create synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start="2022-01-01", periods=n, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))

    data = pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 + np.random.randn(n) * 0.01),
        "high": prices * (1 + np.abs(np.random.randn(n) * 0.02)),
        "low": prices * (1 - np.abs(np.random.randn(n) * 0.02)),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, n),
        "returns": np.concatenate([[0], np.diff(prices) / prices[:-1]]),
        "rsi_14": np.random.rand(n) * 100,
        "macd": np.random.randn(n) * 0.1,
    })

    # Create evaluator
    evaluator = StrategyEvaluator(data)

    # Test walk-forward (would need a real agent)
    print("\nTesting Walk-Forward Config...")
    wf_config = WalkForwardConfig(train_period=100, test_period=50, step_size=25)
    print(f"Config: train={wf_config.train_period}, test={wf_config.test_period}")

    # Test benchmark strategies
    print("\nTesting Benchmark Strategies...")
    bh_returns = BenchmarkStrategy.buy_and_hold(data)
    print(f"Buy and hold returns: {len(bh_returns)} periods")
    print(f"Total return: {np.prod(1 + bh_returns) - 1:.2%}")

    print("\nStrategy evaluation tests passed!")
