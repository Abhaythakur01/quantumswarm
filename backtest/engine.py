"""
Backtesting engine for evaluating trading strategies.
Simulates trading on historical data with realistic execution modeling.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable
from enum import Enum
import numpy as np
import pandas as pd
from loguru import logger

from agents.base import BaseAgent, Signal, Action


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    filled: bool = False
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Trade:
    """Represents an executed trade."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    holding_period: int = 0  # in bars

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005  # 0.05% slippage
    max_position_size: float = 0.25  # Max 25% per position
    allow_shorting: bool = False
    margin_requirement: float = 1.0  # 1.0 = no margin
    risk_free_rate: float = 0.02  # For Sharpe calculation
    trading_days_per_year: int = 252


@dataclass
class Position:
    """Current position in an asset."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price

    def update_price(self, price: float):
        self.current_price = price
        self.unrealized_pnl = self.market_value - self.cost_basis


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Features:
    - Realistic order execution with slippage and commission
    - Position tracking and P&L calculation
    - Support for multiple assets
    - Flexible signal handling
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.logger = logger.bind(component="backtest")

        # State
        self.cash = self.config.initial_capital
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self.trades: list[Trade] = []
        self.open_trades: dict[str, Trade] = {}

        # History tracking
        self.equity_curve: list[dict] = []
        self.daily_returns: list[float] = []
        self.signals_history: list[dict] = []

        # Current state
        self.current_bar = 0
        self.current_timestamp: Optional[datetime] = None
        self.current_prices: dict[str, float] = {}

    def reset(self):
        """Reset engine to initial state."""
        self.cash = self.config.initial_capital
        self.positions = {}
        self.orders = []
        self.trades = []
        self.open_trades = {}
        self.equity_curve = []
        self.daily_returns = []
        self.signals_history = []
        self.current_bar = 0
        self.current_timestamp = None
        self.current_prices = {}

    @property
    def equity(self) -> float:
        """Total portfolio equity."""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value

    @property
    def buying_power(self) -> float:
        """Available buying power."""
        return self.cash / self.config.margin_requirement

    def run(
        self,
        data: pd.DataFrame,
        agent: BaseAgent,
        symbol: str = "ASSET",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """
        Run backtest with a single agent on a single asset.

        Args:
            data: DataFrame with OHLCV and features
            agent: Trading agent to evaluate
            symbol: Asset symbol
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with backtest results
        """
        self.reset()
        n_bars = len(data)

        self.logger.info(f"Starting backtest: {n_bars} bars, agent={agent.name}")

        # Get feature columns for the agent
        feature_cols = agent.get_feature_names()
        available_features = [f for f in feature_cols if f in data.columns]

        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            self.logger.warning(f"Missing features: {missing}")

        # Minimum bars needed for agent
        min_bars = getattr(agent, "sequence_length", 60)

        for i in range(min_bars, n_bars):
            self.current_bar = i
            self.current_timestamp = data.iloc[i].get("timestamp", pd.Timestamp.now())

            # Update current prices
            bar = data.iloc[i]
            self.current_prices[symbol] = bar["close"]

            # Update positions
            self._update_positions()

            # Get features for agent
            start_idx = max(0, i - min_bars)
            features = data.iloc[start_idx:i + 1][available_features]

            # Get signal from agent
            try:
                signal = agent.predict(features, symbol)
                self._record_signal(signal)

                # Process signal
                self._process_signal(signal, bar)

            except Exception as e:
                self.logger.error(f"Error at bar {i}: {e}")

            # Record equity
            self._record_equity()

            # Progress callback
            if progress_callback and i % 100 == 0:
                progress_callback(i, n_bars)

        # Close any remaining positions
        self._close_all_positions()

        # Calculate final metrics
        results = self._calculate_results()

        self.logger.info(
            f"Backtest complete: Return={results['total_return']:.2%}, "
            f"Sharpe={results['sharpe_ratio']:.2f}, "
            f"MaxDD={results['max_drawdown']:.2%}"
        )

        return results

    def run_multi_agent(
        self,
        data: pd.DataFrame,
        agents: list[BaseAgent],
        aggregation_method: str = "majority",
        symbol: str = "ASSET",
    ) -> dict:
        """
        Run backtest with multiple agents.

        Args:
            data: DataFrame with OHLCV and features
            agents: List of trading agents
            aggregation_method: How to combine signals ('majority', 'average', 'best')
            symbol: Asset symbol

        Returns:
            Dictionary with backtest results
        """
        from core.aggregator import SignalAggregator, AggregationMethod

        # Create aggregator
        method_map = {
            "majority": AggregationMethod.MAJORITY_VOTE,
            "average": AggregationMethod.WEIGHTED_AVERAGE,
            "confidence": AggregationMethod.CONFIDENCE_WEIGHTED,
            "best": AggregationMethod.BEST_CONFIDENCE,
        }

        aggregator = SignalAggregator(method=method_map.get(aggregation_method, AggregationMethod.CONFIDENCE_WEIGHTED))

        for agent in agents:
            aggregator.register_agent(agent)

        self.reset()
        n_bars = len(data)
        min_bars = max(getattr(a, "sequence_length", 60) for a in agents)

        self.logger.info(f"Starting multi-agent backtest: {len(agents)} agents")

        for i in range(min_bars, n_bars):
            self.current_bar = i
            self.current_timestamp = data.iloc[i].get("timestamp", pd.Timestamp.now())

            bar = data.iloc[i]
            self.current_prices[symbol] = bar["close"]
            self._update_positions()

            # Get features
            start_idx = max(0, i - min_bars)
            features = data.iloc[start_idx:i + 1]

            # Get aggregated signal
            try:
                agg_signal = aggregator.aggregate(features, symbol)

                # Convert to regular signal
                signal = Signal(
                    symbol=agg_signal.symbol,
                    action=agg_signal.action,
                    confidence=agg_signal.confidence,
                    position_size=agg_signal.position_size,
                    timestamp=agg_signal.timestamp,
                    agent_name="aggregator",
                    metadata={
                        "contributors": agg_signal.contributing_agents,
                        "agreement": agg_signal.agreement_score,
                    },
                )

                self._record_signal(signal)
                self._process_signal(signal, bar)

            except Exception as e:
                self.logger.error(f"Error at bar {i}: {e}")

            self._record_equity()

        self._close_all_positions()
        return self._calculate_results()

    def _process_signal(self, signal: Signal, bar: pd.Series):
        """Process a trading signal."""
        symbol = signal.symbol
        current_price = bar["close"]

        # Skip if HOLD or low confidence
        if signal.action == Action.HOLD or signal.confidence < 0.5:
            return

        # Calculate position size
        target_value = self.equity * signal.position_size * self.config.max_position_size
        current_position = self.positions.get(symbol)
        current_value = current_position.market_value if current_position else 0

        if signal.action == Action.BUY:
            if current_value < target_value:
                # Buy
                buy_value = min(target_value - current_value, self.buying_power * 0.95)
                if buy_value > 100:  # Minimum trade size
                    quantity = buy_value / current_price
                    self._execute_buy(symbol, quantity, current_price)

        elif signal.action == Action.SELL:
            if current_position and current_position.quantity > 0:
                # Sell all
                self._execute_sell(symbol, current_position.quantity, current_price)

    def _execute_buy(self, symbol: str, quantity: float, price: float):
        """Execute a buy order."""
        # Apply slippage (worse price for buyer)
        fill_price = price * (1 + self.config.slippage_rate)
        total_cost = quantity * fill_price

        # Apply commission
        commission = total_cost * self.config.commission_rate
        total_cost += commission

        if total_cost > self.cash:
            # Reduce quantity to fit available cash
            quantity = (self.cash - commission) / fill_price
            total_cost = quantity * fill_price + commission

        if quantity <= 0:
            return

        # Update cash
        self.cash -= total_cost

        # Update position
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_qty = pos.quantity + quantity
            pos.avg_price = (pos.cost_basis + quantity * fill_price) / total_qty
            pos.quantity = total_qty
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=fill_price,
                current_price=price,
            )

        # Record trade
        trade = Trade(
            symbol=symbol,
            side="buy",
            quantity=quantity,
            entry_price=fill_price,
            entry_time=self.current_timestamp,
            commission=commission,
        )
        self.open_trades[symbol] = trade

        # Create order record
        order = Order(
            symbol=symbol,
            side="buy",
            quantity=quantity,
            filled=True,
            fill_price=fill_price,
            fill_timestamp=self.current_timestamp,
            commission=commission,
        )
        self.orders.append(order)

    def _execute_sell(self, symbol: str, quantity: float, price: float):
        """Execute a sell order."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        sell_qty = min(quantity, pos.quantity)

        # Apply slippage (worse price for seller)
        fill_price = price * (1 - self.config.slippage_rate)
        proceeds = sell_qty * fill_price

        # Apply commission
        commission = proceeds * self.config.commission_rate
        proceeds -= commission

        # Calculate P&L
        cost_basis = sell_qty * pos.avg_price
        pnl = proceeds - cost_basis + commission  # Add back commission for true P&L

        # Update cash
        self.cash += proceeds

        # Update position
        pos.quantity -= sell_qty
        pos.realized_pnl += pnl

        if pos.quantity <= 0.0001:
            del self.positions[symbol]

        # Close trade
        if symbol in self.open_trades:
            trade = self.open_trades[symbol]
            trade.exit_price = fill_price
            trade.exit_time = self.current_timestamp
            trade.pnl = (fill_price - trade.entry_price) * trade.quantity - trade.commission - commission
            trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)
            trade.holding_period = self.current_bar - self.trades[-1].holding_period if self.trades else 0
            self.trades.append(trade)
            del self.open_trades[symbol]

        # Create order record
        order = Order(
            symbol=symbol,
            side="sell",
            quantity=sell_qty,
            filled=True,
            fill_price=fill_price,
            fill_timestamp=self.current_timestamp,
            commission=commission,
        )
        self.orders.append(order)

    def _update_positions(self):
        """Update position prices and P&L."""
        for symbol, pos in self.positions.items():
            if symbol in self.current_prices:
                pos.update_price(self.current_prices[symbol])

    def _close_all_positions(self):
        """Close all remaining positions at current prices."""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            if pos.quantity > 0:
                price = self.current_prices.get(symbol, pos.current_price)
                self._execute_sell(symbol, pos.quantity, price)

    def _record_equity(self):
        """Record current equity to history."""
        self.equity_curve.append({
            "bar": self.current_bar,
            "timestamp": self.current_timestamp,
            "equity": self.equity,
            "cash": self.cash,
            "positions_value": self.equity - self.cash,
        })

        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]["equity"]
            daily_return = (self.equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_return)

    def _record_signal(self, signal: Signal):
        """Record signal to history."""
        self.signals_history.append({
            "bar": self.current_bar,
            "timestamp": self.current_timestamp,
            "symbol": signal.symbol,
            "action": signal.action.name,
            "confidence": signal.confidence,
            "position_size": signal.position_size,
            "agent": signal.agent_name,
        })

    def _calculate_results(self) -> dict:
        """Calculate backtest results and metrics."""
        from .metrics import PerformanceMetrics

        # Create equity DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        returns = np.array(self.daily_returns)

        # Calculate metrics
        metrics = PerformanceMetrics(
            returns=returns,
            equity_curve=equity_df["equity"].values if len(equity_df) > 0 else np.array([self.config.initial_capital]),
            risk_free_rate=self.config.risk_free_rate,
            trading_days=self.config.trading_days_per_year,
        )

        # Trade statistics
        closed_trades = [t for t in self.trades if t.is_closed]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]

        results = {
            # Returns
            "total_return": metrics.total_return,
            "annualized_return": metrics.annualized_return,
            "daily_returns": returns,

            # Risk metrics
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "calmar_ratio": metrics.calmar_ratio,
            "max_drawdown": metrics.max_drawdown,
            "volatility": metrics.volatility,

            # Trade statistics
            "total_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / len(closed_trades) if closed_trades else 0,
            "profit_factor": metrics.profit_factor(closed_trades),
            "avg_trade_pnl": np.mean([t.pnl for t in closed_trades]) if closed_trades else 0,
            "avg_win": np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            "avg_loss": np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,

            # Portfolio
            "final_equity": self.equity,
            "initial_capital": self.config.initial_capital,

            # History
            "equity_curve": equity_df,
            "trades": closed_trades,
            "signals": self.signals_history,
        }

        return results

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame(self.equity_curve)

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "entry_time": t.entry_time,
                "exit_price": t.exit_price,
                "exit_time": t.exit_time,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "holding_period": t.holding_period,
            })

        return pd.DataFrame(records)
