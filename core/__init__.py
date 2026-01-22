"""
Core system components for the multi-agent trading system.
"""

from .environment import (
    TradingEnvironment,
    MultiAssetEnvironment,
    EnvironmentConfig,
    Position,
    PortfolioState,
)

from .portfolio import (
    PortfolioManager,
    RiskManager,
    RiskLimits,
    Trade,
    PositionInfo,
)

from .aggregator import (
    SignalAggregator,
    ConstitutionalAggregator,
    AggregationMethod,
    AggregatedSignal,
)

__all__ = [
    # Environment
    "TradingEnvironment",
    "MultiAssetEnvironment",
    "EnvironmentConfig",
    "Position",
    "PortfolioState",
    # Portfolio
    "PortfolioManager",
    "RiskManager",
    "RiskLimits",
    "Trade",
    "PositionInfo",
    # Aggregator
    "SignalAggregator",
    "ConstitutionalAggregator",
    "AggregationMethod",
    "AggregatedSignal",
]
