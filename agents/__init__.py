"""
Trading agents module.
Contains all 7 specialized AI agents for the multi-agent trading system.

Available Agents:
================

Signal Generators:
- MambaAgent: State space model with linear-time complexity (5x faster than Transformers)
- KANAgent: Kolmogorov-Arnold Networks for interpretable trading formulas
- LiquidAgent: Liquid Neural Networks that adapt to market regime changes
- TestTimeTrainingAgent: Adapts during inference using self-supervised learning

Scenario & Risk:
- DiffusionScenarioGenerator: Generates synthetic market scenarios using diffusion models
- ConstitutionalRiskManager: Self-critiquing safety layer with trading constitution

Coordination:
- MixtureOfExpertsCoordinator: Dynamic expert selection based on market conditions
"""

from .base import (
    BaseAgent,
    BaseNeuralAgent,
    Signal,
    Action,
    AgentState,
    CORE_FEATURES,
    MOMENTUM_FEATURES,
    TREND_FEATURES,
    VOLATILITY_FEATURES,
    ALL_FEATURES,
)

# Signal generating agents
from .mamba_agent import MambaAgent
from .kan_agent import KANAgent
from .liquid_agent import LiquidAgent
from .ttt_agent import TestTimeTrainingAgent

# Scenario generation and risk management
from .diffusion_agent import DiffusionScenarioGenerator
from .constitutional_agent import ConstitutionalRiskManager

# Coordination
from .moe_agent import MixtureOfExpertsCoordinator

__all__ = [
    # Base classes
    "BaseAgent",
    "BaseNeuralAgent",
    "Signal",
    "Action",
    "AgentState",
    # Feature sets
    "CORE_FEATURES",
    "MOMENTUM_FEATURES",
    "TREND_FEATURES",
    "VOLATILITY_FEATURES",
    "ALL_FEATURES",
    # Signal generating agents
    "MambaAgent",
    "KANAgent",
    "LiquidAgent",
    "TestTimeTrainingAgent",
    # Scenario and risk agents
    "DiffusionScenarioGenerator",
    "ConstitutionalRiskManager",
    # Coordination
    "MixtureOfExpertsCoordinator",
]
