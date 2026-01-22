"""
Signal aggregator for multi-agent coordination.
Combines signals from multiple trading agents using various strategies.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable
import numpy as np
import pandas as pd
from loguru import logger

from agents.base import Signal, Action, BaseAgent


class AggregationMethod(Enum):
    """Methods for combining agent signals."""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    MIXTURE_OF_EXPERTS = "mixture_of_experts"
    UNANIMOUS = "unanimous"
    BEST_CONFIDENCE = "best_confidence"


@dataclass
class AgentWeight:
    """Weight and metadata for an agent in the ensemble."""
    agent: BaseAgent
    weight: float = 1.0
    recent_performance: float = 0.0
    total_signals: int = 0
    correct_signals: int = 0

    @property
    def accuracy(self) -> float:
        if self.total_signals == 0:
            return 0.5
        return self.correct_signals / self.total_signals


@dataclass
class AggregatedSignal:
    """
    Combined signal from multiple agents.
    """
    symbol: str
    action: Action
    confidence: float
    position_size: float
    timestamp: datetime
    contributing_agents: list[str]
    agent_signals: dict[str, Signal]
    agreement_score: float  # How much agents agree [0, 1]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "action": self.action.name,
            "confidence": self.confidence,
            "position_size": self.position_size,
            "timestamp": self.timestamp.isoformat(),
            "contributing_agents": self.contributing_agents,
            "agreement_score": self.agreement_score,
            "metadata": self.metadata,
        }


class SignalAggregator:
    """
    Combines signals from multiple trading agents.

    Supports multiple aggregation strategies:
    - Majority vote: Action with most votes wins
    - Weighted average: Weights can be static or dynamic
    - Confidence-weighted: Higher confidence signals count more
    - Mixture of experts: Gating network selects experts
    - Unanimous: Only act when all agents agree
    """

    def __init__(
        self,
        method: AggregationMethod = AggregationMethod.CONFIDENCE_WEIGHTED,
        min_confidence: float = 0.5,
        min_agreement: float = 0.5,
        update_weights: bool = True,
    ):
        """
        Initialize signal aggregator.

        Args:
            method: Aggregation method to use
            min_confidence: Minimum confidence to act
            min_agreement: Minimum agreement score to act
            update_weights: Whether to update weights based on performance
        """
        self.method = method
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self.update_weights = update_weights

        self.agents: dict[str, AgentWeight] = {}
        self.signal_history: list[AggregatedSignal] = []
        self.logger = logger.bind(component="aggregator")

        # For mixture of experts
        self.gating_network: Optional[Callable] = None

    def register_agent(
        self,
        agent: BaseAgent,
        weight: float = 1.0,
    ):
        """
        Register an agent with the aggregator.

        Args:
            agent: Trading agent instance
            weight: Initial weight for this agent
        """
        self.agents[agent.name] = AgentWeight(
            agent=agent,
            weight=weight,
        )
        self.logger.info(f"Registered agent: {agent.name} (weight={weight})")

    def remove_agent(self, agent_name: str):
        """Remove an agent from the aggregator."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            self.logger.info(f"Removed agent: {agent_name}")

    def set_gating_network(self, gating_fn: Callable[[pd.DataFrame], dict[str, float]]):
        """
        Set gating network for mixture of experts.

        Args:
            gating_fn: Function that takes features and returns agent weights
        """
        self.gating_network = gating_fn
        self.method = AggregationMethod.MIXTURE_OF_EXPERTS

    def aggregate(
        self,
        features: pd.DataFrame,
        symbol: str,
        timestamp: Optional[datetime] = None,
    ) -> AggregatedSignal:
        """
        Aggregate signals from all registered agents.

        Args:
            features: Market features
            symbol: Asset symbol
            timestamp: Signal timestamp

        Returns:
            Aggregated signal combining all agent outputs
        """
        timestamp = timestamp or datetime.now()

        # Collect signals from all agents
        agent_signals = {}
        for name, agent_weight in self.agents.items():
            try:
                signal = agent_weight.agent.predict(features, symbol)
                agent_signals[name] = signal
                agent_weight.total_signals += 1
            except Exception as e:
                self.logger.error(f"Error getting signal from {name}: {e}")

        if not agent_signals:
            # No signals - return HOLD
            return AggregatedSignal(
                symbol=symbol,
                action=Action.HOLD,
                confidence=0.0,
                position_size=0.0,
                timestamp=timestamp,
                contributing_agents=[],
                agent_signals={},
                agreement_score=0.0,
            )

        # Aggregate based on method
        if self.method == AggregationMethod.MAJORITY_VOTE:
            result = self._majority_vote(agent_signals)
        elif self.method == AggregationMethod.WEIGHTED_AVERAGE:
            result = self._weighted_average(agent_signals)
        elif self.method == AggregationMethod.CONFIDENCE_WEIGHTED:
            result = self._confidence_weighted(agent_signals)
        elif self.method == AggregationMethod.MIXTURE_OF_EXPERTS:
            result = self._mixture_of_experts(agent_signals, features)
        elif self.method == AggregationMethod.UNANIMOUS:
            result = self._unanimous(agent_signals)
        elif self.method == AggregationMethod.BEST_CONFIDENCE:
            result = self._best_confidence(agent_signals)
        else:
            result = self._confidence_weighted(agent_signals)

        # Calculate agreement score
        agreement = self._calculate_agreement(agent_signals)

        # Build aggregated signal
        aggregated = AggregatedSignal(
            symbol=symbol,
            action=result["action"],
            confidence=result["confidence"],
            position_size=result["position_size"],
            timestamp=timestamp,
            contributing_agents=result["contributors"],
            agent_signals=agent_signals,
            agreement_score=agreement,
            metadata={
                "method": self.method.value,
                "num_agents": len(agent_signals),
            },
        )

        # Apply minimum thresholds
        if aggregated.confidence < self.min_confidence:
            aggregated.action = Action.HOLD
            aggregated.position_size = 0.0

        if aggregated.agreement_score < self.min_agreement:
            aggregated.action = Action.HOLD
            aggregated.position_size = 0.0

        # Record history
        self.signal_history.append(aggregated)

        return aggregated

    def _majority_vote(self, signals: dict[str, Signal]) -> dict:
        """Simple majority vote on action."""
        votes = {Action.HOLD: 0, Action.BUY: 0, Action.SELL: 0}

        for signal in signals.values():
            votes[signal.action] += 1

        winning_action = max(votes, key=votes.get)
        vote_count = votes[winning_action]
        total_votes = len(signals)

        # Get signals that voted for winning action
        contributors = [
            name for name, sig in signals.items()
            if sig.action == winning_action
        ]

        # Average confidence and position size of winners
        winner_signals = [s for s in signals.values() if s.action == winning_action]
        avg_confidence = np.mean([s.confidence for s in winner_signals])
        avg_position = np.mean([s.position_size for s in winner_signals])

        return {
            "action": winning_action,
            "confidence": avg_confidence * (vote_count / total_votes),
            "position_size": avg_position,
            "contributors": contributors,
        }

    def _weighted_average(self, signals: dict[str, Signal]) -> dict:
        """Weighted average using agent weights."""
        action_scores = {Action.HOLD: 0.0, Action.BUY: 0.0, Action.SELL: 0.0}
        total_weight = 0.0
        position_sum = 0.0

        for name, signal in signals.items():
            weight = self.agents[name].weight
            action_scores[signal.action] += weight * signal.confidence
            position_sum += weight * signal.position_size
            total_weight += weight

        if total_weight == 0:
            return {
                "action": Action.HOLD,
                "confidence": 0.0,
                "position_size": 0.0,
                "contributors": [],
            }

        # Normalize
        for action in action_scores:
            action_scores[action] /= total_weight
        position_sum /= total_weight

        winning_action = max(action_scores, key=action_scores.get)

        contributors = [
            name for name, sig in signals.items()
            if sig.action == winning_action
        ]

        return {
            "action": winning_action,
            "confidence": action_scores[winning_action],
            "position_size": position_sum,
            "contributors": contributors,
        }

    def _confidence_weighted(self, signals: dict[str, Signal]) -> dict:
        """Weight by each signal's confidence."""
        action_scores = {Action.HOLD: 0.0, Action.BUY: 0.0, Action.SELL: 0.0}
        total_confidence = 0.0
        position_sum = 0.0

        for name, signal in signals.items():
            conf = signal.confidence
            action_scores[signal.action] += conf
            position_sum += conf * signal.position_size
            total_confidence += conf

        if total_confidence == 0:
            return {
                "action": Action.HOLD,
                "confidence": 0.0,
                "position_size": 0.0,
                "contributors": [],
            }

        # Normalize
        for action in action_scores:
            action_scores[action] /= total_confidence
        position_sum /= total_confidence

        winning_action = max(action_scores, key=action_scores.get)

        contributors = [
            name for name, sig in signals.items()
            if sig.action == winning_action
        ]

        return {
            "action": winning_action,
            "confidence": action_scores[winning_action],
            "position_size": position_sum,
            "contributors": contributors,
        }

    def _mixture_of_experts(
        self,
        signals: dict[str, Signal],
        features: pd.DataFrame,
    ) -> dict:
        """Use gating network to select experts dynamically."""
        if self.gating_network is None:
            # Fall back to confidence-weighted
            return self._confidence_weighted(signals)

        # Get dynamic weights from gating network
        try:
            expert_weights = self.gating_network(features)
        except Exception as e:
            self.logger.error(f"Gating network error: {e}")
            return self._confidence_weighted(signals)

        action_scores = {Action.HOLD: 0.0, Action.BUY: 0.0, Action.SELL: 0.0}
        total_weight = 0.0
        position_sum = 0.0

        for name, signal in signals.items():
            weight = expert_weights.get(name, 0.0)
            action_scores[signal.action] += weight * signal.confidence
            position_sum += weight * signal.position_size
            total_weight += weight

        if total_weight == 0:
            return self._confidence_weighted(signals)

        # Normalize
        for action in action_scores:
            action_scores[action] /= total_weight
        position_sum /= total_weight

        winning_action = max(action_scores, key=action_scores.get)

        contributors = [
            name for name, sig in signals.items()
            if sig.action == winning_action and expert_weights.get(name, 0) > 0.1
        ]

        return {
            "action": winning_action,
            "confidence": action_scores[winning_action],
            "position_size": position_sum,
            "contributors": contributors,
        }

    def _unanimous(self, signals: dict[str, Signal]) -> dict:
        """Only act if all agents agree."""
        if len(signals) < 2:
            sig = list(signals.values())[0]
            return {
                "action": sig.action,
                "confidence": sig.confidence,
                "position_size": sig.position_size,
                "contributors": list(signals.keys()),
            }

        actions = set(s.action for s in signals.values())

        if len(actions) == 1:
            # All agree
            action = actions.pop()
            avg_conf = np.mean([s.confidence for s in signals.values()])
            avg_pos = np.mean([s.position_size for s in signals.values()])

            return {
                "action": action,
                "confidence": avg_conf,
                "position_size": avg_pos,
                "contributors": list(signals.keys()),
            }
        else:
            # No consensus - hold
            return {
                "action": Action.HOLD,
                "confidence": 0.0,
                "position_size": 0.0,
                "contributors": [],
            }

    def _best_confidence(self, signals: dict[str, Signal]) -> dict:
        """Use signal from agent with highest confidence."""
        best_name = max(signals.keys(), key=lambda n: signals[n].confidence)
        best_signal = signals[best_name]

        return {
            "action": best_signal.action,
            "confidence": best_signal.confidence,
            "position_size": best_signal.position_size,
            "contributors": [best_name],
        }

    def _calculate_agreement(self, signals: dict[str, Signal]) -> float:
        """Calculate how much agents agree on the action."""
        if len(signals) <= 1:
            return 1.0

        actions = [s.action for s in signals.values()]
        most_common = max(set(actions), key=actions.count)
        agreement = actions.count(most_common) / len(actions)

        return agreement

    def update_performance(
        self,
        symbol: str,
        actual_return: float,
        lookback_signals: int = 1,
    ):
        """
        Update agent weights based on signal performance.

        Args:
            symbol: Asset symbol
            actual_return: Actual return achieved
            lookback_signals: Number of recent signals to evaluate
        """
        if not self.update_weights:
            return

        # Get recent signals for this symbol
        recent = [
            s for s in self.signal_history[-lookback_signals:]
            if s.symbol == symbol
        ]

        if not recent:
            return

        for agg_signal in recent:
            for agent_name, signal in agg_signal.agent_signals.items():
                if agent_name not in self.agents:
                    continue

                # Check if signal direction was correct
                correct = False
                if signal.action == Action.BUY and actual_return > 0:
                    correct = True
                elif signal.action == Action.SELL and actual_return < 0:
                    correct = True
                elif signal.action == Action.HOLD and abs(actual_return) < 0.001:
                    correct = True

                if correct:
                    self.agents[agent_name].correct_signals += 1

                # Update recent performance (exponential moving average)
                alpha = 0.1
                agent_return = actual_return if signal.action == Action.BUY else -actual_return
                self.agents[agent_name].recent_performance = (
                    alpha * agent_return +
                    (1 - alpha) * self.agents[agent_name].recent_performance
                )

    def get_agent_stats(self) -> pd.DataFrame:
        """Get performance statistics for all agents."""
        records = []
        for name, agent_weight in self.agents.items():
            records.append({
                "agent": name,
                "weight": agent_weight.weight,
                "total_signals": agent_weight.total_signals,
                "accuracy": agent_weight.accuracy,
                "recent_performance": agent_weight.recent_performance,
            })

        return pd.DataFrame(records)

    def rebalance_weights(self, method: str = "performance"):
        """
        Rebalance agent weights based on performance.

        Args:
            method: 'performance' or 'accuracy'
        """
        if len(self.agents) < 2:
            return

        if method == "performance":
            # Weight by recent performance (with softmax)
            perfs = np.array([
                max(a.recent_performance, 0.001)
                for a in self.agents.values()
            ])
        else:  # accuracy
            perfs = np.array([a.accuracy for a in self.agents.values()])

        # Softmax normalization
        exp_perfs = np.exp(perfs - perfs.max())
        weights = exp_perfs / exp_perfs.sum()

        for (name, agent_weight), w in zip(self.agents.items(), weights):
            agent_weight.weight = w

        self.logger.info(f"Rebalanced weights: {dict(zip(self.agents.keys(), weights))}")


class ConstitutionalAggregator(SignalAggregator):
    """
    Aggregator with Constitutional AI-style safety checks.
    Applies rules and self-critique before executing signals.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Safety rules
        self.rules: list[Callable[[AggregatedSignal, dict], tuple[bool, str]]] = []
        self.add_default_rules()

    def add_default_rules(self):
        """Add default safety rules."""

        # Rule 1: Don't trade on very low confidence
        def low_confidence_check(signal, context):
            if signal.confidence < 0.3 and signal.action != Action.HOLD:
                return False, "Confidence too low for action"
            return True, ""

        # Rule 2: Don't trade when agents strongly disagree
        def disagreement_check(signal, context):
            if signal.agreement_score < 0.4 and signal.action != Action.HOLD:
                return False, "Agents strongly disagree"
            return True, ""

        # Rule 3: Position size sanity check
        def position_size_check(signal, context):
            if signal.position_size > 0.5:
                return False, "Position size too large"
            return True, ""

        self.rules.extend([
            low_confidence_check,
            disagreement_check,
            position_size_check,
        ])

    def add_rule(self, rule: Callable[[AggregatedSignal, dict], tuple[bool, str]]):
        """Add a custom safety rule."""
        self.rules.append(rule)

    def aggregate(
        self,
        features: pd.DataFrame,
        symbol: str,
        timestamp: Optional[datetime] = None,
        context: Optional[dict] = None,
    ) -> AggregatedSignal:
        """Aggregate with safety checks."""
        # Get base aggregated signal
        signal = super().aggregate(features, symbol, timestamp)

        # Apply safety rules
        context = context or {}
        for rule in self.rules:
            passed, reason = rule(signal, context)
            if not passed:
                self.logger.warning(f"Safety rule violated: {reason}")
                signal.action = Action.HOLD
                signal.position_size = 0.0
                signal.metadata["safety_override"] = reason
                break

        return signal
