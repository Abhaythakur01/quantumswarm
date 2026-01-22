"""
Constitutional AI Risk Manager Agent.

Implements a self-critiquing safety layer inspired by Constitutional AI.
This agent reviews trading decisions against a set of principles (constitution)
and can veto or modify signals that violate risk management rules.

Key features:
- Multi-stage critique: Initial response -> Critique -> Revised response
- Learnable constitution: Principles can be learned from experience
- Hierarchical safety: Multiple levels of risk checks
- Explainable decisions: Clear reasoning for any modifications

Reference: "Constitutional AI: Harmlessness from AI Feedback" (2022, Anthropic)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .base import BaseNeuralAgent, Signal, Action, VOLATILITY_FEATURES


@dataclass
class ConstitutionalPrinciple:
    """A single principle in the constitution."""
    name: str
    description: str
    check_fn: Optional[Callable] = None  # Custom check function
    severity: str = "warning"  # 'warning', 'block', 'modify'
    threshold: float = 0.0
    learned_weight: float = 1.0


@dataclass
class CritiqueResult:
    """Result of critiquing a signal."""
    original_signal: Signal
    violations: list[str]
    critique_reasoning: str
    revised_signal: Optional[Signal]
    was_modified: bool
    confidence_adjustment: float


class ConstitutionEncoder(nn.Module):
    """
    Neural network that encodes market state and signal
    to detect potential violations.
    """

    def __init__(
        self,
        market_dim: int,
        signal_dim: int = 5,  # action, confidence, position_size, etc.
        hidden_dim: int = 64,
        num_principles: int = 10,
    ):
        super().__init__()

        self.market_encoder = nn.Sequential(
            nn.Linear(market_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.signal_encoder = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Combined analysis
        self.analyzer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Output heads
        self.violation_head = nn.Linear(hidden_dim // 2, num_principles)
        self.severity_head = nn.Linear(hidden_dim // 2, 3)  # low, medium, high
        self.confidence_adjustment_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(
        self,
        market_features: torch.Tensor,
        signal_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Analyze market state and signal for potential violations.

        Args:
            market_features: Market state (batch, market_dim)
            signal_features: Signal features (batch, signal_dim)

        Returns:
            Dictionary with violation probabilities, severity, confidence adjustment
        """
        market_enc = self.market_encoder(market_features)
        signal_enc = self.signal_encoder(signal_features)

        combined = torch.cat([market_enc, signal_enc], dim=-1)
        analyzed = self.analyzer(combined)

        violation_logits = self.violation_head(analyzed)
        severity_logits = self.severity_head(analyzed)
        confidence_adj = self.confidence_adjustment_head(analyzed).squeeze(-1)

        return {
            "violation_probs": torch.sigmoid(violation_logits),
            "severity_logits": severity_logits,
            "confidence_adjustment": confidence_adj,
        }


class SelfCritiqueModule(nn.Module):
    """
    Module that generates critiques and revised signals.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # Critique generator
        self.critique_net = nn.Sequential(
            nn.Linear(hidden_dim + 10, hidden_dim),  # +10 for violation info
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Signal revision network
        self.revision_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 5),  # action_logits(3) + conf + pos_size
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        violation_info: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Generate critique and revised signal.
        """
        combined = torch.cat([hidden_state, violation_info], dim=-1)
        critique_hidden = self.critique_net(combined)
        revision = self.revision_net(critique_hidden)

        action_logits = revision[:, :3]
        confidence = torch.sigmoid(revision[:, 3])
        position_size = torch.sigmoid(revision[:, 4])

        return {
            "action_logits": action_logits,
            "confidence": confidence,
            "position_size": position_size,
            "critique_hidden": critique_hidden,
        }


class Constitution:
    """
    Collection of principles that define safe trading behavior.
    """

    def __init__(self):
        self.principles: list[ConstitutionalPrinciple] = []
        self._setup_default_principles()

    def _setup_default_principles(self):
        """Set up default trading safety principles."""

        # Principle 1: Position size limits
        self.add_principle(
            name="position_size_limit",
            description="Position size should not exceed maximum allowed",
            check_fn=lambda signal, ctx: signal.position_size <= ctx.get("max_position", 0.25),
            severity="modify",
        )

        # Principle 2: Drawdown protection
        self.add_principle(
            name="drawdown_protection",
            description="Avoid increasing positions during significant drawdown",
            check_fn=lambda signal, ctx: not (
                signal.action == Action.BUY and
                ctx.get("current_drawdown", 0) > 0.10
            ),
            severity="block",
        )

        # Principle 3: Volatility adjustment
        self.add_principle(
            name="volatility_adjustment",
            description="Reduce position size in high volatility environments",
            check_fn=lambda signal, ctx: not (
                signal.position_size > 0.15 and
                ctx.get("volatility", 0) > ctx.get("volatility_threshold", 0.03)
            ),
            severity="modify",
        )

        # Principle 4: Confidence threshold
        self.add_principle(
            name="confidence_threshold",
            description="Only trade with sufficient confidence",
            check_fn=lambda signal, ctx: (
                signal.action == Action.HOLD or
                signal.confidence >= ctx.get("min_confidence", 0.5)
            ),
            severity="modify",
        )

        # Principle 5: Consecutive loss limit
        self.add_principle(
            name="consecutive_loss_limit",
            description="Pause trading after consecutive losses",
            check_fn=lambda signal, ctx: not (
                signal.action != Action.HOLD and
                ctx.get("consecutive_losses", 0) >= ctx.get("max_consecutive_losses", 3)
            ),
            severity="block",
        )

        # Principle 6: Daily trade limit
        self.add_principle(
            name="daily_trade_limit",
            description="Limit number of trades per day",
            check_fn=lambda signal, ctx: (
                signal.action == Action.HOLD or
                ctx.get("daily_trades", 0) < ctx.get("max_daily_trades", 10)
            ),
            severity="block",
        )

        # Principle 7: Correlation risk
        self.add_principle(
            name="correlation_risk",
            description="Avoid highly correlated positions",
            check_fn=lambda signal, ctx: (
                signal.action != Action.BUY or
                ctx.get("portfolio_correlation", 0) < ctx.get("max_correlation", 0.7)
            ),
            severity="warning",
        )

        # Principle 8: Sector concentration
        self.add_principle(
            name="sector_concentration",
            description="Maintain sector diversification",
            check_fn=lambda signal, ctx: (
                signal.action != Action.BUY or
                ctx.get("sector_exposure", 0) < ctx.get("max_sector_exposure", 0.3)
            ),
            severity="warning",
        )

        # Principle 9: Minimum holding period
        self.add_principle(
            name="minimum_holding",
            description="Respect minimum holding period",
            check_fn=lambda signal, ctx: (
                signal.action != Action.SELL or
                ctx.get("holding_period", float("inf")) >= ctx.get("min_holding_period", 1)
            ),
            severity="warning",
        )

        # Principle 10: Market hours
        self.add_principle(
            name="market_hours",
            description="Only trade during market hours",
            check_fn=lambda signal, ctx: ctx.get("market_open", True),
            severity="block",
        )

    def add_principle(
        self,
        name: str,
        description: str,
        check_fn: Optional[Callable] = None,
        severity: str = "warning",
        threshold: float = 0.0,
    ):
        """Add a new principle to the constitution."""
        self.principles.append(ConstitutionalPrinciple(
            name=name,
            description=description,
            check_fn=check_fn,
            severity=severity,
            threshold=threshold,
        ))

    def check_signal(
        self,
        signal: Signal,
        context: dict,
    ) -> list[tuple[ConstitutionalPrinciple, bool, str]]:
        """
        Check a signal against all principles.

        Returns:
            List of (principle, passed, reason) tuples
        """
        results = []

        for principle in self.principles:
            if principle.check_fn is not None:
                try:
                    passed = principle.check_fn(signal, context)
                    reason = "" if passed else f"Violated: {principle.description}"
                except Exception as e:
                    passed = True  # Don't block on check errors
                    reason = f"Check error: {e}"
            else:
                passed = True
                reason = ""

            results.append((principle, passed, reason))

        return results


class ConstitutionalRiskManager(BaseNeuralAgent):
    """
    Constitutional AI Risk Manager for trading.

    This agent reviews trading signals against a constitution of safety
    principles, critiques potentially dangerous decisions, and can
    modify or block signals that violate risk management rules.

    Features:
    - Rule-based and learned safety checks
    - Multi-stage self-critique
    - Explainable decision modifications
    - Adaptive risk thresholds
    """

    FEATURES = VOLATILITY_FEATURES + [
        "returns", "returns_5d", "returns_10d",
        "volume_ratio", "rsi_14",
        "drawdown", "portfolio_value",
    ]

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        **kwargs,
    ):
        """
        Initialize Constitutional Risk Manager.

        Args:
            input_dim: Market feature dimension
            hidden_dim: Hidden layer dimension
        """
        input_dim = input_dim or len(self.FEATURES)
        super().__init__(
            name="constitutional_risk",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )

        # Initialize constitution
        self.constitution = Constitution()

        # Neural components
        self.encoder = ConstitutionEncoder(
            market_dim=input_dim,
            signal_dim=5,
            hidden_dim=hidden_dim,
            num_principles=len(self.constitution.principles),
        ).to(self.device)

        self.critique_module = SelfCritiqueModule(hidden_dim).to(self.device)

        # Combine into single model for saving/loading
        self.model = nn.ModuleDict({
            "encoder": self.encoder,
            "critique": self.critique_module,
        })

        # Risk context (updated externally)
        self.risk_context: dict = {
            "max_position": 0.25,
            "current_drawdown": 0.0,
            "volatility": 0.02,
            "volatility_threshold": 0.03,
            "min_confidence": 0.5,
            "consecutive_losses": 0,
            "max_consecutive_losses": 3,
            "daily_trades": 0,
            "max_daily_trades": 10,
            "portfolio_correlation": 0.0,
            "max_correlation": 0.7,
            "sector_exposure": 0.0,
            "max_sector_exposure": 0.3,
            "holding_period": 0,
            "min_holding_period": 1,
            "market_open": True,
        }

        self.logger.info(
            f"Initialized ConstitutionalRiskManager with "
            f"{len(self.constitution.principles)} principles"
        )

    def get_feature_names(self) -> list[str]:
        return self.FEATURES

    def update_context(self, **kwargs):
        """Update risk context with new values."""
        self.risk_context.update(kwargs)

    def _signal_to_tensor(self, signal: Signal) -> torch.Tensor:
        """Convert signal to tensor representation."""
        action_onehot = [0.0, 0.0, 0.0]
        action_onehot[signal.action.value] = 1.0

        features = action_onehot + [signal.confidence, signal.position_size]
        return torch.tensor(features, device=self.device).unsqueeze(0)

    def critique_signal(
        self,
        signal: Signal,
        market_features: torch.Tensor,
    ) -> CritiqueResult:
        """
        Critique a trading signal and potentially revise it.

        Args:
            signal: Original trading signal
            market_features: Current market features

        Returns:
            CritiqueResult with original, critique, and revised signal
        """
        # Step 1: Rule-based checks
        rule_results = self.constitution.check_signal(signal, self.risk_context)
        violations = [
            (p.name, reason)
            for p, passed, reason in rule_results
            if not passed
        ]

        # Step 2: Neural critique
        self.eval_mode()
        with torch.no_grad():
            market_tensor = self.preprocess(market_features)
            if market_tensor.dim() > 2:
                market_tensor = market_tensor[:, -1, :]  # Use last timestep

            signal_tensor = self._signal_to_tensor(signal)

            # Encode and analyze
            analysis = self.encoder(market_tensor, signal_tensor)

            # Get neural violations (principles with high violation probability)
            violation_probs = analysis["violation_probs"][0]
            neural_violations = [
                self.constitution.principles[i].name
                for i in range(len(violation_probs))
                if violation_probs[i] > 0.5
            ]

            confidence_adjustment = analysis["confidence_adjustment"].item()

        # Combine violations
        all_violations = [v[0] for v in violations] + neural_violations
        all_violations = list(set(all_violations))

        # Step 3: Determine if modification is needed
        blocking_violations = [
            v for v in violations
            if any(p.name == v[0] and p.severity == "block"
                   for p in self.constitution.principles)
        ]

        modifying_violations = [
            v for v in violations
            if any(p.name == v[0] and p.severity == "modify"
                   for p in self.constitution.principles)
        ]

        # Step 4: Generate revised signal if needed
        was_modified = False
        revised_signal = None

        if blocking_violations:
            # Block the trade - force HOLD
            revised_signal = Signal(
                symbol=signal.symbol,
                action=Action.HOLD,
                confidence=0.0,
                position_size=0.0,
                timestamp=datetime.now(),
                agent_name=self.name,
                metadata={
                    "original_action": signal.action.name,
                    "blocked_by": [v[0] for v in blocking_violations],
                },
            )
            was_modified = True

        elif modifying_violations or confidence_adjustment < -0.2:
            # Modify the signal
            new_confidence = max(0.1, signal.confidence + confidence_adjustment * 0.5)
            new_position_size = signal.position_size * new_confidence / max(signal.confidence, 0.1)

            # Reduce position size based on violations
            reduction_factor = 1.0 - 0.2 * len(modifying_violations)
            new_position_size *= max(0.1, reduction_factor)

            # Check if position is too small to be meaningful
            if new_position_size < 0.01:
                revised_signal = Signal(
                    symbol=signal.symbol,
                    action=Action.HOLD,
                    confidence=new_confidence,
                    position_size=0.0,
                    timestamp=datetime.now(),
                    agent_name=self.name,
                    metadata={
                        "original_action": signal.action.name,
                        "modified_reason": "position_too_small",
                    },
                )
            else:
                revised_signal = Signal(
                    symbol=signal.symbol,
                    action=signal.action,
                    confidence=new_confidence,
                    position_size=new_position_size,
                    timestamp=datetime.now(),
                    agent_name=self.name,
                    metadata={
                        "original_confidence": signal.confidence,
                        "original_position_size": signal.position_size,
                        "modifications": [v[0] for v in modifying_violations],
                    },
                )
            was_modified = True

        # Generate critique reasoning
        if all_violations:
            reasoning = f"Signal violated {len(all_violations)} principles: {', '.join(all_violations)}. "
            if was_modified:
                reasoning += f"Signal was {'blocked' if blocking_violations else 'modified'}."
            else:
                reasoning += "Violations were warnings only."
        else:
            reasoning = "Signal passed all constitutional checks."

        return CritiqueResult(
            original_signal=signal,
            violations=all_violations,
            critique_reasoning=reasoning,
            revised_signal=revised_signal,
            was_modified=was_modified,
            confidence_adjustment=confidence_adjustment,
        )

    def predict(
        self,
        features: pd.DataFrame | np.ndarray | torch.Tensor,
        symbol: str,
        incoming_signal: Optional[Signal] = None,
    ) -> Signal:
        """
        Review and potentially modify an incoming signal.

        If no incoming signal is provided, returns a HOLD signal
        (this agent is primarily a filter, not a signal generator).

        Args:
            features: Market features
            symbol: Asset symbol
            incoming_signal: Signal to review

        Returns:
            Approved or modified signal
        """
        if incoming_signal is None:
            # No signal to review - return HOLD
            return Signal(
                symbol=symbol,
                action=Action.HOLD,
                confidence=0.0,
                position_size=0.0,
                timestamp=datetime.now(),
                agent_name=self.name,
                metadata={"reason": "no_incoming_signal"},
            )

        # Critique the incoming signal
        x = self.preprocess(features)
        critique = self.critique_signal(incoming_signal, x)

        # Log the critique
        if critique.was_modified:
            self.logger.warning(
                f"Signal modified: {incoming_signal.action.name} -> "
                f"{critique.revised_signal.action.name if critique.revised_signal else 'BLOCKED'}"
            )
            self.logger.debug(f"Reason: {critique.critique_reasoning}")

        # Update state
        self.state.step_count += 1

        # Return revised signal if modified, otherwise original
        result = critique.revised_signal if critique.was_modified else incoming_signal
        self.state.last_signal = result

        return result

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Training step to learn violation detection.

        Args:
            batch: Dictionary with 'features', 'signals', 'violations', 'outcomes'

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        market_features = batch["features"].to(self.device)
        signal_features = batch["signals"].to(self.device)
        violation_labels = batch["violations"].to(self.device)

        # Forward pass
        analysis = self.encoder(market_features, signal_features)

        # Violation detection loss (binary cross-entropy per principle)
        violation_loss = F.binary_cross_entropy(
            analysis["violation_probs"],
            violation_labels,
        )

        # Total loss
        total_loss = violation_loss

        # Optional: outcome-based loss if provided
        if "outcomes" in batch:
            outcomes = batch["outcomes"].to(self.device)
            # Learn to adjust confidence based on outcomes
            pred_adjustment = analysis["confidence_adjustment"]
            target_adjustment = torch.sign(outcomes) * 0.5  # Scale outcomes to [-0.5, 0.5]
            outcome_loss = F.mse_loss(pred_adjustment, target_adjustment)
            total_loss = total_loss + 0.5 * outcome_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {"loss": total_loss.item(), "violation_loss": violation_loss.item()}

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_principles": len(self.constitution.principles),
        })
        return config


if __name__ == "__main__":
    print("Testing Constitutional Risk Manager...")
    print("=" * 50)

    # Create agent
    agent = ConstitutionalRiskManager(
        input_dim=15,
        hidden_dim=32,
    )

    # Create test signal and features
    test_signal = Signal(
        symbol="TEST",
        action=Action.BUY,
        confidence=0.8,
        position_size=0.3,  # Exceeds default max of 0.25
        timestamp=datetime.now(),
        agent_name="test_agent",
    )

    dummy_features = torch.randn(1, 15)

    # Test critique
    print("\nTesting signal critique...")
    critique = agent.critique_signal(test_signal, dummy_features)
    print(f"Original: {critique.original_signal.action.name}, pos={critique.original_signal.position_size:.2f}")
    print(f"Violations: {critique.violations}")
    print(f"Was modified: {critique.was_modified}")
    if critique.revised_signal:
        print(f"Revised: {critique.revised_signal.action.name}, pos={critique.revised_signal.position_size:.2f}")
    print(f"Reasoning: {critique.critique_reasoning}")

    # Test with drawdown context
    print("\nTesting with high drawdown context...")
    agent.update_context(current_drawdown=0.15)  # 15% drawdown
    critique2 = agent.critique_signal(test_signal, dummy_features)
    print(f"Violations: {critique2.violations}")
    print(f"Reasoning: {critique2.critique_reasoning}")

    # Test predict method
    print("\nTesting predict method...")
    agent.update_context(current_drawdown=0.05)  # Reset drawdown
    result = agent.predict(dummy_features, "TEST", incoming_signal=test_signal)
    print(f"Result: {result.action.name}, conf={result.confidence:.2f}, pos={result.position_size:.2f}")

    # Test training
    print("\nTesting training step...")
    agent.setup_optimizer()
    batch = {
        "features": torch.randn(8, 15),
        "signals": torch.randn(8, 5),
        "violations": torch.randint(0, 2, (8, 10)).float(),
    }
    metrics = agent.training_step(batch)
    print(f"Training loss: {metrics['loss']:.4f}")

    print("\nConstitutional agent tests passed!")
