"""
Mixture of Experts (MoE) Coordinator Agent.

Implements a gating mechanism that dynamically selects which expert agents
to use based on current market conditions. This enables:
- Specialized experts for different market regimes
- Sparse activation for efficiency
- Learned routing based on market features
- Automatic expert discovery

Reference: "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer" (2017)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .base import BaseNeuralAgent, BaseAgent, Signal, Action, ALL_FEATURES


@dataclass
class ExpertInfo:
    """Information about a registered expert."""
    agent: BaseAgent
    specialization: str  # e.g., "trending", "volatile", "mean_reversion"
    historical_accuracy: float = 0.5
    recent_performance: float = 0.0
    activation_count: int = 0


class TopKGating(nn.Module):
    """
    Top-K gating mechanism for expert selection.
    Implements noisy top-k gating with load balancing.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
    ):
        """
        Initialize Top-K gating.

        Args:
            input_dim: Dimension of input features
            num_experts: Number of experts to route between
            top_k: Number of experts to activate
            noise_std: Standard deviation of gating noise (for exploration)
        """
        super().__init__()

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.noise_std = noise_std

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, num_experts),
        )

        # Learnable noise for exploration
        self.noise_weight = nn.Parameter(torch.zeros(num_experts))

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gating weights for experts.

        Args:
            x: Input features (batch, input_dim)
            training: Whether in training mode (adds noise)

        Returns:
            gates: Gating weights (batch, num_experts)
            top_k_indices: Indices of top-k experts (batch, top_k)
            top_k_gates: Weights for top-k experts (batch, top_k)
        """
        # Compute raw gate logits
        logits = self.gate(x)

        # Add noise during training for exploration
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            noise = noise * F.softplus(self.noise_weight)
            logits = logits + noise

        # Compute softmax gates
        gates = F.softmax(logits, dim=-1)

        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)

        # Renormalize top-k gates
        top_k_gates = top_k_gates / (top_k_gates.sum(dim=-1, keepdim=True) + 1e-8)

        return gates, top_k_indices, top_k_gates

    def load_balancing_loss(self, gates: torch.Tensor) -> torch.Tensor:
        """
        Compute load balancing loss to encourage even expert usage.

        Args:
            gates: Gating weights (batch, num_experts)

        Returns:
            Load balancing loss scalar
        """
        # Mean gate activation per expert
        expert_usage = gates.mean(dim=0)

        # Target is uniform distribution
        target = torch.ones_like(expert_usage) / self.num_experts

        # MSE loss to encourage uniform usage
        return F.mse_loss(expert_usage, target)


class ExpertNetwork(nn.Module):
    """
    Individual expert network for processing market features.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MixtureOfExpertsModel(nn.Module):
    """
    Full Mixture of Experts model with internal experts.
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_dim: int = 64,
        expert_dim: int = 32,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # Input encoding
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Gating network
        self.gating = TopKGating(
            input_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
        )

        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(hidden_dim, hidden_dim, expert_dim)
            for _ in range(num_experts)
        ])

        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(expert_dim, expert_dim // 2),
            nn.GELU(),
            nn.Linear(expert_dim // 2, 3),  # BUY, SELL, HOLD
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(expert_dim, 1),
            nn.Sigmoid(),
        )

        self.position_head = nn.Sequential(
            nn.Linear(expert_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with expert routing.

        Args:
            x: Input features (batch, seq_len, input_dim) or (batch, input_dim)
            training: Whether in training mode

        Returns:
            Dictionary with predictions and routing info
        """
        # Handle sequence input
        if x.dim() == 3:
            x = x[:, -1, :]  # Use last timestep

        batch_size = x.shape[0]

        # Encode input
        encoded = self.input_encoder(x)

        # Get routing decisions
        gates, top_k_indices, top_k_gates = self.gating(encoded, training)

        # Compute expert outputs
        # For efficiency, we compute all experts but only use top-k
        expert_outputs = torch.stack([
            expert(encoded) for expert in self.experts
        ], dim=1)  # (batch, num_experts, expert_dim)

        # Gather top-k expert outputs
        # Expand indices for gathering
        indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.shape[-1])
        top_k_outputs = torch.gather(expert_outputs, 1, indices_expanded)  # (batch, top_k, expert_dim)

        # Weighted combination of top-k experts
        combined = (top_k_outputs * top_k_gates.unsqueeze(-1)).sum(dim=1)  # (batch, expert_dim)

        # Output heads
        action_logits = self.action_head(combined)
        confidence = self.confidence_head(combined).squeeze(-1)
        position_size = self.position_head(combined).squeeze(-1)

        # Load balancing loss
        lb_loss = self.gating.load_balancing_loss(gates)

        return {
            "action_logits": action_logits,
            "confidence": confidence,
            "position_size": position_size,
            "gates": gates,
            "top_k_indices": top_k_indices,
            "top_k_gates": top_k_gates,
            "load_balancing_loss": lb_loss,
            "expert_outputs": expert_outputs,
        }


class MixtureOfExpertsCoordinator(BaseNeuralAgent):
    """
    Coordinator that routes decisions to specialized expert agents.

    Can work in two modes:
    1. Internal experts: Uses built-in neural network experts
    2. External experts: Coordinates other registered agent instances

    Features:
    - Top-K sparse gating for efficiency
    - Load balancing to use all experts
    - Learns optimal expert routing from market conditions
    - Can combine multiple expert opinions
    """

    FEATURES = ALL_FEATURES

    def __init__(
        self,
        input_dim: Optional[int] = None,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_dim: int = 64,
        use_external_experts: bool = False,
        **kwargs,
    ):
        """
        Initialize MoE Coordinator.

        Args:
            input_dim: Input feature dimension
            num_experts: Number of experts (internal mode)
            top_k: Number of experts to activate per input
            hidden_dim: Hidden dimension
            use_external_experts: Whether to use external agent experts
        """
        input_dim = input_dim or len(self.FEATURES)
        super().__init__(
            name="moe_coordinator",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )

        self.num_experts = num_experts
        self.top_k = top_k
        self.use_external_experts = use_external_experts

        # Internal MoE model
        self.model = MixtureOfExpertsModel(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=hidden_dim,
        ).to(self.device)

        # External experts registry
        self.external_experts: dict[str, ExpertInfo] = {}

        # Gating network for external experts
        if use_external_experts:
            self.external_gating = TopKGating(
                input_dim=hidden_dim,
                num_experts=1,  # Will be updated when experts register
                top_k=top_k,
            ).to(self.device)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Expert specializations (for interpretability)
        self.expert_specializations = [
            "trending_up",
            "trending_down",
            "high_volatility",
            "mean_reversion",
        ][:num_experts]

        self.logger.info(
            f"Initialized MoECoordinator: {num_experts} experts, top_k={top_k}"
        )

    def get_feature_names(self) -> list[str]:
        return self.FEATURES

    def register_expert(
        self,
        agent: BaseAgent,
        specialization: str = "general",
    ):
        """
        Register an external expert agent.

        Args:
            agent: Agent instance to register
            specialization: What market conditions this expert handles
        """
        self.external_experts[agent.name] = ExpertInfo(
            agent=agent,
            specialization=specialization,
        )

        # Update external gating network
        if self.use_external_experts:
            num_external = len(self.external_experts)
            self.external_gating = TopKGating(
                input_dim=self.hidden_dim,
                num_experts=num_external,
                top_k=min(self.top_k, num_external),
            ).to(self.device)

        self.logger.info(f"Registered external expert: {agent.name} ({specialization})")

    def unregister_expert(self, agent_name: str):
        """Remove an external expert."""
        if agent_name in self.external_experts:
            del self.external_experts[agent_name]
            self.logger.info(f"Unregistered expert: {agent_name}")

    def _get_internal_prediction(
        self,
        features: torch.Tensor,
    ) -> tuple[Signal, dict]:
        """Get prediction using internal experts."""
        outputs = self.model(features, training=False)

        # Get action
        action_probs = F.softmax(outputs["action_logits"], dim=-1)
        action_idx = action_probs.argmax(dim=-1).item()
        action = Action(action_idx)

        # Get confidence
        model_conf = outputs["confidence"].item()
        action_prob = action_probs[0, action_idx].item()
        confidence = (model_conf + action_prob) / 2

        # Get position size
        position_size = outputs["position_size"].item() * confidence

        # Expert routing info
        top_experts = outputs["top_k_indices"][0].cpu().numpy().tolist()
        expert_weights = outputs["top_k_gates"][0].cpu().numpy().tolist()

        routing_info = {
            "top_experts": top_experts,
            "expert_weights": expert_weights,
            "expert_names": [
                self.expert_specializations[i] if i < len(self.expert_specializations) else f"expert_{i}"
                for i in top_experts
            ],
            "all_gates": outputs["gates"][0].cpu().numpy().tolist(),
        }

        return action, confidence, position_size, routing_info

    def _get_external_prediction(
        self,
        features: pd.DataFrame | np.ndarray | torch.Tensor,
        symbol: str,
    ) -> tuple[Signal, dict]:
        """Get prediction by routing to external experts."""
        if not self.external_experts:
            # No external experts - fall back to internal
            x = self.preprocess(features)
            return self._get_internal_prediction(x)

        # Encode features for gating
        x = self.preprocess(features)
        if x.dim() == 3:
            x = x[:, -1, :]
        encoded = self.model.input_encoder(x)

        # Get routing weights
        gates, top_k_indices, top_k_gates = self.external_gating(encoded, training=False)

        # Get signals from top-k external experts
        expert_names = list(self.external_experts.keys())
        expert_signals = []
        expert_weights = []

        for i in range(self.external_gating.top_k):
            if i < len(top_k_indices[0]):
                idx = top_k_indices[0, i].item()
                if idx < len(expert_names):
                    expert_name = expert_names[idx]
                    expert_info = self.external_experts[expert_name]

                    # Get signal from expert
                    signal = expert_info.agent.predict(features, symbol)
                    expert_signals.append(signal)
                    expert_weights.append(top_k_gates[0, i].item())

                    # Update activation count
                    expert_info.activation_count += 1

        # Aggregate expert signals
        if expert_signals:
            action, confidence, position_size = self._aggregate_expert_signals(
                expert_signals, expert_weights
            )
        else:
            # Fallback to internal
            action, confidence, position_size, _ = self._get_internal_prediction(x)

        routing_info = {
            "top_experts": top_k_indices[0].cpu().numpy().tolist(),
            "expert_weights": expert_weights,
            "expert_names": [
                expert_names[i] if i < len(expert_names) else f"expert_{i}"
                for i in top_k_indices[0].cpu().numpy().tolist()
            ],
            "all_gates": gates[0].cpu().numpy().tolist(),
            "expert_signals": [
                {"action": s.action.name, "confidence": s.confidence}
                for s in expert_signals
            ],
        }

        return action, confidence, position_size, routing_info

    def _aggregate_expert_signals(
        self,
        signals: list[Signal],
        weights: list[float],
    ) -> tuple[Action, float, float]:
        """Aggregate multiple expert signals using weighted voting."""
        # Weighted vote for action
        action_scores = {Action.HOLD: 0.0, Action.BUY: 0.0, Action.SELL: 0.0}

        total_weight = sum(weights)
        for signal, weight in zip(signals, weights):
            normalized_weight = weight / total_weight if total_weight > 0 else 1.0 / len(weights)
            action_scores[signal.action] += normalized_weight * signal.confidence

        # Select action with highest weighted score
        action = max(action_scores, key=action_scores.get)

        # Weighted average confidence
        confidence = sum(
            s.confidence * w for s, w in zip(signals, weights)
        ) / total_weight if total_weight > 0 else 0.5

        # Weighted average position size
        position_size = sum(
            s.position_size * w for s, w in zip(signals, weights)
        ) / total_weight if total_weight > 0 else 0.0

        return action, confidence, position_size

    def predict(
        self,
        features: pd.DataFrame | np.ndarray | torch.Tensor,
        symbol: str,
    ) -> Signal:
        """
        Generate trading signal using mixture of experts.

        Args:
            features: Market features
            symbol: Asset symbol

        Returns:
            Trading signal from aggregated experts
        """
        self.eval_mode()

        with torch.no_grad():
            if self.use_external_experts and self.external_experts:
                action, confidence, position_size, routing_info = self._get_external_prediction(
                    features, symbol
                )
            else:
                x = self.preprocess(features)
                action, confidence, position_size, routing_info = self._get_internal_prediction(x)

        # Create signal
        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size=position_size,
            timestamp=datetime.now(),
            agent_name=self.name,
            metadata={
                "routing": routing_info,
                "mode": "external" if self.use_external_experts else "internal",
            },
        )

        self.state.step_count += 1
        self.state.last_signal = signal

        return signal

    def get_expert_statistics(self) -> pd.DataFrame:
        """Get statistics about expert usage."""
        if self.use_external_experts and self.external_experts:
            records = []
            for name, info in self.external_experts.items():
                records.append({
                    "name": name,
                    "specialization": info.specialization,
                    "activation_count": info.activation_count,
                    "historical_accuracy": info.historical_accuracy,
                    "recent_performance": info.recent_performance,
                })
            return pd.DataFrame(records)
        else:
            # Return internal expert stats
            records = []
            for i, spec in enumerate(self.expert_specializations):
                records.append({
                    "name": f"expert_{i}",
                    "specialization": spec,
                    "activation_count": 0,  # Would need tracking
                })
            return pd.DataFrame(records)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Training step for MoE model.

        Args:
            batch: Dictionary with 'features', 'actions'

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        features = batch["features"].to(self.device)
        targets = batch["actions"].to(self.device)

        outputs = self.model(features, training=True)

        # Classification loss
        action_loss = self.loss_fn(outputs["action_logits"], targets)

        # Load balancing loss
        lb_loss = outputs["load_balancing_loss"]

        # Total loss
        total_loss = action_loss + 0.01 * lb_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "load_balancing_loss": lb_loss.item(),
        }

    def update_expert_performance(
        self,
        expert_name: str,
        outcome: float,
        decay: float = 0.95,
    ):
        """
        Update an external expert's performance metrics.

        Args:
            expert_name: Name of the expert
            outcome: Outcome of the expert's recent signal (e.g., return)
            decay: Decay factor for exponential moving average
        """
        if expert_name in self.external_experts:
            expert = self.external_experts[expert_name]
            expert.recent_performance = (
                decay * expert.recent_performance + (1 - decay) * outcome
            )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "use_external_experts": self.use_external_experts,
            "expert_specializations": self.expert_specializations,
        })
        return config


if __name__ == "__main__":
    print("Testing Mixture of Experts Coordinator...")
    print("=" * 50)

    # Create agent
    agent = MixtureOfExpertsCoordinator(
        input_dim=20,
        num_experts=4,
        top_k=2,
        hidden_dim=32,
    )

    # Create dummy data
    batch_size = 8
    seq_len = 30
    n_features = 20

    dummy_features = torch.randn(batch_size, seq_len, n_features)

    # Test forward pass
    print("\nTesting forward pass...")
    agent.eval_mode()
    with torch.no_grad():
        outputs = agent.model(dummy_features.to(agent.device))
        print(f"Action logits shape: {outputs['action_logits'].shape}")
        print(f"Gates shape: {outputs['gates'].shape}")
        print(f"Top-k indices: {outputs['top_k_indices'][0].cpu().numpy()}")
        print(f"Top-k gates: {outputs['top_k_gates'][0].cpu().numpy()}")

    # Test signal generation
    print("\nTesting signal generation...")
    signal = agent.predict(dummy_features[0], "TEST")
    print(f"Signal: {signal.action.name}")
    print(f"Confidence: {signal.confidence:.3f}")
    print(f"Routing: {signal.metadata['routing']['expert_names']}")
    print(f"Weights: {signal.metadata['routing']['expert_weights']}")

    # Test training step
    print("\nTesting training step...")
    agent.setup_optimizer()
    batch = {
        "features": dummy_features,
        "actions": torch.randint(0, 3, (batch_size,)),
    }
    metrics = agent.training_step(batch)
    print(f"Training loss: {metrics['loss']:.4f}")
    print(f"Load balancing loss: {metrics['load_balancing_loss']:.4f}")

    # Test external expert mode
    print("\nTesting external expert mode...")

    # Create a mock external expert
    class MockExpert(BaseAgent):
        def __init__(self, name):
            super().__init__(name, input_dim=20)

        def predict(self, features, symbol):
            return Signal(
                symbol=symbol,
                action=Action.BUY,
                confidence=0.7,
                position_size=0.1,
                timestamp=datetime.now(),
                agent_name=self.name,
            )

        def get_feature_names(self):
            return []

    # Create external expert coordinator
    external_agent = MixtureOfExpertsCoordinator(
        input_dim=20,
        num_experts=2,
        top_k=2,
        use_external_experts=True,
    )

    external_agent.register_expert(MockExpert("trend_expert"), "trending")
    external_agent.register_expert(MockExpert("volatility_expert"), "volatile")

    signal = external_agent.predict(dummy_features[0], "TEST")
    print(f"External mode signal: {signal.action.name}")
    print(f"Expert names: {signal.metadata['routing']['expert_names']}")

    print("\nMoE agent tests passed!")
