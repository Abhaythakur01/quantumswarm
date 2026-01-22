"""
Mamba SSM (State Space Model) Trading Agent.

Mamba is a selective state space model that achieves linear-time sequence modeling,
making it 5x faster than Transformers while maintaining comparable performance.

Key features:
- Selective state spaces: Input-dependent parameters
- Hardware-aware algorithm: Optimized for GPU memory
- Linear complexity: O(n) vs O(n²) for attention
"""

from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .base import BaseNeuralAgent, Signal, Action, CORE_FEATURES, TREND_FEATURES


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model block.
    Implements the core Mamba architecture with input-dependent dynamics.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize Selective SSM.

        Args:
            d_model: Model dimension
            d_state: State dimension (N in paper)
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # SSM parameters projection
        # Projects to: delta, B, C (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Delta (discretization step) projection
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Initialize A (state matrix) - learned log values for stability
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.A_log = nn.Parameter(torch.log(A.repeat(self.d_inner, 1)))

        # D (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through selective SSM.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection -> (batch, seq_len, 2 * d_inner)
        x_and_res = self.in_proj(x)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        # Convolution for local context
        x = x.transpose(1, 2)  # (batch, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # Causal conv
        x = x.transpose(1, 2)  # (batch, seq_len, d_inner)

        x = F.silu(x)

        # Compute SSM parameters (input-dependent)
        x_proj = self.x_proj(x)  # (batch, seq_len, d_state * 2 + 1)
        delta, B, C = x_proj.split([1, self.d_state, self.d_state], dim=-1)

        # Delta transformation
        delta = F.softplus(self.dt_proj(delta))  # (batch, seq_len, d_inner)

        # Get A from log parameterization
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan (simplified version - full version uses custom CUDA kernel)
        y = self._selective_scan(x, delta, A, B, C)

        # Apply skip connection
        y = y * F.silu(res)

        # Output projection
        y = self.out_proj(y)
        y = self.dropout(y)

        return y

    def _selective_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simplified selective scan implementation.
        In production, this would use the optimized CUDA kernel.

        Args:
            x: Input (batch, seq_len, d_inner)
            delta: Discretization step (batch, seq_len, d_inner)
            A: State matrix (d_inner, d_state)
            B: Input matrix (batch, seq_len, d_state)
            C: Output matrix (batch, seq_len, d_state)

        Returns:
            Output (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize A and B
        # deltaA = exp(delta * A)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (batch, seq_len, d_inner, d_state)
        # deltaB = delta * B
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq_len, d_inner, d_state)

        # Initialize state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            # State update: h = A * h + B * x
            h = deltaA[:, t] * h + deltaB[:, t] * x[:, t, :, None]
            # Output: y = C * h
            y = (h * C[:, t, None, :]).sum(-1)  # (batch, d_inner)
            outputs.append(y)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)

        # Add skip connection with D
        y = y + x * self.D

        return y


class MambaBlock(nn.Module):
    """Mamba block with residual connection and normalization."""

    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SelectiveSSM(d_model, d_state, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))


class MambaModel(nn.Module):
    """
    Full Mamba model for time series prediction.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 4,
        dropout: float = 0.1,
        num_classes: int = 3,  # BUY, SELL, HOLD
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])

        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Confidence head (separate from action)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Position sizing head
        self.position_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch, seq_len, input_dim)
            return_hidden: Whether to return hidden states

        Returns:
            Dictionary with action_logits, confidence, position_size
        """
        # Project input
        h = self.input_proj(x)

        # Apply Mamba layers
        for layer in self.layers:
            h = layer(h)

        # Use last timestep for prediction
        h = self.norm(h[:, -1, :])

        # Compute outputs
        action_logits = self.output_head(h)
        confidence = self.confidence_head(h).squeeze(-1)
        position_size = self.position_head(h).squeeze(-1)

        outputs = {
            "action_logits": action_logits,
            "confidence": confidence,
            "position_size": position_size,
        }

        if return_hidden:
            outputs["hidden"] = h

        return outputs


class MambaAgent(BaseNeuralAgent):
    """
    Trading agent using Mamba State Space Model.

    Features:
    - 5x faster inference than Transformer-based agents
    - Efficient handling of long sequences
    - Input-dependent state dynamics for market regime adaptation
    """

    # Features this agent uses
    FEATURES = CORE_FEATURES + TREND_FEATURES + [
        "volatility_10d", "volatility_ratio",
        "volume_ratio", "gap",
    ]

    def __init__(
        self,
        input_dim: Optional[int] = None,
        d_model: int = 128,
        d_state: int = 16,
        n_layers: int = 4,
        sequence_length: int = 60,
        **kwargs,
    ):
        """
        Initialize Mamba trading agent.

        Args:
            input_dim: Input feature dimension (auto-detected if None)
            d_model: Model hidden dimension
            d_state: State space dimension
            n_layers: Number of Mamba blocks
            sequence_length: Length of input sequences
        """
        input_dim = input_dim or len(self.FEATURES)
        super().__init__(
            name="mamba_ssm",
            input_dim=input_dim,
            hidden_dim=d_model,
            num_layers=n_layers,
            **kwargs,
        )

        self.d_model = d_model
        self.d_state = d_state
        self.sequence_length = sequence_length

        # Initialize model
        self.model = MambaModel(
            input_dim=input_dim,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        self.logger.info(
            f"Initialized MambaAgent: d_model={d_model}, d_state={d_state}, "
            f"n_layers={n_layers}, seq_len={sequence_length}"
        )

    def get_feature_names(self) -> list[str]:
        """Get list of required features."""
        return self.FEATURES

    def predict(
        self,
        features: pd.DataFrame | np.ndarray | torch.Tensor,
        symbol: str,
    ) -> Signal:
        """
        Generate trading signal from market features.

        Args:
            features: Market features (should be seq_len x n_features)
            symbol: Asset symbol

        Returns:
            Trading signal
        """
        self.eval_mode()

        with torch.no_grad():
            # Preprocess features
            x = self.preprocess(features)

            # Ensure sequence dimension
            if x.dim() == 2:
                # (batch, features) -> (batch, seq_len, features)
                if x.shape[0] >= self.sequence_length:
                    x = x[-self.sequence_length:].unsqueeze(0)
                else:
                    # Pad if sequence too short
                    padding = torch.zeros(
                        self.sequence_length - x.shape[0],
                        x.shape[1],
                        device=self.device,
                    )
                    x = torch.cat([padding, x], dim=0).unsqueeze(0)
            elif x.dim() == 3 and x.shape[1] < self.sequence_length:
                # Pad sequence
                padding = torch.zeros(
                    x.shape[0],
                    self.sequence_length - x.shape[1],
                    x.shape[2],
                    device=self.device,
                )
                x = torch.cat([padding, x], dim=1)

            # Forward pass
            outputs = self.model(x)

            # Get action
            action_probs = F.softmax(outputs["action_logits"], dim=-1)
            action_idx = action_probs.argmax(dim=-1).item()
            action = Action(action_idx)

            # Get confidence (combination of model confidence and action probability)
            model_conf = outputs["confidence"].item()
            action_prob = action_probs[0, action_idx].item()
            confidence = (model_conf + action_prob) / 2

            # Get position size
            position_size = outputs["position_size"].item()

            # Scale position size by confidence
            position_size = position_size * confidence

        # Update state
        self.state.step_count += 1

        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size=position_size,
            timestamp=datetime.now(),
            agent_name=self.name,
            metadata={
                "action_probs": action_probs[0].cpu().numpy().tolist(),
                "model_confidence": model_conf,
            },
        )

        self.state.last_signal = signal
        return signal

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Single training step.

        Args:
            batch: Dictionary with 'features', 'actions', 'returns'

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        features = batch["features"].to(self.device)
        targets = batch["actions"].to(self.device)

        # Forward pass
        outputs = self.model(features)

        # Classification loss
        action_loss = self.loss_fn(outputs["action_logits"], targets)

        # Optional: Add auxiliary losses if returns are provided
        total_loss = action_loss

        if "returns" in batch:
            returns = batch["returns"].to(self.device)
            # Encourage higher confidence on correct predictions
            probs = F.softmax(outputs["action_logits"], dim=-1)
            correct_probs = probs.gather(1, targets.unsqueeze(1)).squeeze()
            confidence_loss = F.mse_loss(
                outputs["confidence"],
                correct_probs.detach(),
            )
            total_loss = total_loss + 0.1 * confidence_loss

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
        }

    def get_config(self) -> dict:
        """Get agent configuration."""
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_state": self.d_state,
            "sequence_length": self.sequence_length,
        })
        return config


if __name__ == "__main__":
    # Test the Mamba agent
    print("Testing Mamba SSM Agent...")
    print("=" * 50)

    # Create agent
    agent = MambaAgent(
        input_dim=20,
        d_model=64,
        d_state=8,
        n_layers=2,
        sequence_length=30,
    )

    # Create dummy data
    batch_size = 4
    seq_len = 30
    n_features = 20

    dummy_features = torch.randn(batch_size, seq_len, n_features)

    # Test forward pass
    print("\nTesting forward pass...")
    agent.eval_mode()
    with torch.no_grad():
        outputs = agent.model(dummy_features.to(agent.device))
        print(f"Action logits shape: {outputs['action_logits'].shape}")
        print(f"Confidence shape: {outputs['confidence'].shape}")
        print(f"Position size shape: {outputs['position_size'].shape}")

    # Test signal generation
    print("\nTesting signal generation...")
    signal = agent.predict(dummy_features[0], "TEST")
    print(f"Signal: {signal.action.name}")
    print(f"Confidence: {signal.confidence:.3f}")
    print(f"Position size: {signal.position_size:.3f}")

    # Test training step
    print("\nTesting training step...")
    agent.setup_optimizer()
    batch = {
        "features": dummy_features,
        "actions": torch.randint(0, 3, (batch_size,)),
        "returns": torch.randn(batch_size),
    }
    metrics = agent.training_step(batch)
    print(f"Training loss: {metrics['loss']:.4f}")

    print("\nMamba agent tests passed!")
