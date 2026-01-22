"""
Liquid Neural Network Trading Agent.

Liquid Neural Networks (LNNs) use Liquid Time-Constant (LTC) cells that
dynamically adjust their time constants based on input. This makes them
ideal for trading where market regimes change over time.

Key features:
- Adaptive time constants: Responds to volatility changes
- Continuous-time dynamics: Natural fit for financial time series
- Compact representation: Fewer parameters than LSTMs
- Causally structured: Better for time series forecasting

Reference: "Liquid Time-constant Networks" (2021, MIT)
"""

from datetime import datetime
from typing import Optional, Tuple
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .base import BaseNeuralAgent, Signal, Action, CORE_FEATURES, VOLATILITY_FEATURES


class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Cell.

    Implements continuous-time neural dynamics with input-dependent time constants:
        τ * dx/dt = -x + f(x, I, θ)

    where τ is learned and depends on the input.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        ode_unfolds: int = 6,
    ):
        """
        Initialize LTC cell.

        Args:
            input_size: Dimension of input
            hidden_size: Dimension of hidden state
            ode_unfolds: Number of ODE solver steps per forward pass
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds

        # Sensory (input) weights
        self.sensory_w = nn.Linear(input_size, hidden_size)
        self.sensory_erev = nn.Linear(input_size, hidden_size)

        # Recurrent weights
        self.recurrent_w = nn.Linear(hidden_size, hidden_size)
        self.recurrent_erev = nn.Linear(hidden_size, hidden_size)

        # Time constant parameters (input-dependent)
        self.tau_system = nn.Linear(input_size + hidden_size, hidden_size)

        # Membrane capacitance (learnable)
        self.cm = nn.Parameter(torch.ones(hidden_size))

        # Leak conductance
        self.gleak = nn.Parameter(torch.ones(hidden_size) * 0.1)
        self.vleak = nn.Parameter(torch.zeros(hidden_size))

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        time_delta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LTC cell.

        Args:
            x: Input tensor (batch, input_size)
            h: Previous hidden state (batch, hidden_size)
            time_delta: Time step size

        Returns:
            output: Cell output (batch, hidden_size)
            new_h: New hidden state (batch, hidden_size)
        """
        batch_size = x.shape[0]

        # Initialize hidden state if needed
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        # Compute input-dependent time constant
        tau_input = torch.cat([x, h], dim=-1)
        tau = torch.abs(self.tau_system(tau_input)) + 0.01  # Ensure positive

        # ODE integration using Euler method with multiple unfolds
        dt = time_delta / self.ode_unfolds

        for _ in range(self.ode_unfolds):
            # Sensory input current
            sensory_activation = self.sigmoid(self.sensory_w(x))
            sensory_erev = self.sensory_erev(x)
            i_sensory = sensory_activation * (sensory_erev - h)

            # Recurrent current
            recurrent_activation = self.sigmoid(self.recurrent_w(h))
            recurrent_erev = self.recurrent_erev(h)
            i_recurrent = recurrent_activation * (recurrent_erev - h)

            # Leak current
            i_leak = self.gleak * (self.vleak - h)

            # Total current
            i_total = i_sensory + i_recurrent + i_leak

            # Update state: cm * dh/dt = i_total, with time constant tau
            # dh = (i_total / cm) * (dt / tau)
            dh = (i_total / (self.cm + 0.01)) * (dt / tau)
            h = h + dh

        # Output is the hidden state passed through tanh
        output = self.tanh(h)

        return output, h


class LiquidNetwork(nn.Module):
    """
    Full Liquid Neural Network for sequence processing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """
        Initialize Liquid Network.

        Args:
            input_size: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of stacked LTC layers
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional processing
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Build LTC layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            if bidirectional and i > 0:
                layer_input_size = hidden_size * 2
            self.layers.append(LTCCell(layer_input_size, hidden_size))

        # Reverse layers for bidirectional
        if bidirectional:
            self.reverse_layers = nn.ModuleList()
            for i in range(num_layers):
                layer_input_size = input_size if i == 0 else hidden_size * 2
                self.reverse_layers.append(LTCCell(layer_input_size, hidden_size))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[list[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through liquid network.

        Args:
            x: Input sequence (batch, seq_len, input_size)
            hidden: Initial hidden states for each layer

        Returns:
            output: Output sequence (batch, seq_len, hidden_size * (2 if bidirectional else 1))
            hidden: Final hidden states for each layer
        """
        batch_size, seq_len, _ = x.shape

        # Initialize hidden states
        if hidden is None:
            hidden = [None] * self.num_layers

        # Process sequence
        outputs = []
        h_states = hidden

        for t in range(seq_len):
            layer_input = x[:, t, :]

            new_h_states = []
            for i, (layer, h) in enumerate(zip(self.layers, h_states)):
                output, new_h = layer(layer_input, h)
                output = self.dropout(output)
                layer_input = output
                new_h_states.append(new_h)

            h_states = new_h_states
            outputs.append(layer_input)

        output = torch.stack(outputs, dim=1)

        # Bidirectional processing
        if self.bidirectional:
            reverse_outputs = []
            reverse_h_states = [None] * self.num_layers

            for t in range(seq_len - 1, -1, -1):
                layer_input = x[:, t, :]

                new_h_states = []
                for i, (layer, h) in enumerate(zip(self.reverse_layers, reverse_h_states)):
                    if i > 0:
                        # Concatenate with forward output
                        layer_input = torch.cat([layer_input, outputs[t]], dim=-1)
                    rev_output, new_h = layer(layer_input, h)
                    rev_output = self.dropout(rev_output)
                    layer_input = rev_output
                    new_h_states.append(new_h)

                reverse_h_states = new_h_states
                reverse_outputs.insert(0, layer_input)

            reverse_output = torch.stack(reverse_outputs, dim=1)
            output = torch.cat([output, reverse_output], dim=-1)

        return output, h_states


class LiquidTradingModel(nn.Module):
    """
    Liquid Neural Network model for trading signal prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 3,
    ):
        super().__init__()

        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)

        # Liquid network backbone
        self.liquid = LiquidNetwork(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Regime detection head (auxiliary task)
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 4),  # 4 regimes: trending up/down, ranging, volatile
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[list[torch.Tensor]] = None,
        return_hidden: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch, seq_len, input_dim)
            hidden: Previous hidden states
            return_hidden: Whether to return hidden states

        Returns:
            Dictionary with predictions
        """
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Normalize input
        x = self.input_norm(x)

        # Process through liquid network
        output, new_hidden = self.liquid(x, hidden)

        # Use last timestep for predictions
        h = output[:, -1, :]

        # Compute outputs
        action_logits = self.action_head(h)
        confidence = self.confidence_head(h).squeeze(-1)
        position_size = self.position_head(h).squeeze(-1)
        regime_logits = self.regime_head(h)

        outputs = {
            "action_logits": action_logits,
            "confidence": confidence,
            "position_size": position_size,
            "regime_logits": regime_logits,
        }

        if return_hidden:
            outputs["hidden"] = new_hidden
            outputs["last_hidden"] = h

        return outputs


class LiquidAgent(BaseNeuralAgent):
    """
    Trading agent using Liquid Neural Networks.

    Features:
    - Adaptive time constants for market regime changes
    - Continuous-time dynamics for natural time series modeling
    - Regime detection for market state awareness
    - Maintains hidden state across predictions for temporal coherence
    """

    FEATURES = CORE_FEATURES + VOLATILITY_FEATURES + [
        "returns_5d", "returns_10d", "returns_20d",
        "sma_20_50_cross", "sma_50_200_cross",
        "price_sma_20_ratio", "price_sma_50_ratio",
        "momentum_10", "momentum_20",
    ]

    REGIME_NAMES = ["Trending Up", "Trending Down", "Ranging", "Volatile"]

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        num_layers: int = 2,
        sequence_length: int = 30,
        **kwargs,
    ):
        """
        Initialize Liquid Neural Network agent.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            num_layers: Number of LTC layers
            sequence_length: Input sequence length
        """
        input_dim = input_dim or len(self.FEATURES)
        super().__init__(
            name="liquid_nn",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            **kwargs,
        )

        self.sequence_length = sequence_length

        # Initialize model
        self.model = LiquidTradingModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Loss functions
        self.action_loss_fn = nn.CrossEntropyLoss()
        self.regime_loss_fn = nn.CrossEntropyLoss()

        # Persistent hidden state for online inference
        self._hidden_state: Optional[list[torch.Tensor]] = None

        self.logger.info(
            f"Initialized LiquidAgent: hidden_dim={hidden_dim}, "
            f"num_layers={num_layers}, seq_len={sequence_length}"
        )

    def get_feature_names(self) -> list[str]:
        """Get list of required features."""
        return self.FEATURES

    def reset(self):
        """Reset agent state including hidden state."""
        super().reset()
        self._hidden_state = None
        self.logger.debug("Liquid agent state reset")

    def predict(
        self,
        features: pd.DataFrame | np.ndarray | torch.Tensor,
        symbol: str,
        maintain_state: bool = True,
    ) -> Signal:
        """
        Generate trading signal from market features.

        Args:
            features: Market features
            symbol: Asset symbol
            maintain_state: Whether to maintain hidden state across calls

        Returns:
            Trading signal
        """
        self.eval_mode()

        with torch.no_grad():
            x = self.preprocess(features)

            # Ensure sequence dimension
            if x.dim() == 2:
                if x.shape[0] >= self.sequence_length:
                    x = x[-self.sequence_length:].unsqueeze(0)
                else:
                    padding = torch.zeros(
                        self.sequence_length - x.shape[0],
                        x.shape[1],
                        device=self.device,
                    )
                    x = torch.cat([padding, x], dim=0).unsqueeze(0)

            # Use persistent hidden state if maintaining state
            hidden = self._hidden_state if maintain_state else None

            # Forward pass
            outputs = self.model(x, hidden=hidden, return_hidden=True)

            # Update persistent hidden state
            if maintain_state:
                self._hidden_state = outputs["hidden"]

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

            # Get regime
            regime_probs = F.softmax(outputs["regime_logits"], dim=-1)
            regime_idx = regime_probs.argmax(dim=-1).item()
            regime_name = self.REGIME_NAMES[regime_idx]
            regime_confidence = regime_probs[0, regime_idx].item()

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
                "regime": regime_name,
                "regime_confidence": regime_confidence,
                "regime_probs": regime_probs[0].cpu().numpy().tolist(),
            },
        )

        self.state.last_signal = signal
        return signal

    def detect_regime(
        self,
        features: pd.DataFrame | np.ndarray | torch.Tensor,
    ) -> dict:
        """
        Detect current market regime.

        Args:
            features: Market features

        Returns:
            Dictionary with regime information
        """
        self.eval_mode()

        with torch.no_grad():
            x = self.preprocess(features)
            if x.dim() == 2:
                x = x.unsqueeze(0)

            outputs = self.model(x)
            regime_probs = F.softmax(outputs["regime_logits"], dim=-1)[0]

            regime_idx = regime_probs.argmax().item()

        return {
            "regime": self.REGIME_NAMES[regime_idx],
            "confidence": regime_probs[regime_idx].item(),
            "probabilities": {
                name: regime_probs[i].item()
                for i, name in enumerate(self.REGIME_NAMES)
            },
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Single training step with multi-task learning.

        Args:
            batch: Dictionary with 'features', 'actions', optionally 'regimes'

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        features = batch["features"].to(self.device)
        action_targets = batch["actions"].to(self.device)

        outputs = self.model(features)

        # Action loss
        action_loss = self.action_loss_fn(outputs["action_logits"], action_targets)
        total_loss = action_loss

        metrics = {"action_loss": action_loss.item()}

        # Regime loss if labels provided
        if "regimes" in batch:
            regime_targets = batch["regimes"].to(self.device)
            regime_loss = self.regime_loss_fn(outputs["regime_logits"], regime_targets)
            total_loss = total_loss + 0.5 * regime_loss
            metrics["regime_loss"] = regime_loss.item()

        # Confidence regularization
        if "returns" in batch:
            returns = batch["returns"].to(self.device)
            # Higher confidence should correlate with correct predictions
            probs = F.softmax(outputs["action_logits"], dim=-1)
            correct_probs = probs.gather(1, action_targets.unsqueeze(1)).squeeze()
            conf_loss = F.mse_loss(outputs["confidence"], correct_probs.detach())
            total_loss = total_loss + 0.1 * conf_loss
            metrics["conf_loss"] = conf_loss.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        metrics["loss"] = total_loss.item()
        return metrics

    def get_config(self) -> dict:
        """Get agent configuration."""
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
        })
        return config


if __name__ == "__main__":
    # Test the Liquid agent
    print("Testing Liquid Neural Network Agent...")
    print("=" * 50)

    # Create agent
    agent = LiquidAgent(
        input_dim=18,
        hidden_dim=32,
        num_layers=2,
        sequence_length=20,
    )

    # Create dummy data
    batch_size = 4
    seq_len = 20
    n_features = 18

    dummy_features = torch.randn(batch_size, seq_len, n_features)

    # Test forward pass
    print("\nTesting forward pass...")
    agent.eval_mode()
    with torch.no_grad():
        outputs = agent.model(dummy_features.to(agent.device))
        print(f"Action logits shape: {outputs['action_logits'].shape}")
        print(f"Regime logits shape: {outputs['regime_logits'].shape}")

    # Test signal generation with state maintenance
    print("\nTesting signal generation (with state)...")
    for i in range(3):
        signal = agent.predict(dummy_features[0], "TEST", maintain_state=True)
        print(f"Signal {i+1}: {signal.action.name}, regime={signal.metadata['regime']}")

    # Test regime detection
    print("\nTesting regime detection...")
    regime = agent.detect_regime(dummy_features[0])
    print(f"Detected regime: {regime['regime']} (conf={regime['confidence']:.2f})")
    print(f"All probabilities: {regime['probabilities']}")

    # Test reset
    print("\nTesting reset...")
    agent.reset()
    signal_after_reset = agent.predict(dummy_features[0], "TEST")
    print(f"Signal after reset: {signal_after_reset.action.name}")

    # Test training step
    print("\nTesting training step...")
    agent.setup_optimizer()
    batch = {
        "features": dummy_features,
        "actions": torch.randint(0, 3, (batch_size,)),
        "regimes": torch.randint(0, 4, (batch_size,)),
        "returns": torch.randn(batch_size),
    }
    metrics = agent.training_step(batch)
    print(f"Training metrics: {metrics}")

    print("\nLiquid agent tests passed!")
