"""
KAN (Kolmogorov-Arnold Network) Trading Agent.

KAN replaces traditional linear weights with learnable univariate functions,
based on the Kolmogorov-Arnold representation theorem. This provides:
- Interpretable trading formulas
- Automatic feature transformation discovery
- Better generalization with fewer parameters

Reference: "KAN: Kolmogorov-Arnold Networks" (2024)
"""

from datetime import datetime
from typing import Optional
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .base import BaseNeuralAgent, Signal, Action, MOMENTUM_FEATURES, VOLATILITY_FEATURES


class BSplineBasis(nn.Module):
    """
    B-spline basis functions for KAN layers.
    Learns smooth univariate transformations.
    """

    def __init__(
        self,
        num_splines: int = 8,
        spline_order: int = 3,
        grid_range: tuple = (-1, 1),
    ):
        """
        Initialize B-spline basis.

        Args:
            num_splines: Number of spline basis functions
            spline_order: Order of B-splines (3 = cubic)
            grid_range: Range of input values
        """
        super().__init__()
        self.num_splines = num_splines
        self.spline_order = spline_order
        self.grid_range = grid_range

        # Create knot vector with padding for boundary conditions
        num_knots = num_splines + spline_order + 1
        knots = torch.linspace(
            grid_range[0] - spline_order * (grid_range[1] - grid_range[0]) / num_splines,
            grid_range[1] + spline_order * (grid_range[1] - grid_range[0]) / num_splines,
            num_knots,
        )
        self.register_buffer("knots", knots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate B-spline basis functions at x.

        Args:
            x: Input values (any shape)

        Returns:
            Basis function values (shape + [num_splines])
        """
        x = x.unsqueeze(-1)  # Add dimension for splines

        # Compute B-spline basis recursively
        bases = self._bspline_basis(x, self.spline_order)

        return bases

    def _bspline_basis(self, x: torch.Tensor, order: int) -> torch.Tensor:
        """Compute B-spline basis functions of given order."""
        if order == 0:
            # Base case: indicator functions
            bases = ((x >= self.knots[:-1]) & (x < self.knots[1:])).float()
            # Handle right boundary
            bases[..., -1] = ((x[..., 0] >= self.knots[-2]) &
                             (x[..., 0] <= self.knots[-1])).float()
            return bases

        # Recursive case
        bases_prev = self._bspline_basis(x, order - 1)

        # B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x) +
        #              (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)

        t = self.knots
        n = self.num_splines

        # First term
        denom1 = t[order:n+order] - t[:n]
        denom1 = torch.where(denom1 == 0, torch.ones_like(denom1), denom1)
        term1 = (x[..., :n] - t[:n]) / denom1 * bases_prev[..., :n]

        # Second term
        denom2 = t[order+1:n+order+1] - t[1:n+1]
        denom2 = torch.where(denom2 == 0, torch.ones_like(denom2), denom2)
        term2 = (t[order+1:n+order+1] - x[..., :n]) / denom2 * bases_prev[..., 1:n+1]

        bases = term1 + term2

        return bases


class KANLayer(nn.Module):
    """
    Single KAN layer with learnable univariate functions.
    Each edge has its own B-spline function that transforms the input.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_splines: int = 8,
        spline_order: int = 3,
        residual: bool = True,
    ):
        """
        Initialize KAN layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            num_splines: Number of B-spline basis functions
            spline_order: B-spline order
            residual: Whether to include residual (linear) connection
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_splines = num_splines
        self.residual = residual

        # B-spline basis (shared across all edges)
        self.basis = BSplineBasis(num_splines, spline_order)

        # Learnable spline coefficients for each edge
        # Shape: (out_features, in_features, num_splines)
        self.spline_weights = nn.Parameter(
            torch.randn(out_features, in_features, num_splines) * 0.1
        )

        # Optional residual connection
        if residual:
            self.residual_weight = nn.Parameter(
                torch.randn(out_features, in_features) * 0.1
            )

        # Learnable scale for each output
        self.scale = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through KAN layer.

        Args:
            x: Input tensor (batch, in_features)

        Returns:
            Output tensor (batch, out_features)
        """
        batch_size = x.shape[0]

        # Normalize input to basis range
        x_norm = torch.tanh(x)  # Map to [-1, 1]

        # Compute B-spline basis values
        # (batch, in_features) -> (batch, in_features, num_splines)
        basis_values = self.basis(x_norm)

        # Apply spline functions to each input
        # Output shape: (batch, out_features)
        # Each output is sum over inputs of spline(input)

        # Reshape for batch matrix multiplication
        # basis_values: (batch, in_features, num_splines)
        # spline_weights: (out_features, in_features, num_splines)

        # Compute spline contributions
        spline_out = torch.einsum(
            "bin,oin->bo",
            basis_values,
            self.spline_weights,
        )

        # Add residual connection
        if self.residual:
            residual_out = F.linear(x, self.residual_weight)
            spline_out = spline_out + residual_out

        # Apply scale
        output = spline_out * self.scale

        return output

    def get_edge_function(self, out_idx: int, in_idx: int) -> callable:
        """
        Get the learned univariate function for a specific edge.
        Useful for interpretability.

        Args:
            out_idx: Output node index
            in_idx: Input node index

        Returns:
            Function that maps input to output contribution
        """
        weights = self.spline_weights[out_idx, in_idx].detach()

        def edge_fn(x):
            x_tensor = torch.tensor(x).float()
            if x_tensor.dim() == 0:
                x_tensor = x_tensor.unsqueeze(0)
            x_norm = torch.tanh(x_tensor)
            basis = self.basis(x_norm)
            return (basis * weights).sum(-1).numpy()

        return edge_fn


class KANModel(nn.Module):
    """
    Full KAN model for trading signal prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [32, 16],
        num_splines: int = 8,
        num_classes: int = 3,
    ):
        """
        Initialize KAN model.

        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            num_splines: Number of B-spline basis functions
            num_classes: Number of output classes
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build KAN layers
        dims = [input_dim] + hidden_dims
        self.kan_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.kan_layers.append(
                KANLayer(dims[i], dims[i + 1], num_splines)
            )

        # Output heads
        last_dim = hidden_dims[-1]

        self.action_head = nn.Sequential(
            KANLayer(last_dim, last_dim // 2, num_splines),
            nn.SiLU(),
            nn.Linear(last_dim // 2, num_classes),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(last_dim, last_dim // 4),
            nn.SiLU(),
            nn.Linear(last_dim // 4, 1),
            nn.Sigmoid(),
        )

        self.position_head = nn.Sequential(
            nn.Linear(last_dim, last_dim // 4),
            nn.SiLU(),
            nn.Linear(last_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features (batch, input_dim) or (batch, seq_len, input_dim)

        Returns:
            Dictionary with action_logits, confidence, position_size
        """
        # If sequence input, use last timestep
        if x.dim() == 3:
            x = x[:, -1, :]

        # Pass through KAN layers
        h = x
        for layer in self.kan_layers:
            h = layer(h)
            h = F.silu(h)

        # Output heads
        action_logits = self.action_head(h)
        confidence = self.confidence_head(h).squeeze(-1)
        position_size = self.position_head(h).squeeze(-1)

        return {
            "action_logits": action_logits,
            "confidence": confidence,
            "position_size": position_size,
            "hidden": h,
        }

    def get_feature_importance(self) -> dict[int, float]:
        """
        Calculate feature importance based on first layer weights.

        Returns:
            Dictionary mapping feature index to importance score
        """
        first_layer = self.kan_layers[0]
        weights = first_layer.spline_weights.detach()

        # Importance = sum of absolute spline weights for each input
        importance = weights.abs().sum(dim=(0, 2))

        return {i: importance[i].item() for i in range(len(importance))}


class KANAgent(BaseNeuralAgent):
    """
    Trading agent using Kolmogorov-Arnold Networks.

    Features:
    - Interpretable learned functions for each feature
    - Automatic feature transformation discovery
    - Can extract symbolic formulas from learned networks
    """

    # Features this agent uses - focus on technical indicators
    FEATURES = MOMENTUM_FEATURES + VOLATILITY_FEATURES + [
        "returns", "log_returns", "returns_5d", "returns_10d",
        "macd", "macd_signal", "macd_histogram",
        "price_sma_20_ratio", "price_sma_50_ratio",
    ]

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dims: list[int] = [32, 16],
        num_splines: int = 8,
        **kwargs,
    ):
        """
        Initialize KAN trading agent.

        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            num_splines: Number of B-spline basis functions
        """
        input_dim = input_dim or len(self.FEATURES)
        super().__init__(
            name="kan",
            input_dim=input_dim,
            hidden_dim=hidden_dims[0] if hidden_dims else 32,
            **kwargs,
        )

        self.hidden_dims = hidden_dims
        self.num_splines = num_splines

        # Initialize model
        self.model = KANModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_splines=num_splines,
        ).to(self.device)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # Feature names for interpretability
        self._feature_names = self.FEATURES[:input_dim]

        self.logger.info(
            f"Initialized KANAgent: hidden_dims={hidden_dims}, "
            f"num_splines={num_splines}"
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
            features: Market features
            symbol: Asset symbol

        Returns:
            Trading signal
        """
        self.eval_mode()

        with torch.no_grad():
            x = self.preprocess(features)

            # Forward pass
            outputs = self.model(x)

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

        # Update state
        self.state.step_count += 1

        # Get feature importance for interpretability
        importance = self.model.get_feature_importance()
        top_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size=position_size,
            timestamp=datetime.now(),
            agent_name=self.name,
            metadata={
                "action_probs": action_probs[0].cpu().numpy().tolist(),
                "top_features": [
                    {
                        "idx": idx,
                        "name": self._feature_names[idx] if idx < len(self._feature_names) else f"feature_{idx}",
                        "importance": imp,
                    }
                    for idx, imp in top_features
                ],
            },
        )

        self.state.last_signal = signal
        return signal

    def get_interpretable_formula(self, output_idx: int = 0) -> str:
        """
        Extract a human-readable formula for the model's decision.
        This is approximate and for interpretability purposes.

        Args:
            output_idx: Which output to explain (0=HOLD, 1=BUY, 2=SELL)

        Returns:
            String representation of the learned formula
        """
        importance = self.model.get_feature_importance()

        # Get top contributing features
        top_features = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        formula_parts = []
        for idx, imp in top_features:
            name = self._feature_names[idx] if idx < len(self._feature_names) else f"x{idx}"
            formula_parts.append(f"φ_{idx}({name})")

        action_names = ["HOLD", "BUY", "SELL"]
        formula = f"{action_names[output_idx]} ≈ " + " + ".join(formula_parts)

        return formula

    def visualize_learned_functions(self, feature_idx: int = 0) -> dict:
        """
        Get data for visualizing the learned function for a feature.

        Args:
            feature_idx: Index of feature to visualize

        Returns:
            Dictionary with x values and y values for plotting
        """
        x_vals = np.linspace(-3, 3, 100)

        # Get the edge function from first layer
        first_layer = self.model.kan_layers[0]
        edge_fn = first_layer.get_edge_function(0, feature_idx)

        y_vals = edge_fn(x_vals)

        return {
            "x": x_vals.tolist(),
            "y": y_vals.tolist(),
            "feature_name": (
                self._feature_names[feature_idx]
                if feature_idx < len(self._feature_names)
                else f"feature_{feature_idx}"
            ),
        }

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        features = batch["features"].to(self.device)
        targets = batch["actions"].to(self.device)

        outputs = self.model(features)
        loss = self.loss_fn(outputs["action_logits"], targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": loss.item()}

    def get_config(self) -> dict:
        """Get agent configuration."""
        config = super().get_config()
        config.update({
            "hidden_dims": self.hidden_dims,
            "num_splines": self.num_splines,
        })
        return config


if __name__ == "__main__":
    # Test the KAN agent
    print("Testing KAN Agent...")
    print("=" * 50)

    # Create agent
    agent = KANAgent(
        input_dim=15,
        hidden_dims=[24, 12],
        num_splines=6,
    )

    # Create dummy data
    batch_size = 4
    n_features = 15

    dummy_features = torch.randn(batch_size, n_features)

    # Test forward pass
    print("\nTesting forward pass...")
    agent.eval_mode()
    with torch.no_grad():
        outputs = agent.model(dummy_features.to(agent.device))
        print(f"Action logits shape: {outputs['action_logits'].shape}")
        print(f"Confidence shape: {outputs['confidence'].shape}")

    # Test signal generation
    print("\nTesting signal generation...")
    signal = agent.predict(dummy_features[0], "TEST")
    print(f"Signal: {signal.action.name}")
    print(f"Confidence: {signal.confidence:.3f}")
    print(f"Top features: {signal.metadata['top_features'][:3]}")

    # Test interpretability
    print("\nTesting interpretability...")
    formula = agent.get_interpretable_formula(1)  # BUY formula
    print(f"Formula: {formula}")

    # Test learned function visualization
    viz_data = agent.visualize_learned_functions(0)
    print(f"Visualization data for {viz_data['feature_name']}: {len(viz_data['x'])} points")

    # Test training step
    print("\nTesting training step...")
    agent.setup_optimizer()
    batch = {
        "features": dummy_features,
        "actions": torch.randint(0, 3, (batch_size,)),
    }
    metrics = agent.training_step(batch)
    print(f"Training loss: {metrics['loss']:.4f}")

    print("\nKAN agent tests passed!")
