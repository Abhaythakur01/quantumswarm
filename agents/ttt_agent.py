"""
Test-Time Training (TTT) Trading Agent.

Implements adaptation during inference using self-supervised learning.
The model adapts its parameters on each new input before making predictions,
allowing it to handle distribution shift in real-time.

Key features:
- Online adaptation: Updates model on each input
- Self-supervised objectives: No labels needed for adaptation
- Fast adaptation: Few gradient steps per input
- Distribution shift handling: Adapts to changing market regimes

Reference: "Test-Time Training with Self-Supervision for Generalization under Distribution Shift" (2020)
"""

from datetime import datetime
from typing import Optional
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from .base import BaseNeuralAgent, Signal, Action, CORE_FEATURES, TREND_FEATURES, VOLATILITY_FEATURES


class MaskedAutoencoder(nn.Module):
    """
    Masked autoencoder for self-supervised feature reconstruction.
    Used as the auxiliary task for test-time training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        mask_ratio: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.mask_ratio = mask_ratio

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with masking.

        Args:
            x: Input features (batch, input_dim) or (batch, seq_len, input_dim)
            mask: Optional pre-computed mask

        Returns:
            reconstruction: Reconstructed features
            latent: Latent representation
            mask: The mask that was applied
        """
        # Handle sequence input
        original_shape = x.shape
        if x.dim() == 3:
            batch_size, seq_len, feat_dim = x.shape
            x = x.reshape(-1, feat_dim)
        else:
            batch_size = x.shape[0]
            seq_len = 1

        # Create mask if not provided
        if mask is None:
            mask = torch.rand(x.shape, device=x.device) > self.mask_ratio
            mask = mask.float()

        # Apply mask (replace masked values with 0)
        x_masked = x * mask

        # Encode and decode
        latent = self.encoder(x_masked)
        reconstruction = self.decoder(latent)

        # Reshape back if needed
        if len(original_shape) == 3:
            reconstruction = reconstruction.reshape(batch_size, seq_len, feat_dim)
            latent = latent.reshape(batch_size, seq_len, -1)
            mask = mask.reshape(batch_size, seq_len, feat_dim)

        return reconstruction, latent, mask


class TemporalContrastiveLoss(nn.Module):
    """
    Temporal contrastive learning objective.
    Learns representations where temporally close samples are similar.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_t: torch.Tensor,
        z_t1: torch.Tensor,
        z_neg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between consecutive timesteps.

        Args:
            z_t: Representation at time t
            z_t1: Representation at time t+1 (positive)
            z_neg: Negative samples (optional, uses in-batch negatives if None)

        Returns:
            Contrastive loss
        """
        # Normalize
        z_t = F.normalize(z_t, dim=-1)
        z_t1 = F.normalize(z_t1, dim=-1)

        # Positive similarity
        pos_sim = (z_t * z_t1).sum(dim=-1) / self.temperature

        # Negative similarities (in-batch)
        if z_neg is None:
            # Use other samples in batch as negatives
            neg_sim = torch.mm(z_t, z_t1.t()) / self.temperature
        else:
            z_neg = F.normalize(z_neg, dim=-1)
            neg_sim = torch.mm(z_t, z_neg.t()) / self.temperature

        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)


class TTTModel(nn.Module):
    """
    Test-Time Training model with shared encoder and task-specific heads.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_classes: int = 3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Shared encoder (updated during TTT)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Self-supervised head (for TTT adaptation)
        self.ss_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Main task head (frozen during TTT)
        self.task_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim // 2, num_classes),
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Position sizing head
        self.position_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 4),
            nn.GELU(),
            nn.Linear(latent_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_reconstruction: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input features
            return_reconstruction: Whether to return self-supervised reconstruction

        Returns:
            Dictionary with predictions
        """
        # Handle sequence input
        if x.dim() == 3:
            x = x[:, -1, :]  # Use last timestep

        # Encode
        z = self.encoder(x)

        # Main task predictions
        action_logits = self.task_head(z)
        confidence = self.confidence_head(z).squeeze(-1)
        position_size = self.position_head(z).squeeze(-1)

        outputs = {
            "action_logits": action_logits,
            "confidence": confidence,
            "position_size": position_size,
            "latent": z,
        }

        # Self-supervised reconstruction
        if return_reconstruction:
            reconstruction = self.ss_head(z)
            outputs["reconstruction"] = reconstruction

        return outputs

    def get_encoder_params(self):
        """Get encoder parameters (for TTT updates)."""
        return self.encoder.parameters()

    def get_ss_params(self):
        """Get self-supervised head parameters."""
        return list(self.encoder.parameters()) + list(self.ss_head.parameters())


class TestTimeTrainingAgent(BaseNeuralAgent):
    """
    Trading agent with Test-Time Training capability.

    Adapts to each new input using self-supervised learning before
    making predictions. This enables real-time adaptation to
    distribution shift (changing market conditions).

    Features:
    - Masked autoencoding for adaptation
    - Temporal contrastive learning
    - Fast adaptation (1-3 gradient steps)
    - Maintains original model for stability
    """

    FEATURES = CORE_FEATURES + TREND_FEATURES + VOLATILITY_FEATURES

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        adaptation_steps: int = 1,
        adaptation_lr: float = 0.001,
        use_temporal_contrast: bool = True,
        sequence_length: int = 30,
        **kwargs,
    ):
        """
        Initialize TTT Agent.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            latent_dim: Latent representation dimension
            adaptation_steps: Number of gradient steps per input
            adaptation_lr: Learning rate for adaptation
            use_temporal_contrast: Whether to use temporal contrastive loss
            sequence_length: Input sequence length
        """
        input_dim = input_dim or len(self.FEATURES)
        super().__init__(
            name="ttt_agent",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )

        self.latent_dim = latent_dim
        self.adaptation_steps = adaptation_steps
        self.adaptation_lr = adaptation_lr
        self.use_temporal_contrast = use_temporal_contrast
        self.sequence_length = sequence_length

        # Initialize model
        self.model = TTTModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(self.device)

        # Keep a copy of original model for reset
        self._original_state = None

        # Self-supervised components
        self.mae = MaskedAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(self.device)

        self.temporal_loss = TemporalContrastiveLoss()

        # Adaptation optimizer (separate from main training)
        self.adaptation_optimizer = torch.optim.SGD(
            self.model.get_ss_params(),
            lr=adaptation_lr,
        )

        # Recent data buffer for temporal contrast
        self._recent_latents: list[torch.Tensor] = []
        self._max_buffer_size = 32

        # Loss function for main task
        self.loss_fn = nn.CrossEntropyLoss()

        self.logger.info(
            f"Initialized TTTAgent: adaptation_steps={adaptation_steps}, "
            f"adaptation_lr={adaptation_lr}"
        )

    def get_feature_names(self) -> list[str]:
        return self.FEATURES

    def save_original_state(self):
        """Save the current model state as the original."""
        self._original_state = deepcopy(self.model.state_dict())

    def restore_original_state(self):
        """Restore model to original state."""
        if self._original_state is not None:
            self.model.load_state_dict(self._original_state)

    def _adaptation_step(
        self,
        x: torch.Tensor,
    ) -> dict[str, float]:
        """
        Perform one TTT adaptation step.

        Args:
            x: Input features (batch, features) or (batch, seq_len, features)

        Returns:
            Dictionary of adaptation metrics
        """
        self.model.train()

        # Handle sequence input
        if x.dim() == 3:
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            x_flat = x

        total_loss = 0.0
        metrics = {}

        # 1. Masked reconstruction loss
        reconstruction, latent, mask = self.mae(x_flat)
        recon_loss = F.mse_loss(
            reconstruction * (1 - mask),  # Only on masked positions
            x_flat * (1 - mask),
        )
        total_loss = total_loss + recon_loss
        metrics["recon_loss"] = recon_loss.item()

        # 2. Temporal contrastive loss (if using sequences)
        if self.use_temporal_contrast and x.dim() == 3 and x.shape[1] > 1:
            # Get latents for consecutive timesteps
            outputs_t = self.model(x[:, :-1, :], return_reconstruction=False)
            outputs_t1 = self.model(x[:, 1:, :], return_reconstruction=False)

            z_t = outputs_t["latent"].reshape(-1, self.latent_dim)
            z_t1 = outputs_t1["latent"].reshape(-1, self.latent_dim)

            contrast_loss = self.temporal_loss(z_t, z_t1)
            total_loss = total_loss + 0.5 * contrast_loss
            metrics["contrast_loss"] = contrast_loss.item()

        # 3. Entropy regularization (encourage confident predictions)
        outputs = self.model(x, return_reconstruction=False)
        probs = F.softmax(outputs["action_logits"], dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        total_loss = total_loss + 0.1 * entropy
        metrics["entropy"] = entropy.item()

        # Backward pass (only update encoder and SS head)
        self.adaptation_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.get_ss_params(), max_norm=1.0)
        self.adaptation_optimizer.step()

        metrics["total_loss"] = total_loss.item()
        return metrics

    def adapt(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> dict[str, float]:
        """
        Adapt the model to new input.

        Args:
            x: Input features to adapt to
            num_steps: Number of adaptation steps (default: self.adaptation_steps)

        Returns:
            Dictionary of adaptation metrics
        """
        num_steps = num_steps or self.adaptation_steps

        all_metrics = []
        for step in range(num_steps):
            metrics = self._adaptation_step(x)
            all_metrics.append(metrics)

        # Average metrics across steps
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return avg_metrics

    def predict(
        self,
        features: pd.DataFrame | np.ndarray | torch.Tensor,
        symbol: str,
        adapt: bool = True,
    ) -> Signal:
        """
        Generate trading signal with optional test-time adaptation.

        Args:
            features: Market features
            symbol: Asset symbol
            adapt: Whether to perform TTT adaptation

        Returns:
            Trading signal
        """
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

        # Test-time adaptation
        adaptation_metrics = {}
        if adapt:
            # Save state before adaptation
            pre_adapt_state = deepcopy(self.model.state_dict())

            # Adapt
            adaptation_metrics = self.adapt(x)

            # Optional: restore state after prediction (for stability)
            # self.model.load_state_dict(pre_adapt_state)

        # Make prediction
        self.eval_mode()
        with torch.no_grad():
            outputs = self.model(x, return_reconstruction=False)

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

            # Store latent for temporal contrast
            latent = outputs["latent"]
            self._recent_latents.append(latent.detach())
            if len(self._recent_latents) > self._max_buffer_size:
                self._recent_latents.pop(0)

        # Create signal
        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size=position_size,
            timestamp=datetime.now(),
            agent_name=self.name,
            metadata={
                "action_probs": action_probs[0].cpu().numpy().tolist(),
                "adapted": adapt,
                "adaptation_metrics": adaptation_metrics,
            },
        )

        self.state.step_count += 1
        self.state.last_signal = signal

        return signal

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Main training step (not TTT adaptation).

        Args:
            batch: Dictionary with 'features', 'actions'

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        features = batch["features"].to(self.device)
        targets = batch["actions"].to(self.device)

        # Forward pass
        outputs = self.model(features, return_reconstruction=True)

        # Main task loss
        action_loss = self.loss_fn(outputs["action_logits"], targets)

        # Self-supervised loss (for joint training)
        if features.dim() == 3:
            features_flat = features[:, -1, :]
        else:
            features_flat = features

        recon_loss = F.mse_loss(outputs["reconstruction"], features_flat)

        # Total loss
        total_loss = action_loss + 0.1 * recon_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "loss": total_loss.item(),
            "action_loss": action_loss.item(),
            "recon_loss": recon_loss.item(),
        }

    def reset(self):
        """Reset agent state and optionally restore original model."""
        super().reset()
        self._recent_latents = []
        # Optionally restore original state
        # self.restore_original_state()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "latent_dim": self.latent_dim,
            "adaptation_steps": self.adaptation_steps,
            "adaptation_lr": self.adaptation_lr,
            "use_temporal_contrast": self.use_temporal_contrast,
            "sequence_length": self.sequence_length,
        })
        return config


if __name__ == "__main__":
    print("Testing Test-Time Training Agent...")
    print("=" * 50)

    # Create agent
    agent = TestTimeTrainingAgent(
        input_dim=15,
        hidden_dim=32,
        latent_dim=16,
        adaptation_steps=2,
        adaptation_lr=0.01,
        sequence_length=20,
    )

    # Create dummy data
    batch_size = 4
    seq_len = 20
    n_features = 15

    dummy_features = torch.randn(batch_size, seq_len, n_features)

    # Test forward pass without adaptation
    print("\nTesting forward pass (no adaptation)...")
    agent.eval_mode()
    with torch.no_grad():
        outputs = agent.model(dummy_features.to(agent.device))
        print(f"Action logits shape: {outputs['action_logits'].shape}")
        print(f"Latent shape: {outputs['latent'].shape}")

    # Test adaptation
    print("\nTesting TTT adaptation...")
    agent.save_original_state()
    x = dummy_features[0:1].to(agent.device)
    metrics = agent.adapt(x, num_steps=3)
    print(f"Adaptation metrics: {metrics}")

    # Test signal generation with adaptation
    print("\nTesting signal generation with adaptation...")
    signal = agent.predict(dummy_features[0], "TEST", adapt=True)
    print(f"Signal: {signal.action.name}")
    print(f"Confidence: {signal.confidence:.3f}")
    print(f"Adaptation metrics: {signal.metadata['adaptation_metrics']}")

    # Test signal generation without adaptation
    print("\nTesting signal generation without adaptation...")
    signal_no_adapt = agent.predict(dummy_features[1], "TEST", adapt=False)
    print(f"Signal (no adapt): {signal_no_adapt.action.name}")

    # Test main training step
    print("\nTesting main training step...")
    agent.setup_optimizer()
    batch = {
        "features": dummy_features,
        "actions": torch.randint(0, 3, (batch_size,)),
    }
    train_metrics = agent.training_step(batch)
    print(f"Training metrics: {train_metrics}")

    # Test state restoration
    print("\nTesting state restoration...")
    agent.restore_original_state()
    signal_restored = agent.predict(dummy_features[0], "TEST", adapt=False)
    print(f"Signal after restore: {signal_restored.action.name}")

    print("\nTTT agent tests passed!")
