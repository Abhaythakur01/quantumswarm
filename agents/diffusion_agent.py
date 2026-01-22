"""
Diffusion Scenario Generator Agent.

Uses denoising diffusion probabilistic models (DDPMs) to generate synthetic
market scenarios. This enables:
- Data augmentation for training
- Stress testing portfolios
- Monte Carlo simulation for risk assessment
- Generating rare market events for robustness

Reference: "Denoising Diffusion Probabilistic Models" (2020)
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

from .base import BaseNeuralAgent, Signal, Action, ALL_FEATURES


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: Tensor of timestep indices (batch_size,)
        embedding_dim: Dimension of the embedding

    Returns:
        Timestep embeddings (batch_size, embedding_dim)
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


class ResidualBlock(nn.Module):
    """Residual block with timestep conditioning."""

    def __init__(self, dim: int, time_emb_dim: int, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.time_mlp = nn.Linear(time_emb_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.gelu(self.linear1(h))
        h = h + self.time_mlp(t_emb).unsqueeze(1)  # Add time embedding
        h = self.norm2(h)
        h = self.dropout(F.gelu(self.linear2(h)))
        return x + h


class DiffusionUNet(nn.Module):
    """
    U-Net style architecture for diffusion denoising.
    Adapted for 1D time series data.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        time_emb_dim: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        current_dim = hidden_dim
        dims = [hidden_dim]

        for i in range(num_layers):
            self.encoder_blocks.append(
                ResidualBlock(current_dim, time_emb_dim, dropout)
            )
            if i < num_layers - 1:
                next_dim = min(current_dim * 2, 512)
                self.downsample.append(nn.Linear(current_dim, next_dim))
                current_dim = next_dim
                dims.append(current_dim)

        # Middle block
        self.middle_block = ResidualBlock(current_dim, time_emb_dim, dropout)

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i in range(num_layers - 1, -1, -1):
            if i < num_layers - 1:
                prev_dim = dims[i]
                self.upsample.append(nn.Linear(current_dim, prev_dim))
                # Account for skip connection
                self.decoder_blocks.append(
                    ResidualBlock(prev_dim * 2, time_emb_dim, dropout)
                )
                current_dim = prev_dim
            else:
                self.decoder_blocks.append(
                    ResidualBlock(current_dim, time_emb_dim, dropout)
                )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to predict noise.

        Args:
            x: Noisy input (batch, seq_len, input_dim)
            t: Timestep indices (batch,)
            condition: Optional conditioning information

        Returns:
            Predicted noise (batch, seq_len, input_dim)
        """
        # Get timestep embedding
        t_emb = get_timestep_embedding(t, self.time_emb_dim)
        t_emb = self.time_mlp(t_emb)

        # Input projection
        h = self.input_proj(x)

        # Encoder with skip connections
        skips = []
        for i, block in enumerate(self.encoder_blocks):
            h = block(h, t_emb)
            if i < len(self.downsample):
                skips.append(h)
                h = self.downsample[i](h)

        # Middle
        h = self.middle_block(h, t_emb)

        # Decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            if i > 0:
                h = self.upsample[i - 1](h)
                skip = skips.pop()
                h = torch.cat([h, skip], dim=-1)
            h = block(h, t_emb)

        # Output
        return self.output_proj(h)


class GaussianDiffusion:
    """
    Gaussian diffusion process for training and sampling.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        self.num_timesteps = num_timesteps

        # Create beta schedule
        if schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
            alpha_bar = torch.cos((steps / num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device: torch.device):
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion process: q(x_t | x_0).

        Args:
            x_0: Original data
            t: Timestep indices
            noise: Optional pre-sampled noise

        Returns:
            x_t: Noisy data
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

        return x_t, noise

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> dict:
        """
        Compute mean and variance of p(x_{t-1} | x_t).
        """
        # Predict noise
        pred_noise = model(x_t, t)

        # Compute x_0 prediction
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        pred_x_0 = (x_t - sqrt_one_minus_alpha * pred_noise) / sqrt_alpha

        # Clip x_0 prediction for stability
        pred_x_0 = torch.clamp(pred_x_0, -10, 10)

        # Compute posterior mean
        coef1 = self.posterior_mean_coef1[t][:, None, None]
        coef2 = self.posterior_mean_coef2[t][:, None, None]
        posterior_mean = coef1 * pred_x_0 + coef2 * x_t

        # Posterior variance
        posterior_variance = self.posterior_variance[t][:, None, None]
        posterior_log_variance = self.posterior_log_variance_clipped[t][:, None, None]

        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_x_0": pred_x_0,
        }

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from p(x_{t-1} | x_t).
        """
        out = self.p_mean_variance(model, x_t, t)

        # No noise when t == 0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float()[:, None, None]

        return out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples using reverse diffusion.

        Args:
            model: Denoising model
            shape: Shape of samples to generate (batch, seq_len, features)
            device: Device to generate on
            num_steps: Number of denoising steps (default: all)

        Returns:
            Generated samples
        """
        num_steps = num_steps or self.num_timesteps
        step_size = self.num_timesteps // num_steps

        # Start from pure noise
        x = torch.randn(shape, device=device)

        # Reverse diffusion
        for i in reversed(range(0, self.num_timesteps, step_size)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)

        return x


class DiffusionScenarioGenerator(BaseNeuralAgent):
    """
    Agent that generates synthetic market scenarios using diffusion models.

    Use cases:
    - Generate training data augmentation
    - Create stress test scenarios
    - Monte Carlo risk simulation
    - Explore counterfactual market conditions
    """

    FEATURES = ALL_FEATURES

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        sequence_length: int = 60,
        num_timesteps: int = 1000,
        **kwargs,
    ):
        """
        Initialize Diffusion Scenario Generator.

        Args:
            input_dim: Feature dimension
            hidden_dim: Model hidden dimension
            sequence_length: Length of generated sequences
            num_timesteps: Number of diffusion steps
        """
        input_dim = input_dim or len(self.FEATURES)
        super().__init__(
            name="diffusion_generator",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **kwargs,
        )

        self.sequence_length = sequence_length
        self.num_timesteps = num_timesteps

        # Initialize denoising model
        self.model = DiffusionUNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            time_emb_dim=64,
            num_layers=4,
            dropout=self.dropout,
        ).to(self.device)

        # Initialize diffusion process
        self.diffusion = GaussianDiffusion(
            num_timesteps=num_timesteps,
            schedule="cosine",
        ).to(self.device)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Statistics for normalization (learned from data)
        self.register_buffer_safe("data_mean", torch.zeros(input_dim))
        self.register_buffer_safe("data_std", torch.ones(input_dim))

        self.logger.info(
            f"Initialized DiffusionScenarioGenerator: hidden_dim={hidden_dim}, "
            f"seq_len={sequence_length}, timesteps={num_timesteps}"
        )

    def register_buffer_safe(self, name: str, tensor: torch.Tensor):
        """Register a buffer, handling the case where model might not have register_buffer."""
        setattr(self, name, tensor.to(self.device))

    def get_feature_names(self) -> list[str]:
        return self.FEATURES

    def fit_normalizer(self, data: torch.Tensor):
        """
        Fit normalization statistics from training data.

        Args:
            data: Training data (num_samples, seq_len, features)
        """
        # Compute statistics across samples and time
        flat_data = data.reshape(-1, data.shape[-1])
        self.data_mean = flat_data.mean(dim=0).to(self.device)
        self.data_std = flat_data.std(dim=0).clamp(min=1e-6).to(self.device)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize data to zero mean, unit variance."""
        return (x - self.data_mean) / self.data_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to original scale."""
        return x * self.data_std + self.data_mean

    def predict(
        self,
        features: pd.DataFrame | np.ndarray | torch.Tensor,
        symbol: str,
    ) -> Signal:
        """
        Generate a trading signal based on generated scenarios.

        This agent generates multiple future scenarios and aggregates them
        to produce a trading signal based on expected outcomes.

        Args:
            features: Current market features (used as conditioning)
            symbol: Asset symbol

        Returns:
            Trading signal based on scenario analysis
        """
        self.eval_mode()

        with torch.no_grad():
            x = self.preprocess(features)

            # Generate multiple scenarios
            num_scenarios = 50
            scenarios = self.generate_scenarios(
                num_samples=num_scenarios,
                conditioning=x[-self.sequence_length:] if x.shape[0] >= self.sequence_length else x,
            )

            # Analyze scenarios - look at the "returns" feature if available
            # Assuming returns is one of the features
            returns_idx = 0  # Index of returns feature

            # Get expected return from scenarios
            scenario_returns = scenarios[:, -1, returns_idx]  # Last timestep returns
            expected_return = scenario_returns.mean().item()
            return_std = scenario_returns.std().item()

            # Determine action based on expected return and uncertainty
            if expected_return > 0.01 and return_std < 0.05:
                action = Action.BUY
                confidence = min(0.9, 0.5 + expected_return * 10)
            elif expected_return < -0.01 and return_std < 0.05:
                action = Action.SELL
                confidence = min(0.9, 0.5 + abs(expected_return) * 10)
            else:
                action = Action.HOLD
                confidence = 0.5

            # Position size based on confidence and uncertainty
            position_size = confidence * (1 - min(return_std * 10, 0.5))

            # Calculate scenario statistics
            bullish_scenarios = (scenario_returns > 0.01).sum().item() / num_scenarios
            bearish_scenarios = (scenario_returns < -0.01).sum().item() / num_scenarios

        signal = Signal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            position_size=position_size,
            timestamp=datetime.now(),
            agent_name=self.name,
            metadata={
                "expected_return": expected_return,
                "return_std": return_std,
                "bullish_probability": bullish_scenarios,
                "bearish_probability": bearish_scenarios,
                "num_scenarios": num_scenarios,
            },
        )

        self.state.step_count += 1
        self.state.last_signal = signal

        return signal

    @torch.no_grad()
    def generate_scenarios(
        self,
        num_samples: int = 10,
        conditioning: Optional[torch.Tensor] = None,
        num_steps: int = 100,
    ) -> torch.Tensor:
        """
        Generate synthetic market scenarios.

        Args:
            num_samples: Number of scenarios to generate
            conditioning: Optional conditioning data
            num_steps: Number of denoising steps

        Returns:
            Generated scenarios (num_samples, seq_len, features)
        """
        self.eval_mode()

        shape = (num_samples, self.sequence_length, self.input_dim)

        # Generate samples
        samples = self.diffusion.sample(
            self.model,
            shape,
            self.device,
            num_steps=num_steps,
        )

        # Denormalize
        samples = self.denormalize(samples)

        return samples

    def generate_stress_scenarios(
        self,
        base_scenario: torch.Tensor,
        stress_factor: float = 2.0,
        num_samples: int = 10,
    ) -> torch.Tensor:
        """
        Generate stress test scenarios with amplified volatility.

        Args:
            base_scenario: Base market scenario
            stress_factor: Factor to amplify volatility
            num_samples: Number of stress scenarios

        Returns:
            Stress test scenarios
        """
        self.eval_mode()

        with torch.no_grad():
            # Normalize base scenario
            base_norm = self.normalize(base_scenario)

            # Add extra noise for stress scenarios
            noise = torch.randn(
                num_samples, *base_norm.shape,
                device=self.device
            ) * stress_factor

            stressed = base_norm.unsqueeze(0) + noise

            # Denormalize
            stressed = self.denormalize(stressed)

        return stressed

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Training step for diffusion model.

        Args:
            batch: Dictionary with 'features' key

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        x_0 = batch["features"].to(self.device)

        # Normalize
        x_0 = self.normalize(x_0)

        # Sample random timesteps
        batch_size = x_0.shape[0]
        t = torch.randint(
            0, self.num_timesteps,
            (batch_size,),
            device=self.device,
        )

        # Add noise
        x_t, noise = self.diffusion.q_sample(x_0, t)

        # Predict noise
        pred_noise = self.model(x_t, t)

        # Loss
        loss = self.loss_fn(pred_noise, noise)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": loss.item()}

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "num_timesteps": self.num_timesteps,
        })
        return config


if __name__ == "__main__":
    print("Testing Diffusion Scenario Generator...")
    print("=" * 50)

    # Create agent
    agent = DiffusionScenarioGenerator(
        input_dim=10,
        hidden_dim=64,
        sequence_length=30,
        num_timesteps=100,
    )

    # Create dummy training data
    batch_size = 8
    seq_len = 30
    n_features = 10

    dummy_data = torch.randn(batch_size * 10, seq_len, n_features)

    # Fit normalizer
    print("\nFitting normalizer...")
    agent.fit_normalizer(dummy_data)

    # Test scenario generation
    print("\nGenerating scenarios...")
    scenarios = agent.generate_scenarios(num_samples=5, num_steps=50)
    print(f"Generated scenarios shape: {scenarios.shape}")

    # Test signal generation
    print("\nTesting signal generation...")
    signal = agent.predict(dummy_data[0], "TEST")
    print(f"Signal: {signal.action.name}")
    print(f"Expected return: {signal.metadata['expected_return']:.4f}")
    print(f"Bullish probability: {signal.metadata['bullish_probability']:.2f}")

    # Test training step
    print("\nTesting training step...")
    agent.setup_optimizer()
    batch = {"features": dummy_data[:batch_size]}
    metrics = agent.training_step(batch)
    print(f"Training loss: {metrics['loss']:.4f}")

    # Test stress scenarios
    print("\nGenerating stress scenarios...")
    stress = agent.generate_stress_scenarios(
        dummy_data[0].to(agent.device),
        stress_factor=2.0,
        num_samples=3,
    )
    print(f"Stress scenarios shape: {stress.shape}")

    print("\nDiffusion agent tests passed!")
