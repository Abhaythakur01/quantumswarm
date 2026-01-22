#!/usr/bin/env python3
"""
Training script for all trading agents.

Usage:
    python scripts/train_agents.py --agent mamba --data data/processed/features.parquet
    python scripts/train_agents.py --agent all --epochs 100 --wandb
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
from loguru import logger

from agents import (
    MambaAgent,
    KANAgent,
    LiquidAgent,
    TestTimeTrainingAgent,
    DiffusionScenarioGenerator,
    MixtureOfExpertsCoordinator,
)
from training.datasets import (
    DatasetConfig,
    TradingDataset,
    RegimeDataset,
    create_dataloaders,
    load_and_prepare_data,
)
from training.trainer import Trainer, TrainerConfig
from training.checkpointing import ModelRegistry, CheckpointManager
from training.experiment import ExperimentConfig, ExperimentTracker, create_experiment_name
from data.processors.technical import add_technical_features


# Agent registry
AGENT_CLASSES = {
    "mamba": MambaAgent,
    "kan": KANAgent,
    "liquid": LiquidAgent,
    "ttt": TestTimeTrainingAgent,
    "diffusion": DiffusionScenarioGenerator,
    "moe": MixtureOfExpertsCoordinator,
}

# Default configurations for each agent
AGENT_CONFIGS = {
    "mamba": {
        "d_model": 128,
        "d_state": 16,
        "n_layers": 4,
        "sequence_length": 60,
    },
    "kan": {
        "hidden_dims": [64, 32],
        "num_splines": 8,
    },
    "liquid": {
        "hidden_dim": 64,
        "num_layers": 2,
        "sequence_length": 30,
    },
    "ttt": {
        "hidden_dim": 64,
        "latent_dim": 32,
        "adaptation_steps": 1,
        "sequence_length": 30,
    },
    "diffusion": {
        "hidden_dim": 128,
        "sequence_length": 60,
        "num_timesteps": 1000,
    },
    "moe": {
        "num_experts": 4,
        "top_k": 2,
        "hidden_dim": 64,
    },
}


def create_synthetic_data(
    num_samples: int = 5000,
    num_features: int = 20,
) -> pd.DataFrame:
    """Create synthetic market data for testing."""
    import numpy as np

    np.random.seed(42)

    dates = pd.date_range(start="2020-01-01", periods=num_samples, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.randn(num_samples) * 0.02))

    data = pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 + np.random.randn(num_samples) * 0.01),
        "high": prices * (1 + np.abs(np.random.randn(num_samples) * 0.02)),
        "low": prices * (1 - np.abs(np.random.randn(num_samples) * 0.02)),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, num_samples),
    })

    # Add technical features
    data = add_technical_features(data)

    return data


def get_feature_columns(agent_class) -> list[str]:
    """Get feature columns for an agent."""
    if hasattr(agent_class, "FEATURES"):
        return agent_class.FEATURES
    return [
        "returns", "log_returns", "rsi_14", "macd", "macd_signal",
        "bb_position", "atr_pct", "volume_ratio", "momentum_10",
        "volatility_10d", "stoch_k", "stoch_d", "cci", "williams_r",
        "price_sma_20_ratio", "sma_20_50_cross",
    ]


def train_agent(
    agent_name: str,
    data: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    """
    Train a single agent.

    Args:
        agent_name: Name of agent to train
        data: Training data
        args: Command line arguments

    Returns:
        Training results dictionary
    """
    logger.info(f"Training agent: {agent_name}")

    # Get agent class and config
    agent_class = AGENT_CLASSES[agent_name]
    agent_config = AGENT_CONFIGS.get(agent_name, {}).copy()

    # Get features
    feature_columns = get_feature_columns(agent_class)
    available_features = [f for f in feature_columns if f in data.columns]

    if len(available_features) < 5:
        logger.warning(f"Only {len(available_features)} features available, using defaults")
        available_features = [c for c in data.columns if c not in ["timestamp", "symbol"]][:20]

    logger.info(f"Using {len(available_features)} features")

    # Create dataset config
    dataset_config = DatasetConfig(
        sequence_length=agent_config.get("sequence_length", 60),
        feature_columns=available_features,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        label_threshold=0.001,  # 0.1% threshold for BUY/SELL
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data=data,
        config=dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create agent
    agent_config["input_dim"] = len(available_features)
    agent = agent_class(**agent_config)

    # Create trainer config
    trainer_config = TrainerConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        early_stopping=True,
        patience=args.patience,
        checkpoint_dir=f"checkpoints/{agent_name}",
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # Create experiment tracker
    experiment_name = create_experiment_name(agent_name, args.dataset_name)
    experiment_config = ExperimentConfig(
        name=experiment_name,
        project=args.wandb_project,
        use_wandb=args.wandb,
        use_tensorboard=args.tensorboard,
        tags=[agent_name, "training"],
    )

    # Create trainer
    trainer = Trainer(
        agent=agent,
        config=trainer_config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    # Train
    results = trainer.train()

    # Save model to registry
    if args.save_model:
        registry = ModelRegistry(args.model_dir)
        registry.register_model(
            agent=agent,
            metrics=results.get("test_metrics", {}),
            trained_epochs=results["history"]["train_loss"].__len__(),
            best_val_loss=results["best_val_loss"],
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train trading agents")

    # Agent selection
    parser.add_argument(
        "--agent",
        type=str,
        default="mamba",
        choices=list(AGENT_CLASSES.keys()) + ["all"],
        help="Agent to train (or 'all')",
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data file (CSV or Parquet)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="default",
        help="Name of the dataset",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Tracking
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="multiagent-trading", help="W&B project")
    parser.add_argument("--tensorboard", action="store_true", help="Use TensorBoard")

    # Output
    parser.add_argument("--model-dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--save-model", action="store_true", default=True, help="Save trained model")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")

    args = parser.parse_args()

    # Setup logging
    logger.add(
        f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        rotation="100 MB",
    )

    logger.info("=" * 60)
    logger.info("Multi-Agent Trading System - Training")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")

    # Load or create data
    if args.synthetic or args.data is None:
        logger.info("Using synthetic data")
        data = create_synthetic_data(num_samples=5000)
    else:
        logger.info(f"Loading data from: {args.data}")
        data = load_and_prepare_data(
            args.data,
            feature_processor=add_technical_features,
        )

    logger.info(f"Data shape: {data.shape}")

    # Train agent(s)
    if args.agent == "all":
        agents_to_train = list(AGENT_CLASSES.keys())
    else:
        agents_to_train = [args.agent]

    results = {}
    for agent_name in agents_to_train:
        try:
            result = train_agent(agent_name, data, args)
            results[agent_name] = result
            logger.info(f"Completed training {agent_name}: best_val_loss={result['best_val_loss']:.4f}")
        except Exception as e:
            logger.error(f"Error training {agent_name}: {e}")
            results[agent_name] = {"error": str(e)}

    # Summary
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)

    for agent_name, result in results.items():
        if "error" in result:
            logger.error(f"{agent_name}: FAILED - {result['error']}")
        else:
            logger.info(
                f"{agent_name}: best_val_loss={result['best_val_loss']:.4f}, "
                f"test_metrics={result.get('test_metrics', {})}"
            )


if __name__ == "__main__":
    main()
