"""
Training infrastructure for the multi-agent trading system.

Modules:
- datasets: Data loading and preprocessing
- trainer: Training loops and optimization
- checkpointing: Model saving and loading
- experiment: Experiment tracking and logging
"""

from .datasets import (
    DatasetConfig,
    TradingDataset,
    MultiAssetDataset,
    RegimeDataset,
    create_dataloaders,
    load_and_prepare_data,
)

from .trainer import (
    Trainer,
    TrainerConfig,
    TrainingState,
    MultiAgentTrainer,
)

from .checkpointing import (
    ModelRegistry,
    CheckpointManager,
    ModelExporter,
    ModelMetadata,
)

from .experiment import (
    ExperimentConfig,
    ExperimentTracker,
    MetricsAggregator,
    create_experiment_name,
    log_training_run,
)

__all__ = [
    # Datasets
    "DatasetConfig",
    "TradingDataset",
    "MultiAssetDataset",
    "RegimeDataset",
    "create_dataloaders",
    "load_and_prepare_data",
    # Trainer
    "Trainer",
    "TrainerConfig",
    "TrainingState",
    "MultiAgentTrainer",
    # Checkpointing
    "ModelRegistry",
    "CheckpointManager",
    "ModelExporter",
    "ModelMetadata",
    # Experiment
    "ExperimentConfig",
    "ExperimentTracker",
    "MetricsAggregator",
    "create_experiment_name",
    "log_training_run",
]
