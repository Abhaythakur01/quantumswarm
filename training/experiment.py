"""
Experiment tracking and logging utilities.
Supports Weights & Biases, TensorBoard, and local logging.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import json
import numpy as np
import torch
from loguru import logger


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    name: str
    project: str = "multiagent-trading"
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    log_dir: str = "logs"

    # Backend selection
    use_wandb: bool = False
    use_tensorboard: bool = False

    # Logging settings
    log_every_n_steps: int = 10
    log_model: bool = True
    log_gradients: bool = False


class BaseLogger(ABC):
    """Abstract base class for experiment loggers."""

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int):
        """Log metrics at a given step."""
        pass

    @abstractmethod
    def log_hyperparams(self, params: dict):
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_artifact(self, path: str, name: str, artifact_type: str):
        """Log an artifact (file)."""
        pass

    @abstractmethod
    def finish(self):
        """Finish logging and cleanup."""
        pass


class LocalLogger(BaseLogger):
    """Simple local file-based logger."""

    def __init__(self, log_dir: str | Path, experiment_name: str):
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.params_file = self.log_dir / "params.json"

        self.logger = logger.bind(experiment=experiment_name)

    def log_metrics(self, metrics: dict, step: int):
        """Append metrics to JSONL file."""
        entry = {"step": step, "timestamp": datetime.now().isoformat(), **metrics}

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def log_hyperparams(self, params: dict):
        """Save hyperparameters to JSON file."""
        with open(self.params_file, "w") as f:
            json.dump(params, f, indent=2, default=str)

    def log_artifact(self, path: str, name: str, artifact_type: str):
        """Copy artifact to log directory."""
        import shutil
        dest = self.log_dir / "artifacts" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest)

    def finish(self):
        """Finish logging."""
        self.logger.info(f"Logs saved to {self.log_dir}")


class WandbLogger(BaseLogger):
    """Weights & Biases logger."""

    def __init__(
        self,
        project: str,
        name: str,
        config: dict,
        tags: list[str] = None,
        notes: str = "",
    ):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError("wandb not installed. Install with: pip install wandb")

        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            reinit=True,
        )

        self.logger = logger.bind(experiment=name)
        self.logger.info(f"W&B run initialized: {self.run.url}")

    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to W&B."""
        self.wandb.log(metrics, step=step)

    def log_hyperparams(self, params: dict):
        """Update W&B config."""
        self.wandb.config.update(params)

    def log_artifact(self, path: str, name: str, artifact_type: str):
        """Log artifact to W&B."""
        artifact = self.wandb.Artifact(name, type=artifact_type)
        artifact.add_file(path)
        self.run.log_artifact(artifact)

    def log_model(self, model: torch.nn.Module, name: str = "model"):
        """Log model to W&B."""
        self.wandb.watch(model, log="all")

    def log_table(self, name: str, data: list[list], columns: list[str]):
        """Log a table to W&B."""
        table = self.wandb.Table(data=data, columns=columns)
        self.wandb.log({name: table})

    def log_image(self, name: str, image: Any):
        """Log an image to W&B."""
        self.wandb.log({name: self.wandb.Image(image)})

    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """Log histogram to W&B."""
        self.wandb.log({name: self.wandb.Histogram(values)}, step=step)

    def finish(self):
        """Finish W&B run."""
        self.wandb.finish()


class TensorBoardLogger(BaseLogger):
    """TensorBoard logger."""

    def __init__(self, log_dir: str | Path, experiment_name: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("tensorboard not installed. Install with: pip install tensorboard")

        self.log_dir = Path(log_dir) / experiment_name
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.logger = logger.bind(experiment=experiment_name)

    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(name, value, step)

    def log_hyperparams(self, params: dict):
        """Log hyperparameters."""
        # Flatten nested dicts
        flat_params = {}
        for key, value in params.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    flat_params[f"{key}/{k}"] = v
            else:
                flat_params[key] = value

        self.writer.add_hparams(flat_params, {})

    def log_artifact(self, path: str, name: str, artifact_type: str):
        """TensorBoard doesn't support artifacts directly."""
        pass

    def log_histogram(self, name: str, values: np.ndarray, step: int):
        """Log histogram to TensorBoard."""
        self.writer.add_histogram(name, values, step)

    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Log model graph to TensorBoard."""
        self.writer.add_graph(model, input_tensor)

    def finish(self):
        """Close TensorBoard writer."""
        self.writer.close()


class ExperimentTracker:
    """
    Unified experiment tracker that can use multiple backends.
    """

    def __init__(self, config: ExperimentConfig, hyperparams: dict):
        """
        Initialize experiment tracker.

        Args:
            config: Experiment configuration
            hyperparams: Hyperparameters to log
        """
        self.config = config
        self.loggers: list[BaseLogger] = []
        self.step = 0

        # Always use local logger
        self.local_logger = LocalLogger(config.log_dir, config.name)
        self.loggers.append(self.local_logger)

        # Add W&B if enabled
        if config.use_wandb:
            try:
                wandb_logger = WandbLogger(
                    project=config.project,
                    name=config.name,
                    config=hyperparams,
                    tags=config.tags,
                    notes=config.notes,
                )
                self.loggers.append(wandb_logger)
            except ImportError:
                logger.warning("W&B not available, skipping")

        # Add TensorBoard if enabled
        if config.use_tensorboard:
            try:
                tb_logger = TensorBoardLogger(config.log_dir, config.name)
                self.loggers.append(tb_logger)
            except ImportError:
                logger.warning("TensorBoard not available, skipping")

        # Log hyperparameters
        self.log_hyperparams(hyperparams)

    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to all backends."""
        if step is None:
            step = self.step
            self.step += 1

        for log in self.loggers:
            log.log_metrics(metrics, step)

    def log_hyperparams(self, params: dict):
        """Log hyperparameters to all backends."""
        for log in self.loggers:
            log.log_hyperparams(params)

    def log_artifact(self, path: str, name: str, artifact_type: str = "file"):
        """Log artifact to all backends."""
        for log in self.loggers:
            log.log_artifact(path, name, artifact_type)

    def log_model_checkpoint(self, path: str, name: str = "checkpoint"):
        """Log a model checkpoint."""
        self.log_artifact(path, name, "model")

    def finish(self):
        """Finish all loggers."""
        for log in self.loggers:
            log.finish()


class MetricsAggregator:
    """
    Aggregates metrics over batches for epoch-level logging.
    """

    def __init__(self):
        self.metrics: dict[str, list] = {}
        self.counts: dict[str, int] = {}

    def update(self, metrics: dict, count: int = 1):
        """Add batch metrics."""
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
                self.counts[name] = 0

            self.metrics[name].append(value * count)
            self.counts[name] += count

    def compute(self) -> dict:
        """Compute averaged metrics."""
        result = {}
        for name, values in self.metrics.items():
            total = sum(values)
            count = self.counts[name]
            result[name] = total / count if count > 0 else 0.0
        return result

    def reset(self):
        """Reset aggregator."""
        self.metrics = {}
        self.counts = {}


def create_experiment_name(
    agent_name: str,
    dataset_name: str = "default",
    suffix: Optional[str] = None,
) -> str:
    """Create a standardized experiment name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{agent_name}_{dataset_name}_{timestamp}"
    if suffix:
        name = f"{name}_{suffix}"
    return name


def log_training_run(
    tracker: ExperimentTracker,
    epoch: int,
    train_metrics: dict,
    val_metrics: Optional[dict] = None,
    lr: Optional[float] = None,
):
    """
    Convenience function to log a training epoch.

    Args:
        tracker: Experiment tracker
        epoch: Current epoch
        train_metrics: Training metrics
        val_metrics: Validation metrics
        lr: Current learning rate
    """
    metrics = {}

    # Add train metrics with prefix
    for name, value in train_metrics.items():
        metrics[f"train/{name}"] = value

    # Add val metrics with prefix
    if val_metrics:
        for name, value in val_metrics.items():
            metrics[f"val/{name}"] = value

    # Add learning rate
    if lr is not None:
        metrics["learning_rate"] = lr

    metrics["epoch"] = epoch

    tracker.log_metrics(metrics, step=epoch)


if __name__ == "__main__":
    # Test experiment tracking
    print("Testing Experiment Tracking...")
    print("=" * 50)

    # Create config
    config = ExperimentConfig(
        name="test_experiment",
        project="test-project",
        tags=["test", "debug"],
        use_wandb=False,  # Set to True if wandb is configured
        use_tensorboard=False,
    )

    # Create tracker
    hyperparams = {
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model": {
            "hidden_dim": 64,
            "num_layers": 2,
        },
    }

    tracker = ExperimentTracker(config, hyperparams)

    # Log some metrics
    print("\nLogging metrics...")
    for epoch in range(5):
        train_metrics = {"loss": 1.0 - epoch * 0.1, "accuracy": 0.5 + epoch * 0.05}
        val_metrics = {"loss": 1.1 - epoch * 0.1, "accuracy": 0.48 + epoch * 0.05}

        log_training_run(tracker, epoch, train_metrics, val_metrics, lr=1e-4)

    # Test metrics aggregator
    print("\nTesting MetricsAggregator...")
    aggregator = MetricsAggregator()

    for batch in range(10):
        aggregator.update({"loss": 0.5 + batch * 0.1, "accuracy": 0.8}, count=32)

    avg_metrics = aggregator.compute()
    print(f"Averaged metrics: {avg_metrics}")

    # Finish
    tracker.finish()

    # Cleanup
    import shutil
    shutil.rmtree("logs/test_experiment", ignore_errors=True)

    print("\nExperiment tracking tests passed!")
