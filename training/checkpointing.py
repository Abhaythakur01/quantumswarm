"""
Checkpointing and model management utilities.
Handles saving, loading, and managing trained models.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Type
import json
import shutil
import torch
from loguru import logger

from agents.base import BaseNeuralAgent


@dataclass
class ModelMetadata:
    """Metadata for a saved model."""
    agent_name: str
    agent_class: str
    created_at: str
    trained_epochs: int
    best_val_loss: float
    config: dict
    metrics: dict


class ModelRegistry:
    """
    Registry for managing trained models.
    Handles saving, loading, versioning, and model discovery.
    """

    def __init__(self, base_dir: str | Path = "models"):
        """
        Initialize model registry.

        Args:
            base_dir: Base directory for storing models
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(component="registry")

        # Index file for quick lookup
        self.index_file = self.base_dir / "index.json"
        self._load_index()

    def _load_index(self):
        """Load or create the model index."""
        if self.index_file.exists():
            with open(self.index_file) as f:
                self.index = json.load(f)
        else:
            self.index = {"models": {}}

    def _save_index(self):
        """Save the model index."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def register_model(
        self,
        agent: BaseNeuralAgent,
        version: Optional[str] = None,
        metrics: Optional[dict] = None,
        trained_epochs: int = 0,
        best_val_loss: float = float("inf"),
    ) -> str:
        """
        Register and save a trained model.

        Args:
            agent: The trained agent
            version: Version string (auto-generated if None)
            metrics: Training/evaluation metrics
            trained_epochs: Number of epochs trained
            best_val_loss: Best validation loss achieved

        Returns:
            Model ID
        """
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_id = f"{agent.name}_{version}"
        model_dir = self.base_dir / agent.name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = model_dir / "model.pt"
        agent.save(str(model_path))

        # Save metadata
        metadata = ModelMetadata(
            agent_name=agent.name,
            agent_class=agent.__class__.__name__,
            created_at=datetime.now().isoformat(),
            trained_epochs=trained_epochs,
            best_val_loss=best_val_loss,
            config=agent.get_config(),
            metrics=metrics or {},
        )

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.__dict__, f, indent=2)

        # Update index
        if agent.name not in self.index["models"]:
            self.index["models"][agent.name] = {}

        self.index["models"][agent.name][version] = {
            "model_id": model_id,
            "path": str(model_dir),
            "created_at": metadata.created_at,
            "best_val_loss": best_val_loss,
        }
        self._save_index()

        self.logger.info(f"Registered model: {model_id}")
        return model_id

    def load_model(
        self,
        agent_class: Type[BaseNeuralAgent],
        agent_name: str,
        version: str = "latest",
    ) -> BaseNeuralAgent:
        """
        Load a registered model.

        Args:
            agent_class: Class of the agent to instantiate
            agent_name: Name of the agent
            version: Version to load ('latest' for most recent)

        Returns:
            Loaded agent instance
        """
        if agent_name not in self.index["models"]:
            raise ValueError(f"No models found for agent: {agent_name}")

        versions = self.index["models"][agent_name]

        if version == "latest":
            # Get most recent version
            version = max(versions.keys())
        elif version == "best":
            # Get version with best validation loss
            version = min(versions.keys(), key=lambda v: versions[v]["best_val_loss"])

        if version not in versions:
            raise ValueError(f"Version {version} not found for {agent_name}")

        model_dir = Path(versions[version]["path"])

        # Load metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Instantiate agent with saved config
        config = metadata["config"]
        agent = agent_class(**config)

        # Load weights
        model_path = model_dir / "model.pt"
        agent.load(str(model_path))

        self.logger.info(f"Loaded model: {agent_name}/{version}")
        return agent

    def list_models(self, agent_name: Optional[str] = None) -> dict:
        """
        List registered models.

        Args:
            agent_name: Filter by agent name (None for all)

        Returns:
            Dictionary of models
        """
        if agent_name:
            return self.index["models"].get(agent_name, {})
        return self.index["models"]

    def get_best_model(self, agent_name: str) -> Optional[str]:
        """
        Get the version with best validation loss.

        Args:
            agent_name: Name of the agent

        Returns:
            Version string or None
        """
        if agent_name not in self.index["models"]:
            return None

        versions = self.index["models"][agent_name]
        if not versions:
            return None

        return min(versions.keys(), key=lambda v: versions[v]["best_val_loss"])

    def delete_model(self, agent_name: str, version: str):
        """
        Delete a model version.

        Args:
            agent_name: Name of the agent
            version: Version to delete
        """
        if agent_name not in self.index["models"]:
            return

        if version not in self.index["models"][agent_name]:
            return

        model_dir = Path(self.index["models"][agent_name][version]["path"])
        if model_dir.exists():
            shutil.rmtree(model_dir)

        del self.index["models"][agent_name][version]
        self._save_index()

        self.logger.info(f"Deleted model: {agent_name}/{version}")

    def cleanup_old_versions(self, agent_name: str, keep: int = 5):
        """
        Remove old model versions, keeping the N most recent.

        Args:
            agent_name: Name of the agent
            keep: Number of versions to keep
        """
        if agent_name not in self.index["models"]:
            return

        versions = self.index["models"][agent_name]
        if len(versions) <= keep:
            return

        # Sort by creation time
        sorted_versions = sorted(
            versions.keys(),
            key=lambda v: versions[v]["created_at"],
            reverse=True,
        )

        # Delete old versions
        for version in sorted_versions[keep:]:
            self.delete_model(agent_name, version)


class CheckpointManager:
    """
    Manages training checkpoints for a single training run.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        max_checkpoints: int = 5,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.logger = logger.bind(component="checkpoint")

    def save(
        self,
        agent: BaseNeuralAgent,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        is_best: bool = False,
        extra_state: Optional[dict] = None,
    ) -> Path:
        """
        Save a training checkpoint.

        Args:
            agent: The agent being trained
            optimizer: The optimizer
            epoch: Current epoch
            metrics: Training metrics
            scheduler: Optional learning rate scheduler
            is_best: Whether this is the best model so far
            extra_state: Additional state to save

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": agent.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "agent_config": agent.get_config(),
            "timestamp": datetime.now().isoformat(),
        }

        if extra_state:
            checkpoint["extra_state"] = extra_state

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f"Saved checkpoint: {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path}")

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Cleanup old checkpoints
        self._cleanup()

        return checkpoint_path

    def load(
        self,
        agent: BaseNeuralAgent,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Optional[str | Path] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> dict:
        """
        Load a training checkpoint.

        Args:
            agent: The agent to load into
            optimizer: The optimizer to load into
            checkpoint_path: Path to checkpoint (default: latest)
            scheduler: Optional scheduler to load into

        Returns:
            Dictionary with epoch and metrics
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        agent.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler and checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

        return {
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"],
            "extra_state": checkpoint.get("extra_state"),
        }

    def load_best(
        self,
        agent: BaseNeuralAgent,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> dict:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError("No best checkpoint found")

        checkpoint = torch.load(best_path, map_location="cpu")
        agent.model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return {
            "epoch": checkpoint["epoch"],
            "metrics": checkpoint["metrics"],
        }

    def _cleanup(self):
        """Remove old checkpoints beyond max_checkpoints."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )

        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            self.logger.debug(f"Removed old checkpoint: {oldest}")

    def list_checkpoints(self) -> list[dict]:
        """List all available checkpoints."""
        checkpoints = []

        for path in self.checkpoint_dir.glob("*.pt"):
            try:
                checkpoint = torch.load(path, map_location="cpu")
                checkpoints.append({
                    "path": str(path),
                    "epoch": checkpoint.get("epoch"),
                    "timestamp": checkpoint.get("timestamp"),
                    "metrics": checkpoint.get("metrics", {}),
                })
            except Exception as e:
                self.logger.warning(f"Could not load {path}: {e}")

        return sorted(checkpoints, key=lambda c: c.get("epoch", 0))


class ModelExporter:
    """
    Export trained models for production deployment.
    """

    def __init__(self, output_dir: str | Path = "exported_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(component="exporter")

    def export_torchscript(
        self,
        agent: BaseNeuralAgent,
        name: str,
        example_input: torch.Tensor,
    ) -> Path:
        """
        Export model as TorchScript for production.

        Args:
            agent: Trained agent
            name: Export name
            example_input: Example input tensor for tracing

        Returns:
            Path to exported model
        """
        agent.eval_mode()

        # Trace the model
        with torch.no_grad():
            traced = torch.jit.trace(agent.model, example_input)

        # Save
        export_path = self.output_dir / f"{name}.pt"
        traced.save(str(export_path))

        self.logger.info(f"Exported TorchScript model: {export_path}")
        return export_path

    def export_onnx(
        self,
        agent: BaseNeuralAgent,
        name: str,
        example_input: torch.Tensor,
        opset_version: int = 14,
    ) -> Path:
        """
        Export model to ONNX format.

        Args:
            agent: Trained agent
            name: Export name
            example_input: Example input tensor
            opset_version: ONNX opset version

        Returns:
            Path to exported model
        """
        agent.eval_mode()

        export_path = self.output_dir / f"{name}.onnx"

        torch.onnx.export(
            agent.model,
            example_input,
            str(export_path),
            opset_version=opset_version,
            input_names=["features"],
            output_names=["action_logits", "confidence", "position_size"],
            dynamic_axes={
                "features": {0: "batch_size", 1: "sequence_length"},
            },
        )

        self.logger.info(f"Exported ONNX model: {export_path}")
        return export_path

    def export_with_metadata(
        self,
        agent: BaseNeuralAgent,
        name: str,
        metrics: dict,
        feature_names: list[str],
    ) -> Path:
        """
        Export model with full metadata for deployment.

        Args:
            agent: Trained agent
            name: Export name
            metrics: Training metrics
            feature_names: List of feature names

        Returns:
            Path to export directory
        """
        export_dir = self.output_dir / name
        export_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = export_dir / "model.pt"
        torch.save(agent.model.state_dict(), model_path)

        # Save config
        config = {
            "agent_name": agent.name,
            "agent_class": agent.__class__.__name__,
            "config": agent.get_config(),
            "feature_names": feature_names,
            "metrics": metrics,
            "exported_at": datetime.now().isoformat(),
        }

        config_path = export_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Exported model with metadata: {export_dir}")
        return export_dir


if __name__ == "__main__":
    # Test checkpointing utilities
    print("Testing Checkpointing Utilities...")
    print("=" * 50)

    # Create a dummy agent
    from agents import MambaAgent

    agent = MambaAgent(
        input_dim=15,
        d_model=32,
        d_state=8,
        n_layers=2,
    )
    agent.setup_optimizer()

    # Test CheckpointManager
    print("\nTesting CheckpointManager...")
    manager = CheckpointManager("test_checkpoints", max_checkpoints=3)

    # Save some checkpoints
    for epoch in range(5):
        manager.save(
            agent=agent,
            optimizer=agent.optimizer,
            epoch=epoch,
            metrics={"loss": 1.0 - epoch * 0.1},
            is_best=(epoch == 3),
        )

    # List checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"Available checkpoints: {len(checkpoints)}")

    # Load latest
    state = manager.load(agent, agent.optimizer)
    print(f"Loaded epoch: {state['epoch']}")

    # Test ModelRegistry
    print("\nTesting ModelRegistry...")
    registry = ModelRegistry("test_models")

    # Register model
    model_id = registry.register_model(
        agent=agent,
        metrics={"test_accuracy": 0.75},
        trained_epochs=10,
        best_val_loss=0.5,
    )
    print(f"Registered: {model_id}")

    # List models
    models = registry.list_models()
    print(f"Registered models: {models}")

    # Load model
    loaded_agent = registry.load_model(MambaAgent, "mamba_ssm", "latest")
    print(f"Loaded agent: {loaded_agent.name}")

    # Test ModelExporter
    print("\nTesting ModelExporter...")
    exporter = ModelExporter("test_exports")

    # Export with metadata
    export_path = exporter.export_with_metadata(
        agent=agent,
        name="mamba_v1",
        metrics={"accuracy": 0.8},
        feature_names=["returns", "rsi", "macd"],
    )
    print(f"Exported to: {export_path}")

    # Cleanup
    import shutil
    shutil.rmtree("test_checkpoints", ignore_errors=True)
    shutil.rmtree("test_models", ignore_errors=True)
    shutil.rmtree("test_exports", ignore_errors=True)

    print("\nCheckpointing tests passed!")
