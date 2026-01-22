"""
Trainer class for training trading agents.
Handles training loops, validation, early stopping, and logging.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Any
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
from loguru import logger

from agents.base import BaseNeuralAgent


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""
    # Training parameters
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    gradient_clip: float = 1.0

    # Scheduler
    scheduler_type: str = "plateau"  # 'plateau', 'cosine', 'none'
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best: bool = True
    save_every: int = 10  # Save every N epochs

    # Logging
    log_every: int = 10  # Log every N batches
    use_wandb: bool = False
    wandb_project: str = "multiagent-trading"
    wandb_run_name: Optional[str] = None

    # Device
    device: str = "auto"

    # Mixed precision
    use_amp: bool = False


@dataclass
class TrainingState:
    """Tracks training state for checkpointing and resumption."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    patience_counter: int = 0
    history: dict = field(default_factory=lambda: {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "learning_rate": [],
    })


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Generic trainer for neural network trading agents.

    Features:
    - Flexible training loop
    - Validation and early stopping
    - Checkpointing and model saving
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training (optional)
    - Weights & Biases integration (optional)
    """

    def __init__(
        self,
        agent: BaseNeuralAgent,
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        """
        Initialize trainer.

        Args:
            agent: The agent to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
        """
        self.agent = agent
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger = logger.bind(trainer="main")

        # Set device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Move agent to device
        if agent.model is not None:
            agent.model.to(self.device)

        # Setup optimizer
        self._setup_optimizer()

        # Setup scheduler
        self._setup_scheduler()

        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        ) if config.early_stopping else None

        # Training state
        self.state = TrainingState()

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None

        # Weights & Biases
        self.wandb_run = None
        if config.use_wandb:
            self._setup_wandb()

        self.logger.info(f"Trainer initialized on {self.device}")

    def _setup_optimizer(self):
        """Setup optimizer."""
        self.agent.setup_optimizer(
            optimizer_class=torch.optim.AdamW,
            weight_decay=self.config.weight_decay,
        )
        # Update learning rate
        for param_group in self.agent.optimizer.param_groups:
            param_group["lr"] = self.config.learning_rate

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.agent.optimizer,
                mode="min",
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                verbose=True,
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.agent.optimizer,
                T_0=10,
                T_mult=2,
            )
        else:
            self.scheduler = None

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name or f"{self.agent.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "agent": self.agent.get_config(),
                    "trainer": {
                        "epochs": self.config.epochs,
                        "learning_rate": self.config.learning_rate,
                        "batch_size": self.config.batch_size,
                    },
                },
            )
            self.logger.info("Weights & Biases initialized")
        except ImportError:
            self.logger.warning("wandb not installed, skipping W&B logging")
            self.wandb_run = None

    def train(self) -> dict:
        """
        Run full training loop.

        Returns:
            Dictionary with training history and final metrics
        """
        self.logger.info(f"Starting training for {self.config.epochs} epochs")

        for epoch in range(self.state.epoch, self.config.epochs):
            self.state.epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch()

            # Validation
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate()

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get("loss", train_metrics["loss"]))
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.agent.optimizer.param_groups[0]["lr"]

            # Log metrics
            self._log_epoch(epoch, train_metrics, val_metrics, current_lr)

            # Update history
            self.state.history["train_loss"].append(train_metrics["loss"])
            self.state.history["learning_rate"].append(current_lr)
            if val_metrics:
                self.state.history["val_loss"].append(val_metrics["loss"])
                if "accuracy" in val_metrics:
                    self.state.history["val_accuracy"].append(val_metrics["accuracy"])

            # Checkpointing
            val_loss = val_metrics.get("loss", train_metrics["loss"])
            is_best = val_loss < self.state.best_val_loss
            if is_best:
                self.state.best_val_loss = val_loss
                self.state.patience_counter = 0
            else:
                self.state.patience_counter += 1

            if self.config.save_best and is_best:
                self.save_checkpoint("best.pt")

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Final evaluation on test set
        test_metrics = {}
        if self.test_loader is not None:
            test_metrics = self._evaluate(self.test_loader)
            self.logger.info(f"Test metrics: {test_metrics}")

        # Save final checkpoint
        self.save_checkpoint("final.pt")

        # Close wandb
        if self.wandb_run is not None:
            import wandb
            wandb.finish()

        return {
            "history": self.state.history,
            "best_val_loss": self.state.best_val_loss,
            "test_metrics": test_metrics,
        }

    def _train_epoch(self) -> dict:
        """Run one training epoch."""
        self.agent.train_mode()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.state.epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)

            # Prepare batch for agent
            agent_batch = {
                "features": batch["features"],
                "actions": batch["target"],
            }
            if "returns" in batch:
                agent_batch["returns"] = batch["returns"]
            if "regime" in batch:
                agent_batch["regimes"] = batch["regime"]

            # Training step with optional mixed precision
            if self.config.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    metrics = self.agent.training_step(agent_batch)
            else:
                metrics = self.agent.training_step(agent_batch)

            loss = metrics["loss"]
            total_loss += loss
            num_batches += 1
            self.state.global_step += 1

            # Calculate accuracy for classification
            with torch.no_grad():
                outputs = self.agent.model(batch["features"])
                if "action_logits" in outputs:
                    preds = outputs["action_logits"].argmax(dim=-1)
                    correct = (preds == batch["target"]).sum().item()
                    total_correct += correct
                    total_samples += batch["target"].shape[0]

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "avg_loss": f"{total_loss / num_batches:.4f}",
            })

            # Log to wandb
            if self.wandb_run is not None and batch_idx % self.config.log_every == 0:
                import wandb
                wandb.log({
                    "train/batch_loss": loss,
                    "train/learning_rate": self.agent.optimizer.param_groups[0]["lr"],
                    "global_step": self.state.global_step,
                })

        avg_loss = total_loss / num_batches
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

    def _validate(self) -> dict:
        """Run validation."""
        return self._evaluate(self.val_loader)

    def _evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate on a dataset."""
        self.agent.eval_mode()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                batch = self._move_batch_to_device(batch)

                # Forward pass
                outputs = self.agent.model(batch["features"])

                # Calculate loss
                if "action_logits" in outputs:
                    loss = nn.CrossEntropyLoss()(outputs["action_logits"], batch["target"])
                    preds = outputs["action_logits"].argmax(dim=-1)
                    correct = (preds == batch["target"]).sum().item()
                    total_correct += correct
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(batch["target"].cpu().numpy())
                else:
                    loss = nn.MSELoss()(outputs.squeeze(), batch["target"])

                total_loss += loss.item()
                total_samples += batch["target"].shape[0]

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # Calculate per-class accuracy
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
        }

        if all_preds and all_targets:
            preds_arr = np.array(all_preds)
            targets_arr = np.array(all_targets)
            for c in range(3):  # HOLD, BUY, SELL
                mask = targets_arr == c
                if mask.sum() > 0:
                    class_acc = (preds_arr[mask] == c).mean()
                    metrics[f"accuracy_class_{c}"] = class_acc

        return metrics

    def _move_batch_to_device(self, batch: dict) -> dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        lr: float,
    ):
        """Log epoch metrics."""
        msg = f"Epoch {epoch + 1}/{self.config.epochs} | "
        msg += f"Train Loss: {train_metrics['loss']:.4f}"

        if "accuracy" in train_metrics:
            msg += f" | Train Acc: {train_metrics['accuracy']:.4f}"

        if val_metrics:
            msg += f" | Val Loss: {val_metrics['loss']:.4f}"
            if "accuracy" in val_metrics:
                msg += f" | Val Acc: {val_metrics['accuracy']:.4f}"

        msg += f" | LR: {lr:.2e}"

        self.logger.info(msg)

        # Log to wandb
        if self.wandb_run is not None:
            import wandb
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics.get("accuracy", 0),
                "learning_rate": lr,
            }
            if val_metrics:
                log_dict["val/loss"] = val_metrics["loss"]
                log_dict["val/accuracy"] = val_metrics.get("accuracy", 0)
            wandb.log(log_dict)

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "model_state_dict": self.agent.model.state_dict(),
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.state.best_val_loss,
            "history": self.state.history,
            "config": {
                "agent": self.agent.get_config(),
                "trainer": {
                    "epochs": self.config.epochs,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                },
            },
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str | Path):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.agent.model.load_state_dict(checkpoint["model_state_dict"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.state.epoch = checkpoint["epoch"]
        self.state.global_step = checkpoint["global_step"]
        self.state.best_val_loss = checkpoint["best_val_loss"]
        self.state.history = checkpoint["history"]

        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")

    def get_training_summary(self) -> dict:
        """Get summary of training results."""
        return {
            "agent_name": self.agent.name,
            "epochs_trained": self.state.epoch + 1,
            "best_val_loss": self.state.best_val_loss,
            "final_train_loss": self.state.history["train_loss"][-1] if self.state.history["train_loss"] else None,
            "final_val_loss": self.state.history["val_loss"][-1] if self.state.history["val_loss"] else None,
        }


class MultiAgentTrainer:
    """
    Trainer for training multiple agents together or sequentially.
    """

    def __init__(
        self,
        agents: list[BaseNeuralAgent],
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Initialize multi-agent trainer.

        Args:
            agents: List of agents to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.agents = agents
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trainers: dict[str, Trainer] = {}
        self.logger = logger.bind(trainer="multi")

    def train_sequential(self) -> dict[str, dict]:
        """Train agents one after another."""
        results = {}

        for agent in self.agents:
            self.logger.info(f"Training agent: {agent.name}")

            trainer = Trainer(
                agent=agent,
                config=self.config,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
            )
            self.trainers[agent.name] = trainer

            result = trainer.train()
            results[agent.name] = result

        return results

    def train_parallel(self) -> dict[str, dict]:
        """
        Train agents in parallel (requires multiple GPUs or CPU).
        Simplified version - trains on same data simultaneously.
        """
        # For true parallel training, would need torch.distributed
        # This is a simplified version that trains sequentially but
        # could be extended for distributed training
        return self.train_sequential()


if __name__ == "__main__":
    # Test the trainer
    print("Testing Trainer...")
    print("=" * 50)

    # Create synthetic data and dataloader
    from torch.utils.data import TensorDataset, DataLoader

    n_samples = 500
    seq_len = 30
    n_features = 15

    X = torch.randn(n_samples, seq_len, n_features)
    y = torch.randint(0, 3, (n_samples,))

    dataset = TensorDataset(X, y)

    def collate_fn(batch):
        features = torch.stack([b[0] for b in batch])
        targets = torch.stack([b[1] for b in batch])
        return {"features": features, "target": targets}

    train_loader = DataLoader(dataset[:400], batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset[400:], batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Create a simple agent for testing
    from agents import MambaAgent

    agent = MambaAgent(
        input_dim=n_features,
        d_model=32,
        d_state=8,
        n_layers=2,
        sequence_length=seq_len,
    )

    # Create trainer
    config = TrainerConfig(
        epochs=3,
        learning_rate=1e-3,
        batch_size=32,
        early_stopping=False,
        save_every=5,
        use_wandb=False,
    )

    trainer = Trainer(
        agent=agent,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Train
    print("\nStarting training...")
    results = trainer.train()

    print(f"\nTraining complete!")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Training history length: {len(results['history']['train_loss'])}")

    # Test checkpoint loading
    print("\nTesting checkpoint loading...")
    trainer.load_checkpoint(trainer.checkpoint_dir / "final.pt")
    print("Checkpoint loaded successfully!")

    print("\nTrainer tests passed!")
