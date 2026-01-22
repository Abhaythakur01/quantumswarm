"""
Dataset classes for training trading agents.
Provides efficient data loading for time series financial data.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from loguru import logger


@dataclass
class DatasetConfig:
    """Configuration for trading datasets."""
    sequence_length: int = 60
    prediction_horizon: int = 1
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    normalize: bool = True
    feature_columns: Optional[list[str]] = None
    target_column: str = "returns"
    label_threshold: float = 0.0  # For classification: returns > threshold = BUY


class TradingDataset(Dataset):
    """
    PyTorch Dataset for time series trading data.

    Creates sequences of features and corresponding labels for training.
    Supports both regression (predict returns) and classification (predict action).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: DatasetConfig,
        mode: str = "classification",  # 'classification' or 'regression'
        transform: Optional[Callable] = None,
    ):
        """
        Initialize trading dataset.

        Args:
            data: DataFrame with features and target
            config: Dataset configuration
            mode: 'classification' for action labels, 'regression' for returns
            transform: Optional transform to apply to features
        """
        self.config = config
        self.mode = mode
        self.transform = transform
        self.logger = logger.bind(dataset="trading")

        # Select feature columns
        if config.feature_columns:
            feature_cols = [c for c in config.feature_columns if c in data.columns]
        else:
            # Use all numeric columns except target
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if config.target_column in feature_cols:
                feature_cols.remove(config.target_column)

        self.feature_columns = feature_cols
        self.num_features = len(feature_cols)

        # Extract features and targets
        self.features = data[feature_cols].values.astype(np.float32)

        # Handle target column
        if config.target_column in data.columns:
            self.targets = data[config.target_column].values.astype(np.float32)
        else:
            # Calculate returns if not present
            if "close" in data.columns:
                self.targets = data["close"].pct_change(config.prediction_horizon).shift(-config.prediction_horizon).values.astype(np.float32)
            else:
                self.targets = np.zeros(len(data), dtype=np.float32)

        # Normalize features
        if config.normalize:
            self._normalize_features()

        # Calculate valid indices (accounting for sequence length and prediction horizon)
        self.valid_indices = self._get_valid_indices()

        self.logger.info(
            f"Created dataset: {len(self.valid_indices)} samples, "
            f"{self.num_features} features, mode={mode}"
        )

    def _normalize_features(self):
        """Normalize features to zero mean, unit variance."""
        self.feature_mean = np.nanmean(self.features, axis=0)
        self.feature_std = np.nanstd(self.features, axis=0)
        self.feature_std[self.feature_std == 0] = 1.0  # Avoid division by zero

        self.features = (self.features - self.feature_mean) / self.feature_std

        # Handle NaN values
        self.features = np.nan_to_num(self.features, nan=0.0)

    def _get_valid_indices(self) -> np.ndarray:
        """Get indices where we have complete sequences and valid targets."""
        # Need sequence_length history and prediction_horizon future
        start_idx = self.config.sequence_length
        end_idx = len(self.features) - self.config.prediction_horizon

        # Filter out indices with NaN targets
        valid = []
        for i in range(start_idx, end_idx):
            if not np.isnan(self.targets[i + self.config.prediction_horizon - 1]):
                valid.append(i)

        return np.array(valid)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - features: (sequence_length, num_features)
            - target: scalar (regression) or int (classification)
            - returns: actual return value
        """
        actual_idx = self.valid_indices[idx]

        # Get sequence
        start = actual_idx - self.config.sequence_length
        end = actual_idx
        features = self.features[start:end]

        # Get target
        target_idx = actual_idx + self.config.prediction_horizon - 1
        returns = self.targets[target_idx]

        # Apply transform
        if self.transform:
            features = self.transform(features)

        # Convert to tensors
        features = torch.FloatTensor(features)

        if self.mode == "classification":
            # Convert returns to class labels
            # 0: HOLD (small returns), 1: BUY (positive), 2: SELL (negative)
            threshold = self.config.label_threshold
            if returns > threshold:
                label = 1  # BUY
            elif returns < -threshold:
                label = 2  # SELL
            else:
                label = 0  # HOLD
            target = torch.LongTensor([label]).squeeze()
        else:
            target = torch.FloatTensor([returns]).squeeze()

        return {
            "features": features,
            "target": target,
            "returns": torch.FloatTensor([returns]).squeeze(),
            "index": actual_idx,
        }

    def get_normalization_params(self) -> dict:
        """Get normalization parameters for inference."""
        return {
            "mean": self.feature_mean,
            "std": self.feature_std,
        }


class MultiAssetDataset(Dataset):
    """
    Dataset for training on multiple assets simultaneously.
    Useful for learning cross-asset patterns.
    """

    def __init__(
        self,
        data_dict: dict[str, pd.DataFrame],
        config: DatasetConfig,
        mode: str = "classification",
    ):
        """
        Initialize multi-asset dataset.

        Args:
            data_dict: Dictionary mapping symbol to DataFrame
            config: Dataset configuration
            mode: 'classification' or 'regression'
        """
        self.config = config
        self.mode = mode
        self.symbols = list(data_dict.keys())

        # Create individual datasets
        self.datasets = {
            symbol: TradingDataset(df, config, mode)
            for symbol, df in data_dict.items()
        }

        # Build global index mapping
        self._build_index_map()

        self.logger.info(
            f"Created multi-asset dataset: {len(self.symbols)} assets, "
            f"{len(self)} total samples"
        )

    def _build_index_map(self):
        """Build mapping from global index to (symbol, local_index)."""
        self.index_map = []
        for symbol in self.symbols:
            dataset = self.datasets[symbol]
            for local_idx in range(len(dataset)):
                self.index_map.append((symbol, local_idx))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> dict:
        symbol, local_idx = self.index_map[idx]
        sample = self.datasets[symbol][local_idx]
        sample["symbol"] = symbol
        return sample


class SequenceSampler(Sampler):
    """
    Sampler that respects temporal ordering within sequences.
    Useful for time series to avoid look-ahead bias.
    """

    def __init__(
        self,
        dataset: TradingDataset,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            # Shuffle but maintain some temporal locality
            # Split into chunks and shuffle within chunks
            chunk_size = max(1, len(indices) // 10)
            chunks = [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]
            np.random.shuffle(chunks)
            indices = [idx for chunk in chunks for idx in chunk]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class RegimeDataset(Dataset):
    """
    Dataset that includes market regime labels.
    Useful for training regime-aware agents like LiquidAgent.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: DatasetConfig,
        regime_column: Optional[str] = None,
    ):
        """
        Initialize regime dataset.

        Args:
            data: DataFrame with features
            config: Dataset configuration
            regime_column: Column containing regime labels (auto-detected if None)
        """
        self.base_dataset = TradingDataset(data, config, mode="classification")
        self.config = config

        # Get or compute regime labels
        if regime_column and regime_column in data.columns:
            self.regimes = data[regime_column].values
        else:
            self.regimes = self._detect_regimes(data)

    def _detect_regimes(self, data: pd.DataFrame) -> np.ndarray:
        """
        Automatically detect market regimes based on price action.

        Regimes:
        0: Trending Up
        1: Trending Down
        2: Ranging
        3: High Volatility
        """
        regimes = np.zeros(len(data), dtype=np.int64)

        if "close" not in data.columns:
            return regimes

        close = data["close"].values

        # Calculate rolling metrics
        window = 20
        returns = np.diff(close, prepend=close[0]) / np.maximum(close, 1e-8)

        for i in range(window, len(data)):
            window_returns = returns[i-window:i]
            mean_return = np.mean(window_returns)
            volatility = np.std(window_returns)

            # Classify regime
            if volatility > 0.03:  # High volatility
                regimes[i] = 3
            elif mean_return > 0.001:  # Trending up
                regimes[i] = 0
            elif mean_return < -0.001:  # Trending down
                regimes[i] = 1
            else:  # Ranging
                regimes[i] = 2

        return regimes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.base_dataset[idx]
        actual_idx = self.base_dataset.valid_indices[idx]
        sample["regime"] = torch.LongTensor([self.regimes[actual_idx]]).squeeze()
        return sample


def create_dataloaders(
    data: pd.DataFrame,
    config: DatasetConfig,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "classification",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data: Full DataFrame
        config: Dataset configuration
        batch_size: Batch size
        num_workers: Number of worker processes
        mode: 'classification' or 'regression'

    Returns:
        train_loader, val_loader, test_loader
    """
    n = len(data)
    train_end = int(n * config.train_split)
    val_end = int(n * (config.train_split + config.val_split))

    # Split data temporally (no shuffling across splits)
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()

    # Create datasets
    train_dataset = TradingDataset(train_data, config, mode)
    val_dataset = TradingDataset(val_data, config, mode)
    test_dataset = TradingDataset(test_data, config, mode)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def load_and_prepare_data(
    data_path: str | Path,
    feature_processor: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Load data from file and prepare for training.

    Args:
        data_path: Path to data file (CSV, Parquet, etc.)
        feature_processor: Optional function to add features

    Returns:
        Prepared DataFrame
    """
    data_path = Path(data_path)

    # Load based on file type
    if data_path.suffix == ".csv":
        data = pd.read_csv(data_path, parse_dates=["timestamp"] if "timestamp" in pd.read_csv(data_path, nrows=1).columns else None)
    elif data_path.suffix == ".parquet":
        data = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file type: {data_path.suffix}")

    # Sort by timestamp
    if "timestamp" in data.columns:
        data = data.sort_values("timestamp").reset_index(drop=True)

    # Apply feature processor
    if feature_processor:
        data = feature_processor(data)

    # Calculate returns if not present
    if "returns" not in data.columns and "close" in data.columns:
        data["returns"] = data["close"].pct_change()

    # Drop NaN rows
    data = data.dropna().reset_index(drop=True)

    logger.info(f"Loaded data: {len(data)} rows, {len(data.columns)} columns")

    return data


if __name__ == "__main__":
    # Test the datasets
    print("Testing Dataset Classes...")
    print("=" * 50)

    # Create synthetic data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))

    data = pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 + np.random.randn(n) * 0.01),
        "high": prices * (1 + abs(np.random.randn(n) * 0.02)),
        "low": prices * (1 - abs(np.random.randn(n) * 0.02)),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, n),
        "returns": np.concatenate([[0], np.diff(prices) / prices[:-1]]),
        "rsi_14": np.random.rand(n) * 100,
        "macd": np.random.randn(n) * 0.1,
    })

    # Test TradingDataset
    print("\nTesting TradingDataset...")
    config = DatasetConfig(
        sequence_length=30,
        feature_columns=["returns", "rsi_14", "macd"],
    )
    dataset = TradingDataset(data, config)
    print(f"Dataset length: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample features shape: {sample['features'].shape}")
    print(f"Sample target: {sample['target']}")

    # Test DataLoaders
    print("\nTesting DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data, config, batch_size=16
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get a batch
    batch = next(iter(train_loader))
    print(f"Batch features shape: {batch['features'].shape}")
    print(f"Batch targets shape: {batch['target'].shape}")

    # Test RegimeDataset
    print("\nTesting RegimeDataset...")
    regime_dataset = RegimeDataset(data, config)
    regime_sample = regime_dataset[0]
    print(f"Regime: {regime_sample['regime']}")

    print("\nDataset tests passed!")
