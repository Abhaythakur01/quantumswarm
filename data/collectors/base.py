"""
Base class for all data collectors.
Defines the interface that all data sources must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd
from loguru import logger


@dataclass
class OHLCV:
    """Standard OHLCV data structure."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "symbol": self.symbol,
        }


class BaseCollector(ABC):
    """
    Abstract base class for data collectors.
    All data source implementations must inherit from this.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logger.bind(collector=name)

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC/USDT')
            start_date: Start of data range
            end_date: End of data range
            interval: Time interval ('1d', '1h', '15m', etc.)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        pass

    @abstractmethod
    def fetch_multiple(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading symbols
            start_date: Start of data range
            end_date: End of data range
            interval: Time interval

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> list[str]:
        """Get list of available symbols from this data source."""
        pass

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate that a DataFrame has the required OHLCV columns.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid, raises ValueError otherwise
        """
        required_columns = {"timestamp", "open", "high", "low", "close", "volume"}

        if df.empty:
            self.logger.warning("DataFrame is empty")
            return False

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check for NaN values
        nan_counts = df[list(required_columns)].isna().sum()
        if nan_counts.any():
            self.logger.warning(f"NaN values found: {nan_counts[nan_counts > 0].to_dict()}")

        return True

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize OHLCV DataFrame.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df

        df = df.copy()

        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Forward fill missing values (max 3 days)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        df[numeric_cols] = df[numeric_cols].ffill(limit=3)

        # Drop remaining NaN rows
        df = df.dropna(subset=numeric_cols)

        return df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return columns to DataFrame."""
        df = df.copy()
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        return df


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0.0

    def wait(self):
        """Wait if necessary to respect rate limits."""
        import time

        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


# Import numpy for calculations
import numpy as np
