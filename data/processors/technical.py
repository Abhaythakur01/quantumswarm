"""
Technical indicators and feature engineering.
Calculates a comprehensive set of features from OHLCV data.
"""

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


class TechnicalFeatureEngine:
    """
    Calculates technical indicators and derived features from OHLCV data.

    Features calculated:
    - Trend indicators (MA, EMA, MACD)
    - Momentum indicators (RSI, Stochastic, ROC)
    - Volatility indicators (ATR, Bollinger Bands)
    - Volume indicators (OBV, VWAP)
    - Price patterns and derived metrics
    """

    def __init__(self):
        self.logger = logger.bind(processor="technical")

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical features.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with all features added
        """
        df = df.copy()

        # Basic price features
        df = self._add_returns(df)

        # Moving averages
        df = self._add_moving_averages(df)

        # Momentum indicators
        df = self._add_momentum_indicators(df)

        # Volatility indicators
        df = self._add_volatility_indicators(df)

        # Volume indicators
        df = self._add_volume_indicators(df)

        # Price patterns
        df = self._add_price_patterns(df)

        # Time features
        df = self._add_time_features(df)

        self.logger.info(f"Calculated {len(df.columns)} features")
        return df

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Multi-period returns
        for period in [5, 10, 20]:
            df[f"returns_{period}d"] = df["close"].pct_change(period)

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f"sma_{period}"] = df["close"].rolling(window=period).mean()
            df[f"sma_{period}_slope"] = df[f"sma_{period}"].diff(5) / 5

        # Exponential Moving Averages
        for period in [12, 26, 50]:
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

        # Price relative to MAs
        df["price_sma_20_ratio"] = df["close"] / df["sma_20"]
        df["price_sma_50_ratio"] = df["close"] / df["sma_50"]
        df["price_sma_200_ratio"] = df["close"] / df["sma_200"]

        # MA crossovers
        df["sma_20_50_cross"] = (df["sma_20"] > df["sma_50"]).astype(int)
        df["sma_50_200_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)

        # MACD
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        # RSI
        for period in [14, 21]:
            df[f"rsi_{period}"] = self._calculate_rsi(df["close"], period)

        # Stochastic Oscillator
        df["stoch_k"], df["stoch_d"] = self._calculate_stochastic(df, 14, 3)

        # Rate of Change
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = (df["close"] - df["close"].shift(period)) / df["close"].shift(period) * 100

        # Momentum
        for period in [10, 20]:
            df[f"momentum_{period}"] = df["close"] - df["close"].shift(period)

        # Williams %R
        df["williams_r"] = self._calculate_williams_r(df, 14)

        # CCI (Commodity Channel Index)
        df["cci"] = self._calculate_cci(df, 20)

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        # ATR (Average True Range)
        df["atr_14"] = self._calculate_atr(df, 14)
        df["atr_20"] = self._calculate_atr(df, 20)

        # Normalized ATR
        df["atr_pct"] = df["atr_14"] / df["close"] * 100

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(window=20).mean()
        bb_std = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_middle"] + 2 * bb_std
        df["bb_lower"] = df["bb_middle"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Historical Volatility
        for period in [10, 20, 60]:
            df[f"volatility_{period}d"] = df["log_returns"].rolling(window=period).std() * np.sqrt(252)

        # Volatility ratio
        df["volatility_ratio"] = df["volatility_10d"] / df["volatility_60d"]

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume indicators."""
        # Volume moving averages
        df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

        # On-Balance Volume
        df["obv"] = self._calculate_obv(df)
        df["obv_sma_20"] = df["obv"].rolling(window=20).mean()

        # Volume Price Trend
        df["vpt"] = (df["returns"] * df["volume"]).cumsum()

        # Money Flow Index
        df["mfi"] = self._calculate_mfi(df, 14)

        # Accumulation/Distribution
        df["ad_line"] = self._calculate_ad_line(df)

        return df

    def _add_price_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price pattern features."""
        # Candlestick body and shadows
        df["body"] = abs(df["close"] - df["open"])
        df["upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
        df["body_ratio"] = df["body"] / (df["high"] - df["low"] + 1e-10)

        # Higher highs / Lower lows
        df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
        df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)

        # Gap
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        # Range
        df["daily_range"] = (df["high"] - df["low"]) / df["low"]
        df["range_sma_10"] = df["daily_range"].rolling(window=10).mean()

        # Support/Resistance levels (rolling min/max)
        df["resistance_20"] = df["high"].rolling(window=20).max()
        df["support_20"] = df["low"].rolling(window=20).min()
        df["price_to_resistance"] = df["close"] / df["resistance_20"]
        df["price_to_support"] = df["close"] / df["support_20"]

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if "timestamp" in df.columns:
            df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
            df["month"] = pd.to_datetime(df["timestamp"]).dt.month
            df["quarter"] = pd.to_datetime(df["timestamp"]).dt.quarter
            df["is_month_end"] = pd.to_datetime(df["timestamp"]).dt.is_month_end.astype(int)
            df["is_month_start"] = pd.to_datetime(df["timestamp"]).dt.is_month_start.astype(int)

        return df

    # Helper methods for indicator calculations
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calculate_stochastic(
        self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()

        stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
        stoch_d = stoch_k.rolling(window=d_period).mean()

        return stoch_k, stoch_d

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high_max = df["high"].rolling(window=period).max()
        low_min = df["low"].rolling(window=period).min()
        return -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        return (typical_price - sma) / (0.015 * mad + 1e-10)

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = 0

        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + df["volume"].iloc[i]
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - df["volume"].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = typical_price * df["volume"]

        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)

        # Calculate positive and negative money flow
        price_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(price_diff > 0, 0)
        negative_flow = raw_money_flow.where(price_diff < 0, 0)

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        return mfi

    def _calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"] + 1e-10
        )
        ad = (clv * df["volume"]).cumsum()
        return ad

    def get_feature_names(self) -> list[str]:
        """Get list of all feature names calculated."""
        # This is a reference list - actual features depend on calculate_all_features
        return [
            # Returns
            "returns", "log_returns", "returns_5d", "returns_10d", "returns_20d",
            # Moving Averages
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_12", "ema_26", "ema_50",
            "price_sma_20_ratio", "price_sma_50_ratio", "price_sma_200_ratio",
            "sma_20_50_cross", "sma_50_200_cross",
            "macd", "macd_signal", "macd_histogram",
            # Momentum
            "rsi_14", "rsi_21", "stoch_k", "stoch_d",
            "roc_5", "roc_10", "roc_20",
            "momentum_10", "momentum_20",
            "williams_r", "cci",
            # Volatility
            "atr_14", "atr_20", "atr_pct",
            "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_position",
            "volatility_10d", "volatility_20d", "volatility_60d", "volatility_ratio",
            # Volume
            "volume_sma_20", "volume_ratio", "obv", "obv_sma_20", "vpt", "mfi", "ad_line",
            # Price Patterns
            "body", "upper_shadow", "lower_shadow", "body_ratio",
            "higher_high", "lower_low", "gap", "daily_range",
            "resistance_20", "support_20", "price_to_resistance", "price_to_support",
            # Time
            "day_of_week", "month", "quarter", "is_month_end", "is_month_start",
        ]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all technical features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with technical features added
    """
    engine = TechnicalFeatureEngine()
    return engine.calculate_all_features(df)


if __name__ == "__main__":
    # Test the feature engine
    print("Testing Technical Feature Engine...")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    n = 500

    dates = pd.date_range(start="2022-01-01", periods=n, freq="D")
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))

    df = pd.DataFrame({
        "timestamp": dates,
        "open": prices * (1 + np.random.randn(n) * 0.01),
        "high": prices * (1 + abs(np.random.randn(n) * 0.02)),
        "low": prices * (1 - abs(np.random.randn(n) * 0.02)),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, n),
    })

    engine = TechnicalFeatureEngine()
    df_features = engine.calculate_all_features(df)

    print(f"Original columns: {len(df.columns)}")
    print(f"With features: {len(df_features.columns)}")
    print(f"\nFeature columns added: {len(df_features.columns) - len(df.columns)}")
    print(f"\nSample features:")
    print(df_features[["close", "rsi_14", "macd", "bb_position", "atr_pct"]].tail())
