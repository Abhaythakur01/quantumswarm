"""
Cryptocurrency data collector using CCXT.
Supports multiple exchanges (Binance, Coinbase, Kraken, etc.)
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import ccxt
from loguru import logger
from tqdm import tqdm
import time

from .base import BaseCollector, RateLimiter


class CryptoCollector(BaseCollector):
    """
    Collector for cryptocurrency data using CCXT library.

    Features:
    - Supports 100+ exchanges
    - Free historical data
    - Multiple timeframes
    - No API key needed for public data
    """

    # Default crypto universe
    DEFAULT_SYMBOLS = [
        "BTC/USDT",   # Bitcoin
        "ETH/USDT",   # Ethereum
        "BNB/USDT",   # Binance Coin
        "SOL/USDT",   # Solana
        "XRP/USDT",   # Ripple
        "ADA/USDT",   # Cardano
        "DOGE/USDT",  # Dogecoin
        "DOT/USDT",   # Polkadot
        "MATIC/USDT", # Polygon
        "LINK/USDT",  # Chainlink
    ]

    TIMEFRAME_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1w",
    }

    def __init__(self, exchange_id: str = "binance"):
        """
        Initialize crypto collector.

        Args:
            exchange_id: CCXT exchange ID ('binance', 'coinbase', 'kraken', etc.)
        """
        super().__init__(f"crypto_{exchange_id}")
        self.exchange_id = exchange_id

        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
            },
        })

        # Rate limiter (most exchanges allow ~1200 req/min)
        self.rate_limiter = RateLimiter(calls_per_minute=60)

        self.logger.info(f"Initialized {exchange_id} exchange")

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a crypto pair.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            start_date: Start date
            end_date: End date
            interval: Timeframe ('1d', '1h', '15m', etc.)

        Returns:
            DataFrame with OHLCV data
        """
        timeframe = self.TIMEFRAME_MAP.get(interval, interval)

        # Convert dates to timestamps
        since = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)

        all_candles = []

        try:
            # Fetch in batches (most exchanges limit to 500-1000 candles per request)
            while since < end_ts:
                self.rate_limiter.wait()

                candles = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000,
                )

                if not candles:
                    break

                all_candles.extend(candles)

                # Move to next batch
                since = candles[-1][0] + 1

                # Safety check to prevent infinite loops
                if len(candles) < 1000:
                    break

            if not all_candles:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(
                all_candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Filter to requested date range
            df = df[
                (df["timestamp"] >= start_date) &
                (df["timestamp"] <= end_date)
            ]

            df["symbol"] = symbol

            # Clean the data
            df = self.clean_dataframe(df)

            self.logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    def fetch_multiple(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
        show_progress: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols.

        Args:
            symbols: List of trading pairs
            start_date: Start date
            end_date: End date
            interval: Timeframe
            show_progress: Show progress bar

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}

        iterator = tqdm(symbols, desc="Fetching crypto data") if show_progress else symbols

        for symbol in iterator:
            df = self.fetch_ohlcv(symbol, start_date, end_date, interval)
            if not df.empty:
                results[symbol] = df

            # Extra delay between symbols to respect rate limits
            time.sleep(0.5)

        self.logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results

    def fetch_default_universe(
        self,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for the default crypto universe.

        Args:
            start_date: Start date
            end_date: End date
            interval: Timeframe

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        return self.fetch_multiple(
            self.DEFAULT_SYMBOLS,
            start_date,
            end_date,
            interval,
        )

    def get_available_symbols(self) -> list[str]:
        """Get all available trading pairs on the exchange."""
        try:
            self.exchange.load_markets()
            return list(self.exchange.markets.keys())
        except Exception as e:
            self.logger.error(f"Error loading markets: {e}")
            return self.DEFAULT_SYMBOLS

    def get_ticker(self, symbol: str) -> dict:
        """
        Get current ticker data for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            Dictionary with ticker info
        """
        try:
            self.rate_limiter.wait()
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {e}")
            return {}

    def get_orderbook(self, symbol: str, limit: int = 20) -> dict:
        """
        Get current orderbook for a symbol.

        Args:
            symbol: Trading pair
            limit: Number of levels

        Returns:
            Dictionary with bids and asks
        """
        try:
            self.rate_limiter.wait()
            return self.exchange.fetch_order_book(symbol, limit=limit)
        except Exception as e:
            self.logger.error(f"Error fetching orderbook for {symbol}: {e}")
            return {"bids": [], "asks": []}


class MultiExchangeCollector:
    """
    Collector that aggregates data from multiple exchanges.
    Useful for finding best prices or comparing data quality.
    """

    def __init__(self, exchange_ids: list[str] = None):
        """
        Initialize multi-exchange collector.

        Args:
            exchange_ids: List of exchange IDs (default: binance, coinbase)
        """
        if exchange_ids is None:
            exchange_ids = ["binance"]

        self.collectors = {
            exchange_id: CryptoCollector(exchange_id)
            for exchange_id in exchange_ids
        }
        self.logger = logger.bind(collector="multi_exchange")

    def fetch_from_all(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for a symbol from all exchanges.

        Args:
            symbol: Trading pair
            start_date: Start date
            end_date: End date
            interval: Timeframe

        Returns:
            Dictionary mapping exchange to DataFrame
        """
        results = {}

        for exchange_id, collector in self.collectors.items():
            df = collector.fetch_ohlcv(symbol, start_date, end_date, interval)
            if not df.empty:
                results[exchange_id] = df

        return results


# Convenience function
def fetch_crypto_data(
    symbols: list[str] | str,
    start_date: str | datetime,
    end_date: str | datetime,
    interval: str = "1d",
    exchange: str = "binance",
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Quick helper to fetch crypto data.

    Args:
        symbols: Single symbol or list of symbols
        start_date: Start date
        end_date: End date
        interval: Timeframe
        exchange: Exchange ID

    Returns:
        DataFrame for single symbol, dict for multiple
    """
    collector = CryptoCollector(exchange)

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if isinstance(symbols, str):
        return collector.fetch_ohlcv(symbols, start_date, end_date, interval)
    else:
        return collector.fetch_multiple(symbols, start_date, end_date, interval)


if __name__ == "__main__":
    # Test the collector
    collector = CryptoCollector("binance")

    print("Testing Crypto Collector (Binance)...")
    print("=" * 50)

    # Test single symbol
    df = collector.fetch_ohlcv(
        "BTC/USDT",
        datetime(2023, 1, 1),
        datetime(2024, 1, 1),
    )
    print(f"\nBTC/USDT data shape: {df.shape}")
    print(df.head())

    # Test multiple symbols
    data = collector.fetch_multiple(
        ["BTC/USDT", "ETH/USDT"],
        datetime(2023, 1, 1),
        datetime(2024, 1, 1),
    )
    print(f"\nFetched data for: {list(data.keys())}")
