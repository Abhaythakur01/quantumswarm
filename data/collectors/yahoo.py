"""
Yahoo Finance data collector.
Primary source for stock OHLCV data (free, no rate limits).
"""

from datetime import datetime
from typing import Optional
import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm

from .base import BaseCollector, RateLimiter


class YahooFinanceCollector(BaseCollector):
    """
    Collector for Yahoo Finance data.

    Features:
    - Free, no API key required
    - No rate limits (but be respectful)
    - Supports stocks, ETFs, indices, some crypto
    - Daily, weekly, monthly data
    - Includes adjusted close prices
    """

    # Default stock universe for the project
    DEFAULT_SYMBOLS = [
        # Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
        # Finance
        "JPM", "BAC", "GS", "MS", "V", "MA", "BRK-B", "C", "WFC", "AXP",
        # Healthcare
        "JNJ", "UNH", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "LLY", "BMY",
        # Consumer
        "WMT", "PG", "KO", "PEP", "COST", "HD", "MCD", "NKE", "SBUX", "TGT",
        # Energy & Industrial
        "XOM", "CVX", "COP", "SLB", "CAT", "BA", "UPS", "HON", "GE", "MMM",
    ]

    # ETFs for market tracking
    ETF_SYMBOLS = [
        "SPY",   # S&P 500
        "QQQ",   # Nasdaq 100
        "IWM",   # Russell 2000
        "DIA",   # Dow Jones
        "VIX",   # Volatility Index (^VIX for actual index)
    ]

    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "60m",
        "1d": "1d",
        "1wk": "1wk",
        "1mo": "1mo",
    }

    def __init__(self):
        super().__init__("yahoo_finance")
        # Be respectful even without rate limits
        self.rate_limiter = RateLimiter(calls_per_minute=30)

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.

        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            start_date: Start date
            end_date: End date
            interval: '1d', '1h', '5m', etc.

        Returns:
            DataFrame with OHLCV data
        """
        self.rate_limiter.wait()

        try:
            ticker = yf.Ticker(symbol)
            yf_interval = self.INTERVAL_MAP.get(interval, interval)

            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=yf_interval,
                auto_adjust=True,  # Adjust for splits/dividends
            )

            if df.empty:
                self.logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Standardize column names
            df = df.reset_index()
            df.columns = df.columns.str.lower()

            # Rename 'date' to 'timestamp' if present
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})

            # Select and order columns
            df["symbol"] = symbol
            df = df[["timestamp", "open", "high", "low", "close", "volume", "symbol"]]

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
            symbols: List of tickers
            start_date: Start date
            end_date: End date
            interval: Time interval
            show_progress: Show progress bar

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}

        iterator = tqdm(symbols, desc="Fetching data") if show_progress else symbols

        for symbol in iterator:
            df = self.fetch_ohlcv(symbol, start_date, end_date, interval)
            if not df.empty:
                results[symbol] = df

        self.logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results

    def fetch_default_universe(
        self,
        start_date: datetime,
        end_date: datetime,
        include_etfs: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch data for the default symbol universe.

        Args:
            start_date: Start date
            end_date: End date
            include_etfs: Include ETF symbols

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        symbols = self.DEFAULT_SYMBOLS.copy()
        if include_etfs:
            symbols.extend(self.ETF_SYMBOLS)

        return self.fetch_multiple(symbols, start_date, end_date)

    def get_available_symbols(self) -> list[str]:
        """Get the default symbol universe."""
        return self.DEFAULT_SYMBOLS + self.ETF_SYMBOLS

    def get_ticker_info(self, symbol: str) -> dict:
        """
        Get metadata about a ticker.

        Args:
            symbol: Stock ticker

        Returns:
            Dictionary with ticker info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap", 0),
                "currency": info.get("currency", "USD"),
            }
        except Exception as e:
            self.logger.error(f"Error getting info for {symbol}: {e}")
            return {"symbol": symbol}

    def download_batch(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download data for multiple symbols at once using yfinance batch download.
        More efficient than individual fetches for many symbols.

        Args:
            symbols: List of tickers
            start_date: Start date
            end_date: End date
            interval: Time interval

        Returns:
            DataFrame with MultiIndex columns (symbol, ohlcv)
        """
        try:
            df = yf.download(
                tickers=symbols,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=self.INTERVAL_MAP.get(interval, interval),
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )

            self.logger.info(f"Batch downloaded {len(symbols)} symbols")
            return df

        except Exception as e:
            self.logger.error(f"Error in batch download: {e}")
            return pd.DataFrame()


# Convenience function for quick data fetching
def fetch_stock_data(
    symbols: list[str] | str,
    start_date: str | datetime,
    end_date: str | datetime,
    interval: str = "1d",
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """
    Quick helper to fetch stock data.

    Args:
        symbols: Single symbol or list of symbols
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        interval: Time interval

    Returns:
        DataFrame for single symbol, dict for multiple
    """
    collector = YahooFinanceCollector()

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
    collector = YahooFinanceCollector()

    print("Testing Yahoo Finance Collector...")
    print("=" * 50)

    # Test single symbol
    df = collector.fetch_ohlcv(
        "AAPL",
        datetime(2023, 1, 1),
        datetime(2024, 1, 1),
    )
    print(f"\nAAPL data shape: {df.shape}")
    print(df.head())

    # Test multiple symbols
    data = collector.fetch_multiple(
        ["AAPL", "MSFT", "GOOGL"],
        datetime(2023, 1, 1),
        datetime(2024, 1, 1),
    )
    print(f"\nFetched data for: {list(data.keys())}")
