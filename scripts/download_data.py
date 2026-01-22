"""
Download historical data for all symbols.
Run this script to populate the database with 5 years of data.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --symbols AAPL,MSFT,GOOGL
    python scripts/download_data.py --crypto
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collectors.yahoo import YahooFinanceCollector
from data.collectors.crypto import CryptoCollector
from data.processors.technical import TechnicalFeatureEngine
from data.storage.database import Database
from loguru import logger


def download_stock_data(
    symbols: list[str] = None,
    start_date: datetime = None,
    end_date: datetime = None,
    db: Database = None,
):
    """
    Download stock data from Yahoo Finance.

    Args:
        symbols: List of symbols (default: use DEFAULT_SYMBOLS)
        start_date: Start date (default: 5 years ago)
        end_date: End date (default: today)
        db: Database instance
    """
    collector = YahooFinanceCollector()

    if symbols is None:
        symbols = collector.DEFAULT_SYMBOLS + collector.ETF_SYMBOLS

    if start_date is None:
        start_date = datetime(2019, 1, 1)

    if end_date is None:
        end_date = datetime.now()

    logger.info(f"Downloading data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")

    # Fetch data
    data = collector.fetch_multiple(symbols, start_date, end_date)

    logger.info(f"Successfully downloaded data for {len(data)} symbols")

    # Save to database
    if db:
        total_rows = db.save_ohlcv_bulk(data, source="yahoo")
        logger.info(f"Saved {total_rows} total rows to database")

    return data


def download_crypto_data(
    symbols: list[str] = None,
    start_date: datetime = None,
    end_date: datetime = None,
    db: Database = None,
):
    """
    Download crypto data from Binance via CCXT.

    Args:
        symbols: List of trading pairs (default: use DEFAULT_SYMBOLS)
        start_date: Start date (default: 5 years ago)
        end_date: End date (default: today)
        db: Database instance
    """
    collector = CryptoCollector("binance")

    if symbols is None:
        symbols = collector.DEFAULT_SYMBOLS

    if start_date is None:
        start_date = datetime(2019, 1, 1)

    if end_date is None:
        end_date = datetime.now()

    logger.info(f"Downloading crypto data for {len(symbols)} symbols")

    # Fetch data
    data = collector.fetch_multiple(symbols, start_date, end_date)

    logger.info(f"Successfully downloaded data for {len(data)} symbols")

    # Save to database
    if db:
        total_rows = db.save_ohlcv_bulk(data, source="binance")
        logger.info(f"Saved {total_rows} total rows to database")

    return data


def calculate_and_store_features(db: Database):
    """
    Calculate technical features for all symbols in database.

    Args:
        db: Database instance
    """
    engine = TechnicalFeatureEngine()
    symbols = db.get_available_symbols()

    logger.info(f"Calculating features for {len(symbols)} symbols")

    for symbol in symbols:
        df = db.load_ohlcv(symbol)
        if df.empty:
            continue

        # Calculate features
        df_features = engine.calculate_all_features(df)

        # Log progress
        logger.info(f"Calculated {len(df_features.columns)} features for {symbol}")

    logger.info("Feature calculation complete")


def main():
    parser = argparse.ArgumentParser(description="Download historical market data")
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (default: all)",
    )
    parser.add_argument(
        "--crypto",
        action="store_true",
        help="Download crypto data instead of stocks",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2019-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/trading.db",
        help="Database path",
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip feature calculation",
    )

    args = parser.parse_args()

    # Parse dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]

    # Initialize database
    db = Database(args.db_path)

    print("=" * 60)
    print("Multi-Agent Trading System - Data Download")
    print("=" * 60)

    if args.crypto:
        print("\nDownloading cryptocurrency data...")
        download_crypto_data(symbols, start_date, end_date, db)
    else:
        print("\nDownloading stock data...")
        download_stock_data(symbols, start_date, end_date, db)

    # Print summary
    summary = db.get_data_summary()
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Total OHLCV records: {summary['ohlcv_records']:,}")
    print(f"Symbols downloaded: {summary['symbols']}")
    print(f"Database size: {summary['database_size_mb']:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
