"""
SQLite database storage for market data.
Provides persistent storage for OHLCV data and features.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
from loguru import logger
from contextlib import contextmanager


class Database:
    """
    SQLite database manager for trading data.

    Tables:
    - ohlcv: Raw OHLCV data
    - features: Calculated features
    - trades: Trade history
    - metrics: Performance metrics
    """

    def __init__(self, db_path: str | Path = "data/trading.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(storage="database")

        # Initialize database schema
        self._init_schema()
        self.logger.info(f"Database initialized at {self.db_path}")

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # OHLCV table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    source TEXT DEFAULT 'unknown',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, source)
                )
            """)

            # Features table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    feature_name TEXT NOT NULL,
                    feature_value REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timestamp, feature_name)
                )
            """)

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    agent TEXT,
                    confidence REAL,
                    reason TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indices for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_features_symbol_time ON features(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp)")

    def save_ohlcv(
        self,
        df: pd.DataFrame,
        symbol: str,
        source: str = "unknown",
    ) -> int:
        """
        Save OHLCV data to database.

        Args:
            df: DataFrame with OHLCV columns
            symbol: Trading symbol
            source: Data source identifier

        Returns:
            Number of rows inserted
        """
        if df.empty:
            return 0

        df = df.copy()
        df["symbol"] = symbol
        df["source"] = source

        # Ensure required columns exist
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame must have columns: {required}")

        rows_inserted = 0

        with self.get_connection() as conn:
            for _, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO ohlcv
                        (symbol, timestamp, open, high, low, close, volume, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        row["timestamp"],
                        row["open"],
                        row["high"],
                        row["low"],
                        row["close"],
                        row["volume"],
                        source,
                    ))
                    rows_inserted += 1
                except Exception as e:
                    self.logger.warning(f"Error inserting row: {e}")

        self.logger.info(f"Saved {rows_inserted} rows for {symbol}")
        return rows_inserted

    def save_ohlcv_bulk(
        self,
        data: dict[str, pd.DataFrame],
        source: str = "unknown",
    ) -> int:
        """
        Save OHLCV data for multiple symbols.

        Args:
            data: Dictionary mapping symbol to DataFrame
            source: Data source identifier

        Returns:
            Total rows inserted
        """
        total = 0
        for symbol, df in data.items():
            total += self.save_ohlcv(df, symbol, source)
        return total

    def load_ohlcv(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from database.

        Args:
            symbol: Trading symbol
            start_date: Start of date range
            end_date: End of date range
            source: Filter by data source

        Returns:
            DataFrame with OHLCV data
        """
        query = "SELECT timestamp, open, high, low, close, volume FROM ohlcv WHERE symbol = ?"
        params = [symbol]

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY timestamp"

        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["symbol"] = symbol

        return df

    def load_multiple_symbols(
        self,
        symbols: list[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV data for multiple symbols.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        return {
            symbol: self.load_ohlcv(symbol, start_date, end_date)
            for symbol in symbols
            if not self.load_ohlcv(symbol, start_date, end_date).empty
        }

    def get_available_symbols(self) -> list[str]:
        """Get list of symbols with data in database."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT DISTINCT symbol FROM ohlcv ORDER BY symbol")
            return [row[0] for row in cursor.fetchall()]

    def get_date_range(self, symbol: str) -> tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range available for a symbol."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT MIN(timestamp), MAX(timestamp)
                FROM ohlcv
                WHERE symbol = ?
            """, (symbol,))
            row = cursor.fetchone()

            if row and row[0] and row[1]:
                return (
                    datetime.fromisoformat(row[0]),
                    datetime.fromisoformat(row[1]),
                )
            return None, None

    def save_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        agent: Optional[str] = None,
        confidence: Optional[float] = None,
        reason: Optional[str] = None,
    ) -> int:
        """
        Save a trade record.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Trade quantity
            price: Execution price
            agent: Agent that made the trade
            confidence: Confidence level
            reason: Trade reason/explanation

        Returns:
            Trade ID
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO trades (symbol, timestamp, side, quantity, price, agent, confidence, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, datetime.now().isoformat(), side, quantity, price, agent, confidence, reason))
            return cursor.lastrowid

    def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        agent: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get trade history.

        Args:
            symbol: Filter by symbol
            start_date: Start date
            end_date: End date
            agent: Filter by agent

        Returns:
            DataFrame with trade history
        """
        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())

        if agent:
            query += " AND agent = ?"
            params.append(agent)

        query += " ORDER BY timestamp DESC"

        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def save_metric(self, name: str, value: float) -> int:
        """Save a performance metric."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO metrics (timestamp, metric_name, metric_value)
                VALUES (?, ?, ?)
            """, (datetime.now().isoformat(), name, value))
            return cursor.lastrowid

    def get_data_summary(self) -> dict:
        """Get summary of data in database."""
        with self.get_connection() as conn:
            # Count records per table
            ohlcv_count = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
            symbols_count = conn.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv").fetchone()[0]
            trades_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

            return {
                "ohlcv_records": ohlcv_count,
                "symbols": symbols_count,
                "trades": trades_count,
                "database_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0,
            }

    def vacuum(self):
        """Optimize database by running VACUUM."""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
        self.logger.info("Database vacuumed")


# Convenience functions
def get_database(db_path: str = "data/trading.db") -> Database:
    """Get database instance."""
    return Database(db_path)


if __name__ == "__main__":
    # Test the database
    print("Testing Database Storage...")
    print("=" * 50)

    import numpy as np

    db = Database("data/test_trading.db")

    # Create sample data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 120, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.randint(1000000, 10000000, 100),
    })

    # Save data
    rows = db.save_ohlcv(df, "TEST", source="test")
    print(f"Saved {rows} rows")

    # Load data
    loaded = db.load_ohlcv("TEST")
    print(f"Loaded {len(loaded)} rows")

    # Get summary
    summary = db.get_data_summary()
    print(f"Summary: {summary}")

    # Save a trade
    trade_id = db.save_trade(
        symbol="TEST",
        side="buy",
        quantity=100,
        price=105.50,
        agent="test_agent",
        confidence=0.85,
        reason="Test trade",
    )
    print(f"Saved trade with ID: {trade_id}")

    print("\nTest completed successfully!")
