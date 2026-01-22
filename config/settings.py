"""
Configuration management for the trading system.
Loads environment variables and provides typed access to settings.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str
    secret_key: str
    base_url: str = "https://paper-api.alpaca.markets"

    @classmethod
    def from_env(cls) -> "AlpacaConfig":
        return cls(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
            base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        )


@dataclass
class DataSourceConfig:
    """Configuration for data source APIs."""
    alpha_vantage_key: str
    finnhub_key: str
    news_api_key: str
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None

    @classmethod
    def from_env(cls) -> "DataSourceConfig":
        return cls(
            alpha_vantage_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
            finnhub_key=os.getenv("FINNHUB_API_KEY", ""),
            news_api_key=os.getenv("NEWS_API_KEY", ""),
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID"),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        )


@dataclass
class TradingConfig:
    """Trading parameters and risk limits."""
    paper_trading: bool = True
    max_position_size: float = 0.10  # 10% of portfolio per position
    max_portfolio_risk: float = 0.15  # 15% max drawdown trigger
    max_sector_exposure: float = 0.30  # 30% max in single sector
    min_cash_reserve: float = 0.10  # Keep 10% in cash
    max_correlation: float = 0.70  # Max correlation between positions

    @classmethod
    def from_env(cls) -> "TradingConfig":
        return cls(
            paper_trading=os.getenv("PAPER_TRADING", "true").lower() == "true",
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "0.10")),
            max_portfolio_risk=float(os.getenv("MAX_PORTFOLIO_RISK", "0.15")),
        )


@dataclass
class DatabaseConfig:
    """Database configuration."""
    path: Path

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        db_path = os.getenv("DATABASE_PATH", "data/trading.db")
        return cls(path=PROJECT_ROOT / db_path)


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    api_key: str
    project: str = "multiagent-trading"

    @classmethod
    def from_env(cls) -> "WandbConfig":
        return cls(
            api_key=os.getenv("WANDB_API_KEY", ""),
            project=os.getenv("WANDB_PROJECT", "multiagent-trading"),
        )


class Settings:
    """Central settings access point."""

    def __init__(self):
        self.alpaca = AlpacaConfig.from_env()
        self.data_sources = DataSourceConfig.from_env()
        self.trading = TradingConfig.from_env()
        self.database = DatabaseConfig.from_env()
        self.wandb = WandbConfig.from_env()
        self.project_root = PROJECT_ROOT

    def validate(self) -> list[str]:
        """Validate that required settings are configured."""
        errors = []

        if not self.alpaca.api_key:
            errors.append("ALPACA_API_KEY not set")
        if not self.alpaca.secret_key:
            errors.append("ALPACA_SECRET_KEY not set")
        if not self.data_sources.alpha_vantage_key:
            errors.append("ALPHA_VANTAGE_API_KEY not set")

        return errors

    def print_status(self):
        """Print configuration status."""
        print("=" * 50)
        print("Configuration Status")
        print("=" * 50)
        print(f"Alpaca API Key: {'✓' if self.alpaca.api_key else '✗'}")
        print(f"Alpaca Secret: {'✓' if self.alpaca.secret_key else '✗'}")
        print(f"Alpha Vantage: {'✓' if self.data_sources.alpha_vantage_key else '✗'}")
        print(f"Finnhub: {'✓' if self.data_sources.finnhub_key else '✗'}")
        print(f"NewsAPI: {'✓' if self.data_sources.news_api_key else '✗'}")
        print(f"W&B: {'✓' if self.wandb.api_key else '✗'}")
        print(f"Paper Trading: {self.trading.paper_trading}")
        print(f"Database: {self.database.path}")
        print("=" * 50)


# Global settings instance
settings = Settings()


if __name__ == "__main__":
    settings.print_status()
    errors = settings.validate()
    if errors:
        print("\nConfiguration errors:")
        for error in errors:
            print(f"  - {error}")
