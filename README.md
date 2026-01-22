# Multi-Agent AI Trading System

A cutting-edge multi-agent trading system implementing 7 research innovations from 2023-2024.

## Features

- **Mamba SSM Agent** - Linear-time sequence modeling (5x faster than Transformers)
- **KAN Agent** - Kolmogorov-Arnold Networks for interpretable trading formulas
- **Liquid Neural Network Agent** - Adapts to market regime changes in real-time
- **Diffusion Scenario Generator** - Generates synthetic market scenarios for training
- **Constitutional AI Risk Manager** - Self-critiquing safety layer
- **Mixture of Experts Coordinator** - Dynamic expert selection
- **Test-Time Training** - Adaptation during inference

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy template
copy .env.template .env  # Windows
cp .env.template .env    # macOS/Linux

# Edit .env with your API keys
```

### 3. Download Data

```bash
# Download 5 years of stock data
python scripts/download_data.py

# Download crypto data
python scripts/download_data.py --crypto
```

### 4. Train Agents

```bash
python scripts/train_all.py
```

### 5. Run Backtests

```bash
python -m backtest.engine
```

## Project Structure

```
multiagent_trading/
├── config/             # Configuration files
├── data/               # Data collection & processing
│   ├── collectors/     # Data source adapters
│   ├── processors/     # Feature engineering
│   └── storage/        # Database layer
├── agents/             # Individual trading agents
├── core/               # Core system components
├── training/           # Training infrastructure
├── backtest/           # Backtesting engine
├── interpretability/   # Explainability tools
├── deployment/         # Production deployment
├── dashboard/          # Visualization
├── tests/              # Test suite
├── notebooks/          # Jupyter notebooks
├── scripts/            # Utility scripts
└── docs/               # Documentation
```

## Required API Keys

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| Alpaca | Paper trading | Yes |
| Alpha Vantage | Fundamentals | 500 calls/day |
| Finnhub | News | 60 calls/min |
| NewsAPI | News articles | 100 calls/day |
| Weights & Biases | Experiment tracking | Yes |

## Performance Targets

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 2.0 |
| Max Drawdown | < 15% |
| Win Rate | > 52% |
| Inference Latency | < 100ms |

## License

MIT License
