# NBA Score Prediction Project

A machine learning project for predicting NBA game scores and outcomes using LSTM neural networks.

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 2. Initialize Database

```bash
python -c "from nba_score_winpred.database.db import initialize_db; initialize_db()"
```

### 3. Run Data Pipeline

```bash
# Run full pipeline (ingests 2019-20 through 2023-24 seasons)
python -m nba_score_winpred.data_pipeline.run_pipeline

# Include current season (2024-25)
python -m nba_score_winpred.data_pipeline.run_pipeline --include-current
```

This will:
1. Fetch NBA games from the API
2. Extract features from game data
3. Build 10-game sequences for each team
4. Create training dataset

Output: `data/processed/training_data.pt` (PyTorch tensors ready for training)

### 4. Verify Setup

```bash
python tests/quick_check.py
```

## Project Structure

```
nba-score-winpred/
├── src/nba_score_winpred/     # Main package
│   ├── data_pipeline/         # Data ingestion
│   ├── features/              # Feature extraction
│   ├── dataset/               # Dataset building
│   └── database/              # Database management
├── data/                      # Data storage
│   ├── database/              # SQLite database
│   └── processed/             # Processed datasets
├── docs/                      # Documentation
├── tests/                     # Test scripts
└── scripts/                   # Utility scripts
```

## Documentation

- **[Setup & Pipeline Guide](docs/SETUP.md)** - Complete setup and pipeline instructions
- **[Database Guide](docs/DATABASE.md)** - Database structure and data storage

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Usage

### Run Individual Pipeline Steps

```bash
# Step 1: Ingest games
python src/nba_score_winpred/data_pipeline/ingest_games.py

# Step 2: Extract features
python src/nba_score_winpred/features/extract_features.py

# Step 3: Build sequences
python src/nba_score_winpred/features/build_sequences.py

# Step 4: Build dataset
python src/nba_score_winpred/dataset/dataset_builder.py
```

### Testing

```bash
# Quick verification
python tests/quick_check.py

# Comprehensive test suite
python tests/test_pipeline.py
```

## Next Steps

After running the pipeline, you'll have:
- Training dataset: `data/processed/training_data.pt`
- 5,690+ matchup samples
- Each sample: 2 sequences (10 games × 22 features) + scores + outcome

Ready to train your LSTM model!

## License

[Your License Here]
