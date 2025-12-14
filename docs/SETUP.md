# Setup & Pipeline Guide

Complete guide for setting up and running the NBA data processing pipeline.

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Package

```bash
pip install -e .
```

## Pipeline Overview

The pipeline processes NBA game data through 5 steps:

1. **Initialize Database** - Create SQLite database with schema
2. **Ingest Games** - Fetch games from NBA API (2019-20 to 2023-24)
3. **Extract Features** - Extract 22 features per team per game
4. **Build Sequences** - Create 10-game rolling sequences
5. **Build Dataset** - Create matchup training samples

## Quick Start

### Run Full Pipeline

```bash
# Default: Excludes 2024-25 season
python -m nba_score_winpred.data_pipeline.run_pipeline

# Include current season
python -m nba_score_winpred.data_pipeline.run_pipeline --include-current
```

### Step-by-Step

```bash
# Step 1: Initialize database
python -c "from nba_score_winpred.database.db import initialize_db; initialize_db()"

# Step 2: Ingest games
python src/nba_score_winpred/data_pipeline/ingest_games.py

# Step 3: Extract features
python src/nba_score_winpred/features/extract_features.py

# Step 4: Build sequences
python src/nba_score_winpred/features/build_sequences.py

# Step 5: Build dataset
python src/nba_score_winpred/dataset/dataset_builder.py
```

## What Gets Created

### Database (`data/database/nba.db`)

- **games**: All game records with raw JSON stats
- **team_features**: 22 features per team per game
- **team_sequences**: 10-game sequences for each team

### Training Dataset (`data/processed/training_data.pt`)

PyTorch file containing:
- `sequenceA`: Home team sequences (N, 10, 22)
- `sequenceB`: Away team sequences (N, 10, 22)
- `scoreA`, `scoreB`: Actual scores (N, 1)
- `winA`: Win indicator (N, 1)

## Features Extracted

Each team-game has 22 features:

- **Shooting/Scoring (6)**: PTS, FG_PCT, FG3_PCT, FT_PCT, EFG_PCT, TS_PCT
- **Volume Stats (5)**: FGA, FG3A, FTA, AST, TOV
- **Rebounding (3)**: OREB, DREB, REB
- **Defense (3)**: STL, BLK, PF
- **Pace (2)**: POSS, PACE
- **Context (3)**: PLUS_MINUS, HOME, WIN

## Verification

### Quick Check

```bash
python tests/quick_check.py
```

### Comprehensive Test

```bash
python tests/test_pipeline.py
```

### Manual Verification

```bash
# Check database
sqlite3 data/database/nba.db "SELECT COUNT(*) FROM games;"

# Check dataset
python -c "import torch; d=torch.load('data/processed/training_data.pt'); print(f'Samples: {d[\"num_samples\"]}')"
```

## Troubleshooting

### Database Errors

```bash
# Reset database
rm data/database/nba.db
python -c "from nba_score_winpred.database.db import initialize_db; initialize_db()"
```

### Import Errors

```bash
# Reinstall package
pip install -e .
```

### Missing Data

- Check internet connection (API calls required)
- Verify NBA API is accessible
- Check that seasons exist (e.g., "2023-24")

## Performance

Expected processing times:
- **Ingestion**: 2-5 minutes per season
- **Feature Extraction**: 1-2 minutes
- **Sequence Building**: 30 seconds - 1 minute
- **Dataset Building**: 10-30 seconds

**Total**: ~15-30 minutes for 5 seasons

## Output Statistics

After running the pipeline, you should see:
- **Games**: ~5,800-6,000 regular season games
- **Features**: ~11,600-12,000 team-game features
- **Sequences**: ~11,000-11,500 sequences
- **Training Samples**: ~5,600-5,700 matchups

Some games are skipped (first ~9 games per team) because they don't have enough history for 10-game sequences.

## Next Steps

Once the pipeline completes:

1. **Load Dataset**:
   ```python
   import torch
   data = torch.load("data/processed/training_data.pt")
   ```

2. **Split Data**: Create train/val/test splits

3. **Train Model**: Build and train your LSTM model

See the main [README.md](../README.md) for more details.

