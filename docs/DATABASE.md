# Database & Data Storage Guide

Complete guide to the database structure and data storage in this project.

## Database Location

```
data/database/nba.db
```

SQLite database - portable, file-based, no server required.

## Schema

### `games` Table

Stores all NBA game records.

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | TEXT | NBA game identifier (PRIMARY KEY) |
| `game_date` | TEXT | Date of game (YYYY-MM-DD) |
| `season` | TEXT | Season identifier (e.g., "22024") |
| `home_team_id` | INTEGER | Home team NBA ID |
| `away_team_id` | INTEGER | Away team NBA ID |
| `home_team_abbr` | TEXT | Home team abbreviation (e.g., "LAL") |
| `away_team_abbr` | TEXT | Away team abbreviation (e.g., "BOS") |
| `home_score` | INTEGER | Home team points |
| `away_score` | INTEGER | Away team points |
| `raw_json` | TEXT | Complete JSON stats for both teams |

**Example Row:**
```
game_id: "0022401199"
game_date: "2024-01-15"
season: "22024"
home_team_id: 1610612747
away_team_id: 1610612757
home_team_abbr: "LAL"
away_team_abbr: "POR"
home_score: 112
away_score: 108
raw_json: "[{...team stats...}, {...opponent stats...}]"
```

### `team_features` Table

Stores extracted features for each team in each game.

| Column | Type | Description |
|--------|------|-------------|
| `team_id` | INTEGER | Team NBA ID |
| `game_id` | TEXT | Game identifier |
| `game_date` | TEXT | Date of game |
| `feature_vector_json` | TEXT | JSON array of 22 features |

**Feature Vector (22 features):**
1. PTS, FG_PCT, FG3_PCT, FT_PCT, EFG_PCT, TS_PCT (6)
2. FGA, FG3A, FTA, AST, TOV (5)
3. OREB, DREB, REB (3)
4. STL, BLK, PF (3)
5. POSS, PACE (2)
6. PLUS_MINUS, HOME, WIN (3)

### `team_sequences` Table

Stores 10-game rolling sequences for each team.

| Column | Type | Description |
|--------|------|-------------|
| `team_id` | INTEGER | Team NBA ID |
| `game_id` | TEXT | Game identifier (last game in sequence) |
| `sequence_json` | TEXT | JSON array of 10 feature vectors |

**Sequence Structure:**
- 10 consecutive games
- Each game: 22 features
- Shape: (10, 22)

## Data Flow

```
NBA API
  ↓
games table (raw data)
  ↓
team_features table (extracted features)
  ↓
team_sequences table (10-game sequences)
  ↓
training_data.pt (PyTorch tensors)
```

## Database Operations

### Initialize Database

```bash
python -c "from nba_score_winpred.database.db import initialize_db; initialize_db()"
```

### Query Examples

```bash
# Open database shell
sqlite3 data/database/nba.db

# Count games
SELECT COUNT(*) FROM games;

# Count features
SELECT COUNT(*) FROM team_features;

# Count sequences
SELECT COUNT(*) FROM team_sequences;

# View sample game
SELECT * FROM games LIMIT 1;

# Check feature vector length
SELECT json_array_length(feature_vector_json) FROM team_features LIMIT 1;

# Check sequence structure
SELECT 
    json_array_length(sequence_json) as games,
    json_array_length(json_extract(sequence_json, '$[0]')) as features
FROM team_sequences LIMIT 1;

.quit
```

### Reset Database

```bash
# Delete database
rm data/database/nba.db

# Reinitialize
python -c "from nba_score_winpred.database.db import initialize_db; initialize_db()"
```

## Processed Data

### Training Dataset

**Location:** `data/processed/training_data.pt`

**Format:** PyTorch tensor dictionary

**Structure:**
```python
{
    "sequenceA": torch.Tensor,  # (N, 10, 22) - Home team sequences
    "sequenceB": torch.Tensor,  # (N, 10, 22) - Away team sequences
    "scoreA": torch.Tensor,      # (N, 1) - Home team scores
    "scoreB": torch.Tensor,      # (N, 1) - Away team scores
    "winA": torch.Tensor,        # (N, 1) - Win indicator
    "num_samples": int,
    "sequence_shape": tuple,     # (10, 22)
    "num_features": int          # 22
}
```

### Loading the Dataset

```python
import torch

# Load dataset
data = torch.load("data/processed/training_data.pt")

# Access tensors
sequenceA = data["sequenceA"]  # (N, 10, 22)
sequenceB = data["sequenceB"]  # (N, 10, 22)
scoreA = data["scoreA"]        # (N, 1)
scoreB = data["scoreB"]        # (N, 1)
winA = data["winA"]            # (N, 1)

print(f"Dataset contains {data['num_samples']} samples")
```

## Data Statistics

### Typical Values

After processing 5 seasons (2019-20 to 2023-24):

- **Games**: ~5,800-6,000
- **Team-Game Features**: ~11,600-12,000
- **Sequences**: ~11,000-11,500
- **Training Samples**: ~5,600-5,700

### Why Some Games Are Skipped

- Teams need 10+ previous games to build sequences
- First ~9 games per team don't have enough history
- This is expected and ensures all sequences are complete

## Data Quality

### Validation

The pipeline includes automatic validation:

- ✅ Missing data handling (NaN/Inf → 0)
- ✅ Team ID validation (NBA teams only)
- ✅ Sequence length validation (must be 10 games)
- ✅ Feature vector validation (must be 22 features)

### Data Integrity

- All games have `raw_json` with complete stats
- All features are extracted from validated data
- All sequences are complete (10 games × 22 features)
- No NaN or Inf values in final dataset

## Storage Requirements

- **Database**: ~50-100 MB (SQLite with JSON)
- **Training Dataset**: ~5-50 MB (PyTorch tensors)
- **Total**: ~100-150 MB for 5 seasons

## Backup & Recovery

### Backup Database

```bash
cp data/database/nba.db data/database/nba.db.backup
```

### Restore Database

```bash
cp data/database/nba.db.backup data/database/nba.db
```

### Export Data

```bash
# Export to CSV (games only)
sqlite3 -header -csv data/database/nba.db "SELECT * FROM games;" > games_export.csv
```

## Maintenance

### Clean Up

```bash
# Remove processed data (keep database)
rm data/processed/training_data.pt

# Rebuild dataset
python src/nba_score_winpred/dataset/dataset_builder.py
```

### Vacuum Database

```bash
sqlite3 data/database/nba.db "VACUUM;"
```

This optimizes database file size after deletions.

