from nba_score_winpred.data_pipeline.nba_api_client import get_games
from nba_score_winpred.database.db import get_connection
import json
import numpy as np
import pandas as pd
from collections import defaultdict

# --- Helper: Convert Pandas/Numpy types to JSON-safe Python types ---
def to_json_safe(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj

# --- Main ingestion function ---
def main(seasons=None, include_current_season=False):
    """
    Ingest NBA games for multiple seasons.
    
    Args:
        seasons: List of seasons to ingest (e.g., ["2019-20", "2020-21", ...])
                 If None, defaults to 2019-20 through 2023-24
        include_current_season: Whether to include 2024-25 season
    """
    if seasons is None:
        seasons = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24"]
    
    if include_current_season:
        seasons.append("2024-25")
    
    print(f"Fetching NBA games for seasons: {seasons}")
    
    all_rows = []
    for season in seasons:
        rows = get_games(season)
        all_rows.extend(rows)
        print(f"Fetched {len(rows)} rows for season {season}.")
    
    rows = all_rows
    print(f"Total fetched {len(rows)} rows from nba_api.")

    # ------------------------------------------------------------------
    # STEP 1 — KEEP ONLY REGULAR SEASON
    # SEASON_ID prefix:
    #   "1" = Preseason
    #   "2" = Regular Season  <-- ONLY keep these
    #   "3" = All-Star
    #   "4" = Playoffs
    # ------------------------------------------------------------------
    filtered = []
    for r in rows:
        # Keep only REGULAR SEASON (season_id starts with "2")
        if not str(r["SEASON_ID"]).startswith("2"):
            continue

        # Skip missing matchup
        if r.get("MATCHUP") is None:
            continue

        # Keep only REAL NBA team IDs
        team_id = int(r["TEAM_ID"])
        if not (1610612737 <= team_id <= 1610612766):
            continue

        filtered.append(r)


    print(f"Remaining after filtering to REGULAR SEASON: {len(filtered)} rows.")

    # ------------------------------------------------------------------
    # STEP 2 — GROUP TEAM-LEVEL ROWS INTO GAMES (2 rows per game)
    # ------------------------------------------------------------------
    games = defaultdict(list)
    for r in filtered:
        games[r["GAME_ID"]].append(r)

    conn = get_connection()
    cur = conn.cursor()

    inserted = 0

    # ------------------------------------------------------------------
    # STEP 3 — PROCESS EACH GAME
    # ------------------------------------------------------------------
    for game_id, entries in games.items():

        # Must have EXACTLY two records: one home, one away
        if len(entries) != 2:
            print(f"Skipping {game_id}: expected 2 rows, got {len(entries)}")
            continue

        t1, t2 = entries
        m1 = t1.get("MATCHUP")
        m2 = t2.get("MATCHUP")

        # MATCHUP must exist
        if m1 is None or m2 is None:
            print(f"Skipping {game_id}: MATCHUP missing")
            continue

        # Verify BOTH teams are NBA teams (double-check)
        team1_id = int(t1.get("TEAM_ID", 0))
        team2_id = int(t2.get("TEAM_ID", 0))
        
        if not (1610612737 <= team1_id <= 1610612766):
            continue
        if not (1610612737 <= team2_id <= 1610612766):
            continue

        # Determine home vs away by '@' marker
        # Example: "LAL @ BOS" --> LAL is away, BOS is home
        if "@" in m1:
            away = t1
            home = t2
        else:
            home = t1
            away = t2

        # JSON-safe conversion
        json_safe_entries = [
            {k: to_json_safe(v) for k, v in entry.items()}
            for entry in entries
        ]

        # Insert into database
        cur.execute("""
            INSERT OR REPLACE INTO games (
                game_id, game_date, season,
                home_team_id, away_team_id,
                home_team_abbr, away_team_abbr,
                home_score, away_score,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id,
            home["GAME_DATE"].strftime("%Y-%m-%d"),
            home["SEASON_ID"],

            home["TEAM_ID"],
            away["TEAM_ID"],

            home["TEAM_ABBREVIATION"],
            away["TEAM_ABBREVIATION"],

            home["PTS"],
            away["PTS"],

            json.dumps(json_safe_entries)
        ))

        inserted += 1

    conn.commit()
    conn.close()

    print(f"Inserted {inserted} REGULAR-SEASON games into the database.")

# --- Entry point ---
if __name__ == "__main__":
    import sys
    include_current = "--include-current" in sys.argv
    main(include_current_season=include_current)
