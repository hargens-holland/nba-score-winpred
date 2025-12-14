"""
Extract per-game features for each team from raw_json data.

Computes 22 features per team per game:
- Shooting/Scoring (5)
- Volume Stats (5)
- Rebounding (3)
- Defense Indicators (3)
- Pace + Possessions (2)
- Context (4)
"""
from nba_score_winpred.database.db import get_connection
import json
import numpy as np
from typing import Dict, List, Optional


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide, returning default if denominator is 0 or NaN."""
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return default
    return numerator / denominator


def extract_features_from_raw_json(raw_json_str: str, team_id: int, opponent_id: int) -> Optional[Dict]:
    """
    Extract features for a team from raw_json.
    
    Args:
        raw_json_str: JSON string containing stats for both teams
        team_id: The team ID to extract features for
        opponent_id: The opponent team ID
    
    Returns:
        Dictionary with feature vector and metadata, or None if extraction fails
    """
    try:
        entries = json.loads(raw_json_str)
        if len(entries) != 2:
            return None
        
        # Find team and opponent entries
        team_entry = None
        opp_entry = None
        
        for entry in entries:
            if int(entry.get("TEAM_ID", 0)) == team_id:
                team_entry = entry
            elif int(entry.get("TEAM_ID", 0)) == opponent_id:
                opp_entry = entry
        
        if team_entry is None or opp_entry is None:
            return None
        
        # Extract basic stats
        pts = float(team_entry.get("PTS", 0))
        fgm = float(team_entry.get("FGM", 0))
        fga = float(team_entry.get("FGA", 0))
        fg3m = float(team_entry.get("FG3M", 0))
        fg3a = float(team_entry.get("FG3A", 0))
        ftm = float(team_entry.get("FTM", 0))
        fta = float(team_entry.get("FTA", 0))
        ast = float(team_entry.get("AST", 0))
        tov = float(team_entry.get("TOV", 0))
        oreb = float(team_entry.get("OREB", 0))
        dreb = float(team_entry.get("DREB", 0))
        reb = float(team_entry.get("REB", 0))
        stl = float(team_entry.get("STL", 0))
        blk = float(team_entry.get("BLK", 0))
        pf = float(team_entry.get("PF", 0))
        plus_minus = float(team_entry.get("PLUS_MINUS", 0))
        
        # Opponent stats
        opp_fgm = float(opp_entry.get("FGM", 0))
        opp_fga = float(opp_entry.get("FGA", 0))
        opp_fta = float(opp_entry.get("FTA", 0))
        opp_oreb = float(opp_entry.get("OREB", 0))
        opp_tov = float(opp_entry.get("TOV", 0))
        opp_pts = float(opp_entry.get("PTS", 0))
        
        # Minutes played (default to 48 for full game)
        min_played = float(team_entry.get("MIN", 48))
        if min_played == 0 or np.isnan(min_played):
            min_played = 48.0
        
        # A. Shooting / Scoring
        fg_pct = safe_divide(fgm, fga)
        fg3_pct = safe_divide(fg3m, fg3a)
        ft_pct = safe_divide(ftm, fta)
        efg_pct = safe_divide(fgm + 0.5 * fg3m, fga) if fga > 0 else 0.0
        ts_pct = safe_divide(pts, 2 * (fga + 0.44 * fta)) if (fga + 0.44 * fta) > 0 else 0.0
        
        # B. Volume Stats
        # fga, fg3a, fta, ast, tov already extracted
        
        # C. Rebounding
        # oreb, dreb, reb already extracted
        
        # D. Defense Indicators
        # stl, blk, pf already extracted
        
        # E. Pace + Possessions
        team_poss = fga + 0.4 * fta - oreb + tov
        opp_poss = opp_fga + 0.4 * opp_fta - opp_oreb + opp_tov
        poss = 0.5 * (team_poss + opp_poss)
        pace = poss * (48.0 / min_played) if min_played > 0 else 0.0
        
        # F. Context
        # plus_minus already extracted
        # HOME and WIN will be determined from game context
        
        # Build feature vector (20 base features, HOME and WIN added later = 22 total)
        feature_vector = [
            # A. Shooting / Scoring (6)
            pts,
            fg_pct,
            fg3_pct,
            ft_pct,
            efg_pct,
            ts_pct,
            # B. Volume Stats (5)
            fga,
            fg3a,
            fta,
            ast,
            tov,
            # C. Rebounding (3)
            oreb,
            dreb,
            reb,
            # D. Defense Indicators (3)
            stl,
            blk,
            pf,
            # E. Pace + Possessions (2)
            poss,
            pace,
            # F. Context (1 - plus_minus; HOME and WIN added later)
            plus_minus,
        ]
        
        # Replace NaN and Inf with 0
        feature_vector = [0.0 if (np.isnan(x) or np.isinf(x)) else float(x) for x in feature_vector]
        
        return {
            "feature_vector": feature_vector,
            "pts": pts,
            "opp_pts": opp_pts
        }
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def extract_all_features():
    """
    Extract features for all games in the database.
    Processes each game and extracts features for both home and away teams.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Get all games
    cur.execute("""
        SELECT game_id, game_date, season, 
               home_team_id, away_team_id,
               home_team_abbr, away_team_abbr,
               home_score, away_score,
               raw_json
        FROM games
        ORDER BY game_date, game_id
    """)
    
    games = cur.fetchall()
    print(f"Processing {len(games)} games for feature extraction...")
    
    inserted = 0
    skipped = 0
    
    for game in games:
        game_id, game_date, season, home_team_id, away_team_id, \
        home_team_abbr, away_team_abbr, home_score, away_score, raw_json = game
        
        # Extract features for home team
        home_features = extract_features_from_raw_json(raw_json, home_team_id, away_team_id)
        if home_features:
            # Add HOME=1 and WIN
            home_feature_vec = home_features["feature_vector"] + [
                1.0,  # HOME
                1.0 if home_score > away_score else 0.0  # WIN
            ]
            
            cur.execute("""
                INSERT OR REPLACE INTO team_features 
                (team_id, game_id, game_date, feature_vector_json)
                VALUES (?, ?, ?, ?)
            """, (
                home_team_id,
                game_id,
                game_date,
                json.dumps(home_feature_vec)
            ))
            inserted += 1
        else:
            skipped += 1
        
        # Extract features for away team
        away_features = extract_features_from_raw_json(raw_json, away_team_id, home_team_id)
        if away_features:
            # Add HOME=0 and WIN
            away_feature_vec = away_features["feature_vector"] + [
                0.0,  # HOME
                1.0 if away_score > home_score else 0.0  # WIN
            ]
            
            cur.execute("""
                INSERT OR REPLACE INTO team_features 
                (team_id, game_id, game_date, feature_vector_json)
                VALUES (?, ?, ?, ?)
            """, (
                away_team_id,
                game_id,
                game_date,
                json.dumps(away_feature_vec)
            ))
            inserted += 1
        else:
            skipped += 1
        
        if (inserted + skipped) % 100 == 0:
            print(f"Processed {inserted + skipped} team-games ({inserted} inserted, {skipped} skipped)")
    
    conn.commit()
    conn.close()
    
    print(f"\nFeature extraction complete!")
    print(f"  Inserted: {inserted} team-game features")
    print(f"  Skipped: {skipped} team-games (missing data)")
    print(f"  Total feature vector length: 22 (20 base + HOME + WIN)")


if __name__ == "__main__":
    extract_all_features()

