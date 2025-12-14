"""
Build chronological sequences for each team.

For each team, loads their games ordered by game_date,
extracts feature vectors, and computes rolling windows of length 10 games.
Each sequence shape: (10, num_features)
"""
from nba_score_winpred.database.db import get_connection
import json
from typing import List, Optional


def build_sequences_for_team(team_id: int, window_size: int = 10) -> int:
    """
    Build sequences for a single team.
    
    Args:
        team_id: Team ID to build sequences for
        window_size: Number of games in each sequence (default 10)
    
    Returns:
        Number of sequences created
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Get all games for this team, ordered by date
    cur.execute("""
        SELECT game_id, game_date, feature_vector_json
        FROM team_features
        WHERE team_id = ?
        ORDER BY game_date, game_id
    """, (team_id,))
    
    games = cur.fetchall()
    
    if len(games) < window_size:
        conn.close()
        return 0
    
    sequences_created = 0
    
    # Build rolling windows
    for i in range(window_size - 1, len(games)):
        # Get window of games
        window_games = games[i - window_size + 1:i + 1]
        
        # Extract feature vectors
        sequence = []
        game_ids_in_sequence = []
        
        for game_id, game_date, feature_vector_json in window_games:
            try:
                feature_vec = json.loads(feature_vector_json)
                sequence.append(feature_vec)
                game_ids_in_sequence.append(game_id)
            except Exception as e:
                print(f"Error parsing features for game {game_id}: {e}")
                conn.close()
                return sequences_created
        
        # Validate sequence length
        if len(sequence) != window_size:
            continue
        
        # Validate all feature vectors have same length
        feature_lengths = [len(fv) for fv in sequence]
        if len(set(feature_lengths)) != 1:
            print(f"Warning: Inconsistent feature lengths for team {team_id} at game {game_ids_in_sequence[-1]}")
            continue
        
        # Store sequence for the current game (last game in window)
        current_game_id = game_ids_in_sequence[-1]
        
        cur.execute("""
            INSERT OR REPLACE INTO team_sequences
            (team_id, game_id, sequence_json)
            VALUES (?, ?, ?)
        """, (
            team_id,
            current_game_id,
            json.dumps(sequence)
        ))
        
        sequences_created += 1
    
    conn.commit()
    conn.close()
    return sequences_created


def build_all_sequences(window_size: int = 10):
    """
    Build sequences for all teams in the database.
    
    Args:
        window_size: Number of games in each sequence (default 10)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Get all unique team IDs
    cur.execute("SELECT DISTINCT team_id FROM team_features ORDER BY team_id")
    team_ids = [row[0] for row in cur.fetchall()]
    
    conn.close()
    
    print(f"Building sequences for {len(team_ids)} teams (window_size={window_size})...")
    
    total_sequences = 0
    teams_with_sequences = 0
    teams_skipped = 0
    
    for team_id in team_ids:
        sequences = build_sequences_for_team(team_id, window_size)
        
        if sequences > 0:
            total_sequences += sequences
            teams_with_sequences += 1
        else:
            teams_skipped += 1
        
        if (teams_with_sequences + teams_skipped) % 5 == 0:
            print(f"  Processed {teams_with_sequences + teams_skipped} teams...")
    
    print(f"\nSequence building complete!")
    print(f"  Teams with sequences: {teams_with_sequences}")
    print(f"  Teams skipped (insufficient games): {teams_skipped}")
    print(f"  Total sequences created: {total_sequences}")
    print(f"  Sequence shape: ({window_size}, num_features)")


if __name__ == "__main__":
    build_all_sequences()

