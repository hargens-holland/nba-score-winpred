"""
Build matchup training samples from team sequences.

For every actual game played:
- Retrieve Team A's sequence for that game
- Retrieve Team B's sequence for that game
- Create input sample with sequences and outcomes
- Save as PyTorch tensors
"""
from nba_score_winpred.database.db import get_connection
import json
import numpy as np
import torch
import os
from typing import List, Dict, Optional


def load_sequence(team_id: int, game_id: str) -> Optional[np.ndarray]:
    """
    Load a team's sequence for a specific game.
    
    Args:
        team_id: Team ID
        game_id: Game ID
    
    Returns:
        Sequence as numpy array (window_size, num_features) or None if not found
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT sequence_json
        FROM team_sequences
        WHERE team_id = ? AND game_id = ?
    """, (team_id, game_id))
    
    result = cur.fetchone()
    conn.close()
    
    if result is None:
        return None
    
    try:
        sequence = json.loads(result[0])
        return np.array(sequence, dtype=np.float32)
    except Exception as e:
        print(f"Error loading sequence for team {team_id}, game {game_id}: {e}")
        return None


def build_matchup_samples():
    """
    Build training samples for all games in the database.
    Each sample contains sequences for both teams and the game outcome.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Get all games with their scores
    cur.execute("""
        SELECT game_id, game_date, season,
               home_team_id, away_team_id,
               home_score, away_score
        FROM games
        ORDER BY game_date, game_id
    """)
    
    games = cur.fetchall()
    conn.close()
    
    print(f"Building matchup samples for {len(games)} games...")
    
    samples = []
    skipped = 0
    skipped_reasons = {
        "no_home_sequence": 0,
        "no_away_sequence": 0,
        "invalid_sequence": 0
    }
    
    for game in games:
        game_id, game_date, season, home_team_id, away_team_id, home_score, away_score = game
        
        # Load sequences for both teams
        home_sequence = load_sequence(home_team_id, game_id)
        away_sequence = load_sequence(away_team_id, game_id)
        
        # Validate sequences
        if home_sequence is None:
            skipped_reasons["no_home_sequence"] += 1
            skipped += 1
            continue
        
        if away_sequence is None:
            skipped_reasons["no_away_sequence"] += 1
            skipped += 1
            continue
        
        # Validate sequence shapes
        if len(home_sequence.shape) != 2 or len(away_sequence.shape) != 2:
            skipped_reasons["invalid_sequence"] += 1
            skipped += 1
            continue
        
        if home_sequence.shape[0] != away_sequence.shape[0]:
            skipped_reasons["invalid_sequence"] += 1
            skipped += 1
            continue
        
        # Determine winner
        win_home = 1 if home_score > away_score else 0
        
        # Create sample
        sample = {
            "game_id": game_id,
            "game_date": game_date,
            "season": season,
            "teamA_id": home_team_id,
            "teamB_id": away_team_id,
            "sequenceA": home_sequence,
            "sequenceB": away_sequence,
            "scoreA": home_score,
            "scoreB": away_score,
            "winA": win_home
        }
        
        samples.append(sample)
        
        if len(samples) % 100 == 0:
            print(f"  Processed {len(samples)} samples, skipped {skipped}...")
    
    print(f"\nSample building complete!")
    print(f"  Created: {len(samples)} samples")
    print(f"  Skipped: {skipped} games")
    print(f"    - No home sequence: {skipped_reasons['no_home_sequence']}")
    print(f"    - No away sequence: {skipped_reasons['no_away_sequence']}")
    print(f"    - Invalid sequence: {skipped_reasons['invalid_sequence']}")
    
    return samples


def convert_to_tensors(samples: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Convert samples to PyTorch tensors for training.
    
    Args:
        samples: List of sample dictionaries
    
    Returns:
        Dictionary with batched tensors
    """
    print("\nConverting samples to tensors...")
    
    # Extract all sequences and targets
    sequences_a = []
    sequences_b = []
    scores_a = []
    scores_b = []
    wins_a = []
    
    for sample in samples:
        sequences_a.append(sample["sequenceA"])
        sequences_b.append(sample["sequenceB"])
        scores_a.append(sample["scoreA"])
        scores_b.append(sample["scoreB"])
        wins_a.append(sample["winA"])
    
    # Convert to numpy arrays first
    sequences_a = np.array(sequences_a, dtype=np.float32)
    sequences_b = np.array(sequences_b, dtype=np.float32)
    scores_a = np.array(scores_a, dtype=np.float32).reshape(-1, 1)
    scores_b = np.array(scores_b, dtype=np.float32).reshape(-1, 1)
    wins_a = np.array(wins_a, dtype=np.float32).reshape(-1, 1)
    
    # Replace NaN and Inf with 0
    sequences_a = np.nan_to_num(sequences_a, nan=0.0, posinf=0.0, neginf=0.0)
    sequences_b = np.nan_to_num(sequences_b, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Convert to PyTorch tensors
    tensor_dict = {
        "sequenceA": torch.from_numpy(sequences_a),
        "sequenceB": torch.from_numpy(sequences_b),
        "scoreA": torch.from_numpy(scores_a),
        "scoreB": torch.from_numpy(scores_b),
        "winA": torch.from_numpy(wins_a)
    }
    
    # Add metadata
    tensor_dict["num_samples"] = len(samples)
    tensor_dict["sequence_shape"] = sequences_a.shape[1:]  # (window_size, num_features)
    tensor_dict["num_features"] = sequences_a.shape[2]
    
    print(f"  Tensor shapes:")
    print(f"    sequenceA: {tensor_dict['sequenceA'].shape}")
    print(f"    sequenceB: {tensor_dict['sequenceB'].shape}")
    print(f"    scoreA: {tensor_dict['scoreA'].shape}")
    print(f"    scoreB: {tensor_dict['scoreB'].shape}")
    print(f"    winA: {tensor_dict['winA'].shape}")
    print(f"  Total samples: {tensor_dict['num_samples']}")
    print(f"  Sequence shape: {tensor_dict['sequence_shape']}")
    print(f"  Features per timestep: {tensor_dict['num_features']}")
    
    return tensor_dict


def save_dataset(tensor_dict: Dict[str, torch.Tensor], output_path: str):
    """
    Save dataset to file.
    
    Args:
        tensor_dict: Dictionary of tensors
        output_path: Path to save the .pt file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(tensor_dict, output_path)
    print(f"\nDataset saved to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / (1024**2):.2f} MB")


def main(output_path: str = "data/processed/training_data.pt"):
    """
    Main function to build and save the training dataset.
    
    Args:
        output_path: Path to save the training data
    """
    print("=" * 60)
    print("NBA Dataset Builder")
    print("=" * 60)
    
    # Build samples
    samples = build_matchup_samples()
    
    if len(samples) == 0:
        print("ERROR: No samples created. Check that sequences have been built.")
        return
    
    # Convert to tensors
    tensor_dict = convert_to_tensors(samples)
    
    # Save dataset
    save_dataset(tensor_dict, output_path)
    
    print("\n" + "=" * 60)
    print("Dataset building complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    output_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/training_data.pt"
    main(output_path)

