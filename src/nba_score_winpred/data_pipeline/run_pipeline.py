"""
Master pipeline script to run the entire data processing workflow.

Usage:
    python -m nba_score_winpred.data_pipeline.run_pipeline [--include-current]
"""
import sys
from nba_score_winpred.database.db import initialize_db
from nba_score_winpred.data_pipeline.ingest_games import main as ingest_main
from nba_score_winpred.features.extract_features import extract_all_features
from nba_score_winpred.features.build_sequences import build_all_sequences
from nba_score_winpred.dataset.dataset_builder import main as dataset_main


def run_full_pipeline(include_current_season=False):
    """
    Run the complete data processing pipeline.
    
    Steps:
    1. Initialize/reset database
    2. Ingest games for multiple seasons
    3. Extract features from raw JSON
    4. Build sequences for each team
    5. Build matchup training samples
    """
    print("=" * 60)
    print("NBA Data Processing Pipeline")
    print("=" * 60)
    
    # Step 1: Initialize database
    print("\n[Step 1/5] Initializing database...")
    initialize_db()
    
    # Step 2: Ingest games
    print("\n[Step 2/5] Ingesting games from NBA API...")
    ingest_main(include_current_season=include_current_season)
    
    # Step 3: Extract features
    print("\n[Step 3/5] Extracting features from raw JSON...")
    extract_all_features()
    
    # Step 4: Build sequences
    print("\n[Step 4/5] Building sequences for each team...")
    build_all_sequences()
    
    # Step 5: Build dataset
    print("\n[Step 5/5] Building training dataset...")
    dataset_main()
    
    print("\n" + "=" * 60)
    print("Pipeline complete! Training data ready at: data/processed/training_data.pt")
    print("=" * 60)


if __name__ == "__main__":
    include_current = "--include-current" in sys.argv
    run_full_pipeline(include_current_season=include_current)

