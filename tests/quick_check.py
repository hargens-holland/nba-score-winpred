#!/usr/bin/env python3
"""
Quick verification script for the NBA data pipeline.

Run this after the pipeline to quickly verify everything is working.
"""
import os
import sqlite3
import torch

def check_database():
    """Check database exists and has data"""
    db_path = "data/database/nba.db"
    
    if not os.path.exists(db_path):
        print("❌ Database not found:", db_path)
        return False
    
    print(f"✅ Database exists: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Check tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        print(f"   Tables: {', '.join(tables)}")
        
        # Count rows
        if 'games' in tables:
            cur.execute("SELECT COUNT(*) FROM games")
            games = cur.fetchone()[0]
            print(f"   Games: {games}")
        
        if 'team_features' in tables:
            cur.execute("SELECT COUNT(*) FROM team_features")
            features = cur.fetchone()[0]
            print(f"   Features: {features}")
        
        if 'team_sequences' in tables:
            cur.execute("SELECT COUNT(*) FROM team_sequences")
            sequences = cur.fetchone()[0]
            print(f"   Sequences: {sequences}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


def check_dataset():
    """Check dataset file exists and is valid"""
    dataset_path = "data/processed/training_data.pt"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found:", dataset_path)
        return False
    
    print(f"✅ Dataset exists: {dataset_path}")
    
    try:
        data = torch.load(dataset_path)
        
        # Check required keys
        required = ['sequenceA', 'sequenceB', 'scoreA', 'scoreB', 'winA']
        missing = [k for k in required if k not in data]
        
        if missing:
            print(f"❌ Missing keys: {missing}")
            return False
        
        # Check shapes
        seqA = data['sequenceA']
        seqB = data['sequenceB']
        
        print(f"   Samples: {len(seqA)}")
        print(f"   Sequence shape: {seqA.shape[1:]}")
        print(f"   Features: {seqA.shape[2]}")
        
        # Check for NaN/Inf
        if torch.isnan(seqA).any() or torch.isinf(seqA).any():
            print("⚠️  Warning: NaN or Inf values found")
        else:
            print("✅ No NaN/Inf values")
        
        # Check score ranges
        scoreA = data['scoreA']
        print(f"   Score range: {scoreA.min().item():.1f} - {scoreA.max().item():.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return False


def main():
    print("=" * 60)
    print("Quick Pipeline Check")
    print("=" * 60)
    
    db_ok = check_database()
    print()
    dataset_ok = check_dataset()
    
    print("\n" + "=" * 60)
    if db_ok and dataset_ok:
        print("✅ All checks passed! Pipeline is ready.")
    else:
        print("❌ Some checks failed. See details above.")
    print("=" * 60)


if __name__ == "__main__":
    main()

