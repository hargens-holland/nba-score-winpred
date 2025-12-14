"""
Test script for the NBA data processing pipeline.

This script tests each component of the pipeline and verifies outputs.
"""
import sys
import os
import sqlite3
import json
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nba_score_winpred.database.db import get_connection, initialize_db
from nba_score_winpred.data_pipeline.ingest_games import main as ingest_main
from nba_score_winpred.features.extract_features import extract_all_features
from nba_score_winpred.features.build_sequences import build_all_sequences
from nba_score_winpred.dataset.dataset_builder import main as dataset_main


def test_database_initialization():
    """Test 1: Database initialization"""
    print("=" * 60)
    print("TEST 1: Database Initialization")
    print("=" * 60)
    
    try:
        initialize_db()
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Check tables exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        
        expected_tables = ['games', 'team_features', 'team_sequences']
        missing = [t for t in expected_tables if t not in tables]
        
        if missing:
            print(f"❌ FAILED: Missing tables: {missing}")
            return False
        
        print(f"✅ PASSED: All tables exist: {tables}")
        
        # Check schema
        for table in expected_tables:
            cur.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cur.fetchall()]
            print(f"   {table}: {len(columns)} columns")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_ingest_games(include_current=False):
    """Test 2: Game ingestion"""
    print("\n" + "=" * 60)
    print("TEST 2: Game Ingestion")
    print("=" * 60)
    
    try:
        # Run ingestion for a single season first (faster test)
        print("Ingesting games for 2023-24 season (test run)...")
        from nba_score_winpred.data_pipeline.ingest_games import main as ingest_main
        ingest_main(seasons=["2023-24"], include_current_season=False)
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Check games were inserted
        cur.execute("SELECT COUNT(*) FROM games")
        count = cur.fetchone()[0]
        
        if count == 0:
            print("❌ FAILED: No games inserted")
            conn.close()
            return False
        
        print(f"✅ PASSED: {count} games inserted")
        
        # Check sample game
        cur.execute("""
            SELECT game_id, game_date, home_team_abbr, away_team_abbr, 
                   home_score, away_score
            FROM games
            LIMIT 1
        """)
        sample = cur.fetchone()
        
        if sample:
            print(f"   Sample game: {sample[0]} on {sample[1]}")
            print(f"   {sample[2]} vs {sample[3]}: {sample[4]}-{sample[5]}")
        
        # Check raw_json exists
        cur.execute("SELECT COUNT(*) FROM games WHERE raw_json IS NOT NULL AND raw_json != ''")
        json_count = cur.fetchone()[0]
        print(f"   Games with raw_json: {json_count}/{count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test 3: Feature extraction"""
    print("\n" + "=" * 60)
    print("TEST 3: Feature Extraction")
    print("=" * 60)
    
    try:
        extract_all_features()
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Check features were inserted
        cur.execute("SELECT COUNT(*) FROM team_features")
        count = cur.fetchone()[0]
        
        if count == 0:
            print("❌ FAILED: No features extracted")
            conn.close()
            return False
        
        print(f"✅ PASSED: {count} team-game features extracted")
        
        # Check feature vector structure
        cur.execute("""
            SELECT feature_vector_json
            FROM team_features
            LIMIT 1
        """)
        sample = cur.fetchone()
        
        if sample:
            features = json.loads(sample[0])
            print(f"   Feature vector length: {len(features)}")
            
            if len(features) == 22:
                print("✅ PASSED: Correct feature vector length (22)")
            else:
                print(f"⚠️  WARNING: Expected 22 features, got {len(features)}")
        
        # Check for NaN/Inf
        cur.execute("""
            SELECT COUNT(*) FROM team_features
            WHERE feature_vector_json LIKE '%null%' 
               OR feature_vector_json LIKE '%NaN%'
               OR feature_vector_json LIKE '%Inf%'
        """)
        invalid_count = cur.fetchone()[0]
        
        if invalid_count > 0:
            print(f"⚠️  WARNING: {invalid_count} feature vectors contain invalid values")
        else:
            print("✅ PASSED: No invalid values in features")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sequence_building():
    """Test 4: Sequence building"""
    print("\n" + "=" * 60)
    print("TEST 4: Sequence Building")
    print("=" * 60)
    
    try:
        build_all_sequences()
        
        conn = get_connection()
        cur = conn.cursor()
        
        # Check sequences were created
        cur.execute("SELECT COUNT(*) FROM team_sequences")
        count = cur.fetchone()[0]
        
        if count == 0:
            print("❌ FAILED: No sequences created")
            conn.close()
            return False
        
        print(f"✅ PASSED: {count} sequences created")
        
        # Check sequence structure
        cur.execute("""
            SELECT sequence_json
            FROM team_sequences
            LIMIT 1
        """)
        sample = cur.fetchone()
        
        if sample:
            sequence = json.loads(sample[0])
            print(f"   Sequence shape: {len(sequence)} games × {len(sequence[0]) if sequence else 0} features")
            
            if len(sequence) == 10:
                print("✅ PASSED: Correct sequence length (10 games)")
            else:
                print(f"⚠️  WARNING: Expected 10 games, got {len(sequence)}")
            
            if sequence and len(sequence[0]) == 22:
                print("✅ PASSED: Correct feature count per game (22)")
            else:
                print(f"⚠️  WARNING: Expected 22 features, got {len(sequence[0]) if sequence else 0}")
        
        # Check teams with sequences
        cur.execute("SELECT COUNT(DISTINCT team_id) FROM team_sequences")
        team_count = cur.fetchone()[0]
        print(f"   Teams with sequences: {team_count}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_building():
    """Test 5: Dataset building"""
    print("\n" + "=" * 60)
    print("TEST 5: Dataset Building")
    print("=" * 60)
    
    try:
        output_path = "data/processed/test_training_data.pt"
        dataset_main(output_path)
        
        if not os.path.exists(output_path):
            print(f"❌ FAILED: Output file not created: {output_path}")
            return False
        
        print(f"✅ PASSED: Dataset file created: {output_path}")
        
        # Load and verify dataset
        data = torch.load(output_path)
        
        required_keys = ['sequenceA', 'sequenceB', 'scoreA', 'scoreB', 'winA']
        missing_keys = [k for k in required_keys if k not in data]
        
        if missing_keys:
            print(f"❌ FAILED: Missing keys: {missing_keys}")
            return False
        
        print("✅ PASSED: All required keys present")
        
        # Check tensor shapes
        seqA = data['sequenceA']
        seqB = data['sequenceB']
        scoreA = data['scoreA']
        scoreB = data['scoreB']
        winA = data['winA']
        
        print(f"   sequenceA shape: {seqA.shape}")
        print(f"   sequenceB shape: {seqB.shape}")
        print(f"   scoreA shape: {scoreA.shape}")
        print(f"   scoreB shape: {scoreB.shape}")
        print(f"   winA shape: {winA.shape}")
        
        # Validate shapes
        if len(seqA.shape) == 3 and seqA.shape[1] == 10 and seqA.shape[2] == 22:
            print("✅ PASSED: Correct sequence tensor shape (N, 10, 22)")
        else:
            print(f"⚠️  WARNING: Unexpected sequence shape: {seqA.shape}")
        
        if seqA.shape == seqB.shape:
            print("✅ PASSED: sequenceA and sequenceB have matching shapes")
        else:
            print(f"❌ FAILED: Shape mismatch between sequenceA and sequenceB")
            return False
        
        # Check for NaN/Inf
        if torch.isnan(seqA).any() or torch.isinf(seqA).any():
            print("⚠️  WARNING: NaN or Inf values found in sequences")
        else:
            print("✅ PASSED: No NaN or Inf values in tensors")
        
        # Check value ranges
        print(f"   Score range: {scoreA.min().item():.1f} - {scoreA.max().item():.1f}")
        print(f"   Win rate: {winA.mean().item():.2%}")
        
        print(f"\n✅ PASSED: Dataset validation complete")
        print(f"   Total samples: {data.get('num_samples', len(seqA))}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_queries():
    """Test 6: Database queries and data quality"""
    print("\n" + "=" * 60)
    print("TEST 6: Database Data Quality")
    print("=" * 60)
    
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Check games
        cur.execute("SELECT COUNT(*) FROM games")
        game_count = cur.fetchone()[0]
        print(f"   Total games: {game_count}")
        
        cur.execute("SELECT COUNT(DISTINCT season) FROM games")
        season_count = cur.fetchone()[0]
        print(f"   Seasons: {season_count}")
        
        # Check features
        cur.execute("SELECT COUNT(*) FROM team_features")
        feature_count = cur.fetchone()[0]
        print(f"   Team-game features: {feature_count}")
        
        # Check sequences
        cur.execute("SELECT COUNT(*) FROM team_sequences")
        sequence_count = cur.fetchone()[0]
        print(f"   Sequences: {sequence_count}")
        
        # Check coverage
        if game_count > 0:
            expected_features = game_count * 2  # 2 teams per game
            coverage = (feature_count / expected_features * 100) if expected_features > 0 else 0
            print(f"   Feature coverage: {coverage:.1f}%")
        
        # Check for missing data
        cur.execute("""
            SELECT COUNT(*) FROM games 
            WHERE raw_json IS NULL OR raw_json = ''
        """)
        missing_json = cur.fetchone()[0]
        
        if missing_json > 0:
            print(f"⚠️  WARNING: {missing_json} games missing raw_json")
        else:
            print("✅ PASSED: All games have raw_json")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def run_all_tests(include_current=False):
    """Run all tests in sequence"""
    print("\n" + "=" * 60)
    print("NBA DATA PIPELINE - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Test 1: Database
    results.append(("Database Initialization", test_database_initialization()))
    
    # Test 2: Ingestion (quick test with one season)
    results.append(("Game Ingestion", test_ingest_games(include_current)))
    
    # Test 3: Features
    results.append(("Feature Extraction", test_feature_extraction()))
    
    # Test 4: Sequences
    results.append(("Sequence Building", test_sequence_building()))
    
    # Test 5: Dataset
    results.append(("Dataset Building", test_dataset_building()))
    
    # Test 6: Data Quality
    results.append(("Data Quality", test_database_queries()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    include_current = "--include-current" in sys.argv
    run_all_tests(include_current=include_current)

