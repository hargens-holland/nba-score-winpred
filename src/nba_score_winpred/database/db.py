import sqlite3
import os

DB_PATH = "data/database/nba.db"

def get_connection():
    os.makedirs("data/database", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn

def initialize_db():
    conn = get_connection()
    cur = conn.cursor()

    # Drop existing tables to ensure clean schema
    cur.execute("DROP TABLE IF EXISTS team_sequences")
    cur.execute("DROP TABLE IF EXISTS team_features")
    cur.execute("DROP TABLE IF EXISTS games")
    
    schema_path = "src/nba_score_winpred/database/schema.sql"
    with open(schema_path, "r") as f:
        schema = f.read()

    cur.executescript(schema)
    conn.commit()
    conn.close()

    print("Database initialized:", DB_PATH)
