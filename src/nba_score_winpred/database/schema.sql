CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    game_date TEXT,
    season TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_team_abbr TEXT,
    away_team_abbr TEXT,
    home_score INTEGER,
    away_score INTEGER,
    raw_json TEXT
);


CREATE TABLE IF NOT EXISTS team_features (
    team_id INTEGER,
    game_id TEXT,
    game_date TEXT,
    feature_vector_json TEXT,
    PRIMARY KEY (team_id, game_id)
);

CREATE TABLE IF NOT EXISTS team_sequences (
    team_id INTEGER,
    game_id TEXT,
    sequence_json TEXT,
    PRIMARY KEY (team_id, game_id)
);
