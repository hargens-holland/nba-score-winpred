from nba_api.stats.endpoints import leaguegamefinder
import pandas as pd

def get_games(season="2024-25"):
    """
    Pull all NBA games for a given season using the nba_api package.
    Returns a list of dicts.
    """
    print(f"Fetching NBA games for season {season}...")

    finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    df = finder.get_data_frames()[0]

    # Convert to datetime
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    print(f"Fetched {len(df)} games.")

    return df.to_dict(orient="records")
