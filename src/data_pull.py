"""
Phase 1: Data Collection
-------------------------
Pulls play-by-play logs for every game across the target NBA seasons
(all 30 teams included automatically, since LeagueGameFinder returns
every game league-wide) and caches each game to disk as parquet so we
never have to re-hit the API.

Run:
    python src/data_pull.py
"""

import time
import random
import logging
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3
from nba_api.stats.library.parameters import SeasonType, SeasonTypePlayoffs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEASONS = ["2023-24", "2024-25"]          # <-- 2 seasons, as decided
SEASON_TYPES = [SeasonType.regular, SeasonTypePlayoffs.playoffs]  # <-- both, as decided

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
GAMES_INDEX_PATH = RAW_DIR / "games_index.parquet"
PBP_DIR = RAW_DIR / "pbp"

MIN_DELAY = 0.6   # seconds between API calls (be polite to stats.nba.com)
MAX_DELAY = 1.2
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Build the list of games (all teams, both seasons)
# ---------------------------------------------------------------------------
def fetch_games_index() -> pd.DataFrame:
    """
    Returns a deduplicated DataFrame of every game_id across the configured
    seasons, with home/away team info and the final score (used later for
    labeling win/loss).
    """
    if GAMES_INDEX_PATH.exists():
        log.info(f"Games index already cached at {GAMES_INDEX_PATH}, loading it.")
        return pd.read_parquet(GAMES_INDEX_PATH)

    all_games = []
    for season in SEASONS:
        for season_type in SEASON_TYPES:
            log.info(f"Fetching game index for season {season} ({season_type})...")
            gf = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable=season_type,
                league_id_nullable="00",  # NBA
            )
            df = gf.get_data_frames()[0]
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            all_games.append(df)
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    games = pd.concat(all_games, ignore_index=True)

    # LeagueGameFinder returns one row PER TEAM per game (so each game_id
    # appears twice: once for home, once for away). Collapse into one row
    # per game with both teams' info.
    games["IS_HOME"] = games["MATCHUP"].str.contains("vs.")
    home = games[games["IS_HOME"]].copy()
    away = games[~games["IS_HOME"]].copy()

    home = home.rename(columns={
        "TEAM_ID": "HOME_TEAM_ID", "TEAM_ABBREVIATION": "HOME_TEAM_ABBR",
        "PTS": "HOME_PTS", "WL": "HOME_WL",
    })[["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE", "HOME_TEAM_ID", "HOME_TEAM_ABBR", "HOME_PTS", "HOME_WL"]]

    away = away.rename(columns={
        "TEAM_ID": "AWAY_TEAM_ID", "TEAM_ABBREVIATION": "AWAY_TEAM_ABBR",
        "PTS": "AWAY_PTS",
    })[["GAME_ID", "AWAY_TEAM_ID", "AWAY_TEAM_ABBR", "AWAY_PTS"]]

    games_index = home.merge(away, on="GAME_ID", how="inner")
    games_index["HOME_WIN"] = (games_index["HOME_WL"] == "W").astype(int)
    # GAME_ID is globally unique per game (NBA encodes season-type in the ID
    # itself, e.g. "002" prefix = regular season, "004" = playoffs), so a
    # simple dedup here is safe even with both season types combined.
    games_index = games_index.drop_duplicates(subset="GAME_ID").reset_index(drop=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    games_index.to_parquet(GAMES_INDEX_PATH, index=False)
    log.info(f"Saved games index: {len(games_index)} games -> {GAMES_INDEX_PATH}")
    return games_index


# ---------------------------------------------------------------------------
# Step 2: Pull play-by-play for each game, with retry + per-game caching
# ---------------------------------------------------------------------------
def fetch_pbp_for_game(game_id: str) -> pd.DataFrame | None:
    out_path = PBP_DIR / f"{game_id}.parquet"
    if out_path.exists():
        return None  # already cached, skip

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            pbp = playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30)
            # Grab the "PlayByPlay" data set explicitly (not get_data_frames()[0]/[1],
            # since dict ordering of the two data sets - PlayByPlay & AvailableVideo -
            # isn't guaranteed).
            df = pbp.play_by_play.get_data_frame()
            if df.empty:
                raise ValueError("empty PlayByPlay result (game may not have PBP data)")
            df.to_parquet(out_path, index=False)
            return df
        except Exception as e:
            wait = attempt * 2
            log.warning(f"game {game_id} attempt {attempt} failed ({e}); retrying in {wait}s")
            time.sleep(wait)
    log.error(f"game {game_id} failed after {MAX_RETRIES} attempts, skipping.")
    return None


def fetch_all_pbp(games_index: pd.DataFrame):
    PBP_DIR.mkdir(parents=True, exist_ok=True)
    total = len(games_index)
    for i, game_id in enumerate(games_index["GAME_ID"], start=1):
        out_path = PBP_DIR / f"{game_id}.parquet"
        if out_path.exists():
            continue  # resume-safe: skip already-pulled games
        fetch_pbp_for_game(game_id)
        if i % 25 == 0:
            log.info(f"Progress: {i}/{total} games processed")
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    log.info("Play-by-play pull complete.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    games_index = fetch_games_index()
    log.info(f"Total games to pull PBP for: {len(games_index)}")
    fetch_all_pbp(games_index)
