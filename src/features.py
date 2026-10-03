"""
Phase 2: Feature Engineering
-----------------------------
Turns raw play-by-play events into the 4 model features:
  1. score_diff          (home_score - away_score)
  2. seconds_remaining    (total seconds left in the game)
  3. possession           (1 = home team has the ball, 0 = away team)
  4. fouls / in_bonus     (team fouls this period, and bonus flag)

Every row also gets the game's final outcome as the label: home_win (1/0).

Run:
    python src/features.py
"""

import re
import logging
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PBP_DIR = RAW_DIR / "pbp"
GAMES_INDEX_PATH = RAW_DIR / "games_index.parquet"

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "features.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REGULATION_PERIOD_SECONDS = 12 * 60   # each of the first 4 quarters = 12 min
OT_PERIOD_SECONDS = 5 * 60            # each overtime period = 5 min
FOULS_TO_BONUS = 5                    # 5th team foul in a period puts opponent in the bonus

CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


# ---------------------------------------------------------------------------
# Feature 1 & building block: time remaining
# ---------------------------------------------------------------------------
def parse_clock_to_seconds(clock_str: str) -> float:
    """'PT11M23.00S' -> 683.0 seconds left IN THIS PERIOD."""
    match = CLOCK_RE.match(clock_str)
    if not match:
        return 0.0
    minutes, seconds = match.groups()
    return int(minutes) * 60 + float(seconds)


def seconds_remaining_in_game(period: int, clock_str: str) -> float:
    """
    Total seconds left in the ENTIRE game (not just this period).
    Regulation periods (1-4) are 12 min each; overtime periods (5+) are 5 min each.
    """
    seconds_left_this_period = parse_clock_to_seconds(clock_str)

    if period <= 4:
        # full periods still to come after this one
        remaining_full_periods = 4 - period
        return seconds_left_this_period + remaining_full_periods * REGULATION_PERIOD_SECONDS
    else:
        # we're in OT; no way to know in advance how many more OTs there will
        # be, so we only count the current OT period's remaining time
        return seconds_left_this_period


# ---------------------------------------------------------------------------
# Feature 3: possession inference
# ---------------------------------------------------------------------------
def infer_possession(events: pd.DataFrame, home_tricode: str, away_tricode: str) -> pd.Series:
    """
    Walks through events in order and tracks which team currently has the ball.
    Returns a Series of 1 (home has ball) / 0 (away has ball) aligned to `events`.

    Heuristic rules (this is the best inference possible from event types alone,
    the NBA doesn't publish a "possession" field directly):
      - Made shot / made final free throw -> ball goes to the OTHER team
      - Turnover -> ball goes to the OTHER team
      - Defensive rebound -> ball goes to the REBOUNDING team
      - Offensive rebound -> SAME team keeps the ball
      - Jump ball -> ball goes to whichever team is credited with the event
      - Fouls -> no possession change by themselves
      - Anything else (timeouts, substitutions, etc.) -> no change
    """
    possession = []
    current_team = None  # tricode of team currently holding the ball

    for _, row in events.iterrows():
        action = str(row.get("actionType", ""))
        sub_type = str(row.get("subType", ""))
        team = row.get("teamTricode", "")

        if action == "Jump Ball" and team:
            current_team = team

        elif action == "Made Shot" and team:
            # after a make, the OTHER team takes the ball
            current_team = away_tricode if team == home_tricode else home_tricode

        elif action == "Free Throw" and team:
            # only the last free throw in a trip matters for possession, and
            # we can't always tell "last" cleanly from this field alone, so
            # we treat every made FT the same as a made shot (safe approximation)
            if "MISS" not in sub_type.upper():
                current_team = away_tricode if team == home_tricode else home_tricode
            # missed FT -> wait for the Rebound event to resolve possession

        elif action == "Turnover" and team:
            current_team = away_tricode if team == home_tricode else home_tricode

        elif action == "Rebound" and team:
            if team == current_team:
                pass  # offensive rebound, same team keeps it
            else:
                current_team = team  # defensive rebound, ball flips

        # else: no possession-changing info in this event, keep current_team as-is

        possession.append(current_team)

    # Convert tricodes to 1 (home) / 0 (away) / NaN (unknown, e.g. start of game
    # before the opening jump ball result is known)
    poss_series = pd.Series(possession, index=events.index)
    return poss_series.map(lambda t: 1 if t == home_tricode else (0 if t == away_tricode else None))


# ---------------------------------------------------------------------------
# Feature 4: fouls & bonus
# ---------------------------------------------------------------------------
def compute_fouls(events: pd.DataFrame, home_tricode: str, away_tricode: str) -> pd.DataFrame:
    """
    Returns a DataFrame with running team-foul counts (reset every period) and
    an `in_bonus` boolean for each team, aligned to `events`.
    """
    home_fouls, away_fouls = [], []
    home_bonus, away_bonus = [], []

    h_count, a_count = 0, 0
    current_period = None

    for _, row in events.iterrows():
        period = row["period"]
        if period != current_period:
            # new period -> foul counts reset
            h_count, a_count = 0, 0
            current_period = period

        if row.get("actionType") == "Foul":
            team = row.get("teamTricode", "")
            if team == home_tricode:
                h_count += 1
            elif team == away_tricode:
                a_count += 1

        home_fouls.append(h_count)
        away_fouls.append(a_count)
        home_bonus.append(h_count >= FOULS_TO_BONUS)
        away_bonus.append(a_count >= FOULS_TO_BONUS)

    return pd.DataFrame({
        "home_team_fouls": home_fouls,
        "away_team_fouls": away_fouls,
        "home_in_bonus": home_bonus,
        "away_in_bonus": away_bonus,
    }, index=events.index)


# ---------------------------------------------------------------------------
# Put it all together for one game
# ---------------------------------------------------------------------------
def build_features_for_game(pbp: pd.DataFrame, home_tricode: str, away_tricode: str,
                             home_win: int) -> pd.DataFrame:
    events = pbp.sort_values(["period", "actionNumber"]).reset_index(drop=True)

    # Feature 1: score differential (home - away), forward-filled since not
    # every event updates the score
    events["scoreHome"] = pd.to_numeric(events["scoreHome"], errors="coerce").ffill().fillna(0)
    events["scoreAway"] = pd.to_numeric(events["scoreAway"], errors="coerce").ffill().fillna(0)
    score_diff = events["scoreHome"] - events["scoreAway"]

    # Feature 2: time remaining
    seconds_left = events.apply(
        lambda r: seconds_remaining_in_game(r["period"], r["clock"]), axis=1
    )

    # Feature 3: possession
    possession = infer_possession(events, home_tricode, away_tricode)

    # Feature 4: fouls / bonus
    fouls_df = compute_fouls(events, home_tricode, away_tricode)

    features = pd.DataFrame({
        "game_id": events["gameId"],
        "period": events["period"],
        "score_diff": score_diff,
        "seconds_remaining": seconds_left,
        "possession_home": possession,
        "home_team_fouls": fouls_df["home_team_fouls"],
        "away_team_fouls": fouls_df["away_team_fouls"],
        "home_in_bonus": fouls_df["home_in_bonus"],
        "away_in_bonus": fouls_df["away_in_bonus"],
        "home_win": home_win,
    })

    # possession is unknown for the first event or two (before the opening
    # jump ball resolves) -> forward/back fill, then drop any still-unknown rows
    features["possession_home"] = features["possession_home"].ffill().bfill()

    return features


# ---------------------------------------------------------------------------
# Main: loop over every cached game and build the master feature table
# ---------------------------------------------------------------------------
def main():
    games_index = pd.read_parquet(GAMES_INDEX_PATH)
    games_index = games_index.set_index("GAME_ID")

    all_features = []
    pbp_files = sorted(PBP_DIR.glob("*.parquet"))
    log.info(f"Building features for {len(pbp_files)} games...")

    for i, path in enumerate(pbp_files, start=1):
        game_id = path.stem
        if game_id not in games_index.index:
            continue

        game_row = games_index.loc[game_id]
        pbp = pd.read_parquet(path)
        if pbp.empty:
            continue

        try:
            feats = build_features_for_game(
                pbp,
                home_tricode=game_row["HOME_TEAM_ABBR"],
                away_tricode=game_row["AWAY_TEAM_ABBR"],
                home_win=int(game_row["HOME_WIN"]),
            )
            all_features.append(feats)
        except Exception as e:
            log.warning(f"game {game_id} failed feature build: {e}")

        if i % 200 == 0:
            log.info(f"Progress: {i}/{len(pbp_files)} games processed")

    master = pd.concat(all_features, ignore_index=True)
    master = master.dropna(subset=["possession_home"])  # drop rows we truly couldn't resolve

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Saved {len(master)} feature rows from {len(all_features)} games -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()