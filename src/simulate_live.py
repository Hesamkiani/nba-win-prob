"""
Phase 5b: Live Game SIMULATOR (for demos when no real game is live)
-----------------------------------------------------------------------
Replays a real, already-finished game from our cached data as if it were
happening live: walks through its events in order, recomputes the model's
win probability at each step, and prints it out with a small delay so it
*feels* like a live broadcast.

This uses the exact same feature functions and model as the real live_feed.py
— the only difference is where the events come from (a cached file instead
of nba_api.live). Great for presentations/demos.

Run:
    python src/simulate_live.py                      # picks a random cached game
    python src/simulate_live.py --game_id 0022300001  # replay a specific game
    python src/simulate_live.py --speed 0.05          # faster playback (delay per event, seconds)
"""

import time
import random
import logging
import argparse
from pathlib import Path

import pandas as pd

from src.live_feed import load_inference_artifacts, predict_win_prob
from src.features import seconds_remaining_in_game, infer_possession, compute_fouls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PBP_DIR = RAW_DIR / "pbp"
GAMES_INDEX_PATH = RAW_DIR / "games_index.parquet"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def pick_game(game_id: str | None):
    games_index = pd.read_parquet(GAMES_INDEX_PATH).set_index("GAME_ID")

    if game_id is None:
        available = [p.stem for p in PBP_DIR.glob("*.parquet") if p.stem in games_index.index]
        game_id = random.choice(available)

    row = games_index.loc[game_id]
    pbp = pd.read_parquet(PBP_DIR / f"{game_id}.parquet")
    return game_id, row, pbp


def simulate(game_id: str | None, speed: float, step: int):
    model, scaler, feature_columns = load_inference_artifacts()
    game_id, game_row, pbp = pick_game(game_id)

    home_tricode = game_row["HOME_TEAM_ABBR"]
    away_tricode = game_row["AWAY_TEAM_ABBR"]
    actual_home_win = int(game_row["HOME_WIN"])

    events = pbp.sort_values(["period", "actionNumber"]).reset_index(drop=True)
    events["scoreHome"] = pd.to_numeric(events["scoreHome"], errors="coerce").ffill().fillna(0)
    events["scoreAway"] = pd.to_numeric(events["scoreAway"], errors="coerce").ffill().fillna(0)

    # Pre-compute possession & fouls for the WHOLE game up front (these need
    # the full event sequence anyway); we'll just reveal them incrementally
    # below to simulate the "live" feel.
    possession_full = infer_possession(events, home_tricode, away_tricode).ffill().bfill()
    fouls_full = compute_fouls(events, home_tricode, away_tricode)

    print(f"\n{'='*60}")
    print(f"  SIMULATING LIVE GAME: {away_tricode} @ {home_tricode}  (game_id={game_id})")
    print(f"{'='*60}\n")

    for i in range(1, len(events), step):
        row = events.iloc[i]
        period = int(row["period"])
        seconds_left = seconds_remaining_in_game(period, str(row["clock"]))
        score_diff = float(row["scoreHome"]) - float(row["scoreAway"])
        possession_home = int(possession_full.iloc[i])
        fouls_row = fouls_full.iloc[i]

        feats = {
            "score_diff": score_diff,
            "seconds_remaining": seconds_left,
            "possession_home": possession_home,
            "home_team_fouls": int(fouls_row["home_team_fouls"]),
            "away_team_fouls": int(fouls_row["away_team_fouls"]),
            "home_in_bonus": int(fouls_row["home_in_bonus"]),
            "away_in_bonus": int(fouls_row["away_in_bonus"]),
        }
        prob_home = predict_win_prob(model, scaler, feature_columns, feats)

        mins, secs = divmod(int(seconds_left if period <= 4 else parse_leftover(row["clock"])), 60)
        bar_len = 30
        filled = int(prob_home * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"Q{period} {row['clock']:>12} | "
            f"{away_tricode} {int(row['scoreAway'])} - {int(row['scoreHome'])} {home_tricode} | "
            f"{home_tricode} [{bar}] {prob_home*100:5.1f}%  {away_tricode} {(1-prob_home)*100:5.1f}%"
        )

        time.sleep(speed)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULT: {'HOME (' + home_tricode + ') WON' if actual_home_win else 'AWAY (' + away_tricode + ') WON'}")
    print(f"  Model's final probability for {home_tricode}: {prob_home*100:.1f}%")
    print(f"{'='*60}\n")


def parse_leftover(clock_str):
    # fallback used only for display formatting in OT periods
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_id", type=str, default=None, help="Specific game to replay (default: random)")
    parser.add_argument("--speed", type=float, default=0.05, help="Seconds to sleep between printed steps")
    parser.add_argument("--step", type=int, default=3, help="Print every Nth event (lower = smoother, slower)")
    args = parser.parse_args()

    simulate(args.game_id, args.speed, args.step)