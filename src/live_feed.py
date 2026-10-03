"""
Phase 5: Live Game Feed + Real-Time Inference
------------------------------------------------
Pulls the play-by-play of an in-progress NBA game from nba_api's LIVE
endpoint, recomputes the exact same 4 features used in training (Phase 2),
and runs the trained model (Phase 4) to produce a live win probability.

This file does NOT run a server or dashboard yet (that's Phase 6/7) — it's
the core "given a game_id, what's the win probability right now?" engine
that the server will call every few seconds.

Run (prints live win prob every 5s for a game currently being played):
    python src/live_feed.py --game_id 0022500123

List today's live/scheduled games:
    python src/live_feed.py --list
"""
import sys
from pathlib import Path

# اضافه کردن ریشه پروژه به مسیر جستجوی پایتون
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
import time
import logging
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from nba_api.live.nba.endpoints import playbyplay as live_playbyplay
from nba_api.live.nba.endpoints import scoreboard as live_scoreboard

# Re-use the EXACT same feature-engineering functions from Phase 2.
# This is critical: if live features were computed even slightly differently
# than training features, the model's predictions would be meaningless.
from src.features import (
    seconds_remaining_in_game,
    infer_possession,
    compute_fouls,
)
from src.train import WinProbNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

POLL_INTERVAL_SECONDS = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load model + scaler + feature column order (all saved back in Phase 3/4)
# ---------------------------------------------------------------------------
def load_inference_artifacts():
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.joblib")

    # n_features must match what the model was trained with
    model = WinProbNet(n_features=len(feature_columns))
    model.load_state_dict(torch.load(MODELS_DIR / "win_prob_net.pt", map_location="cpu"))
    model.eval()

    log.info(f"Loaded model, scaler, and feature order: {feature_columns}")
    return model, scaler, feature_columns


# ---------------------------------------------------------------------------
# Step 1: find today's games (so we know which game_id + home/away tricodes)
# ---------------------------------------------------------------------------
def list_todays_games():
    board = live_scoreboard.ScoreBoard()
    games = board.get_dict()["scoreboard"]["games"]
    out = []
    for g in games:
        out.append({
            "game_id": g["gameId"],
            "status": g["gameStatusText"],       # e.g. "7:00 pm ET", "Q3 4:12", "Final"
            "home_tricode": g["homeTeam"]["teamTricode"],
            "away_tricode": g["awayTeam"]["teamTricode"],
            "home_score": g["homeTeam"]["score"],
            "away_score": g["awayTeam"]["score"],
        })
    return out


# ---------------------------------------------------------------------------
# Step 2: pull live play-by-play and compute the same features as training
# ---------------------------------------------------------------------------
def fetch_live_events(game_id: str) -> pd.DataFrame:
    pbp = live_playbyplay.PlayByPlay(game_id=game_id)
    actions = pbp.get_dict()["game"]["actions"]
    return pd.DataFrame(actions)


def compute_current_features(events: pd.DataFrame, home_tricode: str, away_tricode: str) -> dict:
    """
    Takes the full list of events so far in a live game and returns a single
    dict of feature values representing the CURRENT (most recent) game state
    — i.e. the same feature snapshot the model saw one row of during training.
    """
    events = events.sort_values("actionNumber").reset_index(drop=True)

    events["scoreHome"] = pd.to_numeric(events["scoreHome"], errors="coerce").ffill().fillna(0)
    events["scoreAway"] = pd.to_numeric(events["scoreAway"], errors="coerce").ffill().fillna(0)

    last = events.iloc[-1]
    score_diff = float(last["scoreHome"]) - float(last["scoreAway"])
    seconds_left = seconds_remaining_in_game(int(last["period"]), str(last["clock"]))

    possession_series = infer_possession(events, home_tricode, away_tricode)
    possession_home = possession_series.ffill().bfill().iloc[-1]
    possession_home = 1 if possession_home == 1 else 0  # default to 0 if still unknown

    fouls_df = compute_fouls(events, home_tricode, away_tricode)
    fouls_last = fouls_df.iloc[-1]

    return {
        "score_diff": score_diff,
        "seconds_remaining": seconds_left,
        "possession_home": possession_home,
        "home_team_fouls": int(fouls_last["home_team_fouls"]),
        "away_team_fouls": int(fouls_last["away_team_fouls"]),
        "home_in_bonus": int(fouls_last["home_in_bonus"]),
        "away_in_bonus": int(fouls_last["away_in_bonus"]),
        # extra context, not fed to the model, just useful for display
        "period": int(last["period"]),
        "home_score": int(last["scoreHome"]),
        "away_score": int(last["scoreAway"]),
    }


# ---------------------------------------------------------------------------
# Step 3: run the model
# ---------------------------------------------------------------------------
def predict_win_prob(model, scaler, feature_columns, feature_dict: dict) -> float:
    x = np.array([[feature_dict[col] for col in feature_columns]], dtype=np.float32)
    x_scaled = scaler.transform(x)
    x_tensor = torch.from_numpy(x_scaled.astype(np.float32))

    with torch.no_grad():
        logit = model(x_tensor)
        prob_home_win = torch.sigmoid(logit).item()

    return prob_home_win


# ---------------------------------------------------------------------------
# Main polling loop
# ---------------------------------------------------------------------------
def watch_game(game_id: str, home_tricode: str, away_tricode: str):
    model, scaler, feature_columns = load_inference_artifacts()

    log.info(f"Watching game {game_id} ({away_tricode} @ {home_tricode})... polling every {POLL_INTERVAL_SECONDS}s")
    while True:
        try:
            events = fetch_live_events(game_id)
            if events.empty:
                log.info("No events yet (game hasn't started).")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            feats = compute_current_features(events, home_tricode, away_tricode)
            prob_home = predict_win_prob(model, scaler, feature_columns, feats)

            log.info(
                f"Q{feats['period']} | {away_tricode} {feats['away_score']} - "
                f"{feats['home_score']} {home_tricode} | "
                f"{home_tricode} win prob: {prob_home*100:.1f}% | "
                f"{away_tricode} win prob: {(1-prob_home)*100:.1f}%"
            )
        except Exception as e:
            log.warning(f"Error fetching/predicting: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_id", type=str, help="NBA live game_id to watch")
    parser.add_argument("--list", action="store_true", help="List today's games and exit")
    args = parser.parse_args()

    if args.list:
        for g in list_todays_games():
            print(g)
    elif args.game_id:
        games = {g["game_id"]: g for g in list_todays_games()}
        if args.game_id not in games:
            raise SystemExit(f"game_id {args.game_id} not found in today's scoreboard. Use --list to see options.")
        g = games[args.game_id]
        watch_game(args.game_id, g["home_tricode"], g["away_tricode"])
    else:
        parser.print_help()