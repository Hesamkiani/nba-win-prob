"""
Phase 6: Backend Server (Flask + WebSocket)
-----------------------------------------------
Serves the dashboard (Phase 7) and streams live win-probability updates over
WebSocket. Since we don't have a real live game to poll (network-restricted
environment / presentation timing), this replays a REAL historical game from
our cached data, event by event, exactly like simulate_live.py — but instead
of printing to the terminal, it emits each update to the browser.

Run:
    python -m src.server
Then open:
    http://localhost:5000
"""

import time
import random
import logging
import threading

import pandas as pd
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO

from src.live_feed import load_inference_artifacts, predict_win_prob
from src.features import seconds_remaining_in_game, infer_possession, compute_fouls
from src.simulate_live import PBP_DIR, GAMES_INDEX_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__, template_folder="../dashboard/templates", static_folder="../dashboard/static")
app.config["SECRET_KEY"] = "nba-win-prob-demo"
# threading mode: no eventlet/gevent needed, works reliably on Windows for a demo
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")

# Loaded once at startup and reused for every simulated game
MODEL, SCALER, FEATURE_COLUMNS = load_inference_artifacts()

# Tracks the currently running simulation thread so we can stop/replace it
current_sim = {"thread": None, "stop_flag": False}


# ---------------------------------------------------------------------------
# Helpers: pick games, run the simulation loop
# ---------------------------------------------------------------------------
def get_games_index() -> pd.DataFrame:
    return pd.read_parquet(GAMES_INDEX_PATH)


def sample_random_games(n=20):
    """Just a random sample of games for the dashboard's game picker."""
    df = get_games_index()
    df["margin"] = (df["HOME_PTS"].astype(float) - df["AWAY_PTS"].astype(float)).abs()
    sample = df.sample(n=min(n, len(df)))
    return sample[["GAME_ID", "HOME_TEAM_ABBR", "AWAY_TEAM_ABBR", "HOME_PTS", "AWAY_PTS", "margin"]]


def run_simulation(game_id: str, speed: float, step: int):
    """Runs in a background thread. Emits a 'win_prob_update' event over
    WebSocket for every step, exactly mirroring simulate_live.py's logic."""
    games_index = get_games_index().set_index("GAME_ID")
    if game_id not in games_index.index:
        socketio.emit("sim_error", {"message": f"game_id {game_id} not found"})
        return

    game_row = games_index.loc[game_id]
    home_tricode = game_row["HOME_TEAM_ABBR"]
    away_tricode = game_row["AWAY_TEAM_ABBR"]
    actual_home_win = int(game_row["HOME_WIN"])

    pbp = pd.read_parquet(PBP_DIR / f"{game_id}.parquet")
    events = pbp.sort_values(["period", "actionNumber"]).reset_index(drop=True)
    events["scoreHome"] = pd.to_numeric(events["scoreHome"], errors="coerce").ffill().fillna(0)
    events["scoreAway"] = pd.to_numeric(events["scoreAway"], errors="coerce").ffill().fillna(0)

    possession_full = infer_possession(events, home_tricode, away_tricode).ffill().bfill()
    fouls_full = compute_fouls(events, home_tricode, away_tricode)

    socketio.emit("sim_start", {
        "game_id": game_id,
        "home_tricode": home_tricode,
        "away_tricode": away_tricode,
    })

    for i in range(1, len(events), step):
        if current_sim["stop_flag"]:
            log.info("Simulation stopped early (new simulation requested).")
            return

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
        prob_home = predict_win_prob(MODEL, SCALER, FEATURE_COLUMNS, feats)

        socketio.emit("win_prob_update", {
            "period": period,
            "clock": row["clock"],
            "home_score": int(row["scoreHome"]),
            "away_score": int(row["scoreAway"]),
            "possession_home": possession_home,
            "home_fouls": feats["home_team_fouls"],
            "away_fouls": feats["away_team_fouls"],
            "home_in_bonus": bool(feats["home_in_bonus"]),
            "away_in_bonus": bool(feats["away_in_bonus"]),
            "home_win_prob": round(prob_home * 100, 1),
            "away_win_prob": round((1 - prob_home) * 100, 1),
        })

        time.sleep(speed)

    socketio.emit("sim_end", {
        "actual_winner": home_tricode if actual_home_win else away_tricode,
        "home_tricode": home_tricode,
        "away_tricode": away_tricode,
    })
    log.info(f"Simulation finished for {game_id}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/games")
def api_games():
    """Returns a random shortlist of games for the dashboard's game picker."""
    games = sample_random_games(n=20)
    return jsonify(games.to_dict(orient="records"))


@app.route("/api/start", methods=["POST"])
def api_start():
    body = request.get_json(force=True)
    game_id = body.get("game_id")
    speed = float(body.get("speed", 0.15))
    step = int(body.get("step", 2))

    if not game_id:
        games_index = get_games_index()
        game_id = random.choice(games_index["GAME_ID"].tolist())

    # stop any currently running simulation before starting a new one
    current_sim["stop_flag"] = True
    if current_sim["thread"] is not None:
        current_sim["thread"].join(timeout=2)

    current_sim["stop_flag"] = False
    t = threading.Thread(target=run_simulation, args=(game_id, speed, step), daemon=True)
    current_sim["thread"] = t
    t.start()

    return jsonify({"status": "started", "game_id": game_id})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    current_sim["stop_flag"] = True
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    log.info("Starting NBA Win Probability server on http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)