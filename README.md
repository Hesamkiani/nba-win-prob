# NBA Live Win Probability Model

A live win-probability model for NBA games, trained on real play-by-play data
and served through a real-time web dashboard.

Given the current state of a game — score differential, time remaining,
possession, and fouls — the model outputs each team's probability of winning,
updated live as the game progresses.

---

## What this project does

1. **Collects** play-by-play data for ~2,600 real NBA games (2 seasons, regular season + playoffs) via [`nba_api`](https://github.com/swar/nba_api)
2. **Engineers** 4 features from raw events: score differential, seconds remaining, possession, and team fouls (with bonus detection)
3. **Trains** a PyTorch feedforward neural network to predict home-team win probability
4. **Serves** the trained model through a Flask + WebSocket server
5. **Displays** live win probability on a web dashboard, by replaying real historical games as if they were happening live (since polling an actual live game requires an NBA game in progress)

## Tech stack

| Layer | Tool |
|---|---|
| Data collection | `nba_api` |
| Data processing | `pandas`, `numpy` |
| Storage | Parquet (`pyarrow`) |
| Train/test split, scaling | `scikit-learn` |
| Model | `PyTorch` |
| Backend | `Flask`, `flask-socketio` |
| Real-time transport | WebSocket |
| Frontend | HTML/CSS/JS, Chart.js |

## Project structure

```
nba-win-prob/
├── data/
│   ├── raw/            # cached play-by-play + game index (gitignored, regenerate locally)
│   └── processed/      # engineered feature table + train/test tensors (gitignored)
├── models/              # trained model weights + scaler (gitignored)
├── src/
│   ├── data_pull.py     # Phase 1: pull & cache play-by-play from nba_api
│   ├── features.py      # Phase 2: feature engineering (possession inference, fouls/bonus, etc.)
│   ├── split_data.py    # Phase 3: group-aware train/test split + scaling
│   ├── train.py         # Phase 4: PyTorch model training + evaluation
│   ├── live_feed.py      # Phase 5: real-time inference engine (live NBA games)
│   ├── simulate_live.py # Phase 5b: replay a cached historical game as a live demo
│   └── server.py         # Phase 6: Flask + WebSocket server
├── dashboard/
│   └── templates/
│       └── index.html   # Phase 7: live dashboard UI
└── requirements.txt
```

## Setup

```bash
git clone <this-repo-url>
cd nba-win-prob
pip install -r requirements.txt
```

## Running the full pipeline

```bash
# Phase 1: pull ~2,600 games of play-by-play data (~45-60 min)
python src/data_pull.py

# Phase 2: build the feature table (score_diff, time, possession, fouls)
python src/features.py

# Phase 3: split by game_id (no data leakage) + scale features
python -m src.split_data

# Phase 4: train the neural network
python -m src.train
```

## Running the live dashboard

```bash
python -m src.server
```

Then open **http://localhost:5000**, pick a game from the dropdown, and hit
**Start Simulation** — the dashboard replays that real game's play-by-play
and shows the model's live win-probability prediction as it unfolds.

## Model performance

On a held-out test set (20% of games, split by `game_id` to avoid leakage):

| Metric | Value |
|---|---|
| Accuracy | ~75.6% |
| AUC | ~0.84 |
| Brier Score | ~0.16 |

The model is well-calibrated: when it predicts a 74% win probability, the
actual observed win rate in that bucket is close to 78%.

## Key design decisions

- **Group-aware train/test split**: splitting is done by `game_id`, not by
  row, since events within a single game are highly correlated — a row-level
  split would leak information between train and test.
- **Possession inference**: the NBA doesn't publish a possession field
  directly, so it's inferred from the event sequence (made shots, rebounds,
  turnovers) via a small state machine.
- **Bonus flag over raw foul count**: whether a team is in the bonus (5+
  fouls in a period) is a much stronger signal than the raw foul count alone.
- **Calibration over raw accuracy**: since the goal is a trustworthy
  probability (not just a binary prediction), the model is evaluated with
  Brier score and a reliability curve, not just accuracy.

## Limitations

- Possession inference is a heuristic, not ground truth — the NBA does not
  publish this field.
- The live dashboard replays cached historical games rather than polling a
  real live game, since `nba_api`'s live endpoints may be inaccessible
  depending on network/region, and a real live game isn't always available
  for demo purposes.
