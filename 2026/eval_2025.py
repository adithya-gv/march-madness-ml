"""
Evaluate saved models on the 2025 holdout set.
Loads PKL/PT files from models/ and runs them on 2025 matchups.
"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, log_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DROP_COLS = [
    "YEAR", "CONF", "CONF ID", "QUAD NO", "QUAD ID",
    "TEAM NO", "TEAM ID", "TEAM", "ROUND",
    "CONF_CONF",
]
PCT_COLS = ["HIST_F4%", "HIST_CHAMP%"]

# Loser's ROUND value = which round the game was played in
# Higher number = earlier round (64 = R1, 32 = R2, ...)
ROUND_POINTS = {64: 1, 32: 2, 16: 4, 8: 8, 4: 16, 2: 32}


class MatchupNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


def build_eval_set():
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    matchups_df = pd.read_csv(os.path.join(DATA_DIR, "matchups.csv"))

    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    medians = joblib.load(os.path.join(MODEL_DIR, "medians.pkl"))

    # Preprocess train_df the same way as training
    for col in PCT_COLS:
        if col in train_df.columns:
            train_df[col] = (
                train_df[col]
                .astype(str)
                .str.rstrip("%")
                .apply(lambda x: np.nan if x in ("", "nan") else float(x))
            )
    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)

    # Build feature lookup
    train_df["_key"] = list(zip(train_df["YEAR"], train_df["TEAM NO"]))
    feature_lookup = {
        key: row[feature_cols].values.astype(np.float64)
        for key, row in train_df.set_index("_key")[feature_cols].iterrows()
    }

    # Filter to 2025 completed matchups
    matchups_df["SCORE"] = pd.to_numeric(matchups_df["SCORE"], errors="coerce")
    eval_matchups = (
        matchups_df[matchups_df["YEAR"] == 2025]
        .dropna(subset=["SCORE"])
        .reset_index(drop=True)
    )

    X_list, y_list, game_rounds = [], [], []
    skipped = 0
    for i in range(0, len(eval_matchups) - 1, 2):
        row_a = eval_matchups.iloc[i]
        row_b = eval_matchups.iloc[i + 1]
        if row_a["YEAR"] != row_b["YEAR"]:
            skipped += 1
            continue
        key_a = (int(row_a["YEAR"]), int(row_a["TEAM NO"]))
        key_b = (int(row_b["YEAR"]), int(row_b["TEAM NO"]))
        if key_a not in feature_lookup or key_b not in feature_lookup:
            skipped += 1
            continue
        feats_a = feature_lookup[key_a]
        feats_b = feature_lookup[key_b]
        label = 1 if row_a["SCORE"] > row_b["SCORE"] else 0
        # The game's round = the loser's ROUND (higher number = earlier round)
        game_round = max(int(row_a["ROUND"]), int(row_b["ROUND"]))

        # Original sample
        X_list.append(np.concatenate([feats_a, feats_b]))
        y_list.append(label)
        game_rounds.append(game_round)

        # Augmented (flipped) sample — same game, same round
        X_list.append(np.concatenate([feats_b, feats_a]))
        y_list.append(1 - label)
        game_rounds.append(game_round)

    X = np.array(X_list)
    y = np.array(y_list)
    rounds = np.array(game_rounds)
    n_games = X.shape[0] // 2
    print(f"2025 eval set: {n_games} games, skipped {skipped}")
    return X, y, rounds


def bracket_score(preds, y_true, rounds):
    """Sum March Madness points for correct predictions on original (non-augmented) samples."""
    # Only score original samples (every other, starting at 0)
    orig_idx = np.arange(0, len(preds), 2)
    correct = preds[orig_idx] == y_true[orig_idx]
    pts = np.array([ROUND_POINTS.get(r, 0) for r in rounds[orig_idx]])
    return int((correct * pts).sum()), int(pts.sum()), int(correct.sum()), len(orig_idx)


def main():
    print("=" * 60)
    print("2025 Holdout Evaluation (using saved models)")
    print("=" * 60)

    X_eval, y_eval, rounds = build_eval_set()
    n_games = len(X_eval) // 2
    max_pts = int(np.array([ROUND_POINTS.get(r, 0) for r in rounds[::2]]).sum())

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    X_ev_s = scaler.transform(X_eval)

    results = {}

    # Logistic Regression
    lr = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
    p = lr.predict_proba(X_ev_s)
    preds = lr.predict(X_ev_s)
    pts, _, correct, _ = bracket_score(preds, y_eval, rounds)
    results["LogReg"] = {
        "acc": accuracy_score(y_eval, preds),
        "logloss": log_loss(y_eval, p),
        "correct": correct,
        "pts": pts,
    }

    # Random Forest
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
    p = rf.predict_proba(X_ev_s)
    preds = rf.predict(X_ev_s)
    pts, _, correct, _ = bracket_score(preds, y_eval, rounds)
    results["RandomForest"] = {
        "acc": accuracy_score(y_eval, preds),
        "logloss": log_loss(y_eval, p),
        "correct": correct,
        "pts": pts,
    }

    # XGBoost
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))
    p = xgb.predict_proba(X_ev_s)
    preds = xgb.predict(X_ev_s)
    pts, _, correct, _ = bracket_score(preds, y_eval, rounds)
    results["XGBoost"] = {
        "acc": accuracy_score(y_eval, preds),
        "logloss": log_loss(y_eval, p),
        "correct": correct,
        "pts": pts,
    }

    # Neural Net
    input_dim = X_ev_s.shape[1]
    model = MatchupNet(input_dim)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "neural_net.pt"), map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_ev_s, dtype=torch.float32)).squeeze()
        probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    pts, _, correct, _ = bracket_score(preds, y_eval, rounds)
    results["NeuralNet"] = {
        "acc": accuracy_score(y_eval, preds),
        "logloss": log_loss(y_eval, probs),
        "correct": correct,
        "pts": pts,
    }

    print(f"\nMax possible: {n_games} games, {max_pts} pts\n")
    print(f"{'Model':<15} {'Accuracy':>10} {'Log Loss':>10} {'Correct':>10} {'Pts':>8}")
    print("-" * 57)
    for name, m in results.items():
        print(f"{name:<15} {m['acc']:>10.4f} {m['logloss']:>10.4f}"
              f" {m['correct']:>7}/{n_games}  {m['pts']:>4}/{max_pts}")
    print("=" * 57)


if __name__ == "__main__":
    main()
