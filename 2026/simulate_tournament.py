"""
Simulate the 2026 March Madness tournament using trained matchup models.

Parses the bracket from matchups.csv, handles play-in games, then advances
winners round-by-round through the full bracket for each model.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

DROP_COLS = [
    "YEAR", "CONF", "CONF ID", "QUAD NO", "QUAD ID",
    "TEAM NO", "TEAM ID", "TEAM", "ROUND",
    "CONF_CONF",
]
PCT_COLS = ["HIST_F4%", "HIST_CHAMP%"]

ROUND_NAMES = [
    "First Four", "Round of 64", "Round of 32", "Sweet 16",
    "Elite 8", "Final Four", "Championship",
]


# ── Neural Net (must match training architecture) ───────────────────────────────

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


# ── Prediction ──────────────────────────────────────────────────────────────────

def predict_winner(model, model_name, team_a, team_b, feature_lookup, scaler, team_names):
    """Predict winner of a matchup. Returns (winner_team_no, prob_a_wins)."""
    feats_a = feature_lookup[team_a]
    feats_b = feature_lookup[team_b]
    x = np.concatenate([feats_a, feats_b]).reshape(1, -1)
    x_scaled = scaler.transform(x)

    if model_name in ("NeuralNet", "NeuralNetEntropy"):
        with torch.no_grad():
            logits = model(torch.tensor(x_scaled, dtype=torch.float32)).squeeze()
            prob_a = torch.sigmoid(logits).item()
    else:
        prob_a = model.predict_proba(x_scaled)[0, 1]

    winner = team_a if prob_a > 0.5 else team_b
    return winner, prob_a


# ── Bracket Parsing ─────────────────────────────────────────────────────────────

def parse_bracket(matchups_file):
    """Parse 2026 R64 rows into bracket, detecting play-in games.

    Returns:
        playin_games: list of (team_1_no, team_2_no) play-in matchups
        r64_games: list of 32 (team_a_no, team_b_no) R64 matchups
                   Play-in slots have team_b_no = None (filled after play-ins).
        playin_slots: dict mapping r64_game_index → (opponent_no, playin_game_index)
        team_names: dict of team_no → team_name
    """
    m = pd.read_csv(matchups_file)
    current_round = m["CURRENT ROUND"].min()
    r64 = m[(m["YEAR"] == 2026) & (m["CURRENT ROUND"] == current_round)].reset_index(drop=True)

    team_names = {}
    for _, row in r64.iterrows():
        team_names[int(row["TEAM NO"])] = row["TEAM"]

    playin_games = []
    r64_games = []
    playin_slots = {}

    i = 0
    while i < len(r64) - 1:
        team_a = int(r64.iloc[i]["TEAM NO"])
        team_b = int(r64.iloc[i + 1]["TEAM NO"])

        # Check if next pair shares the same team_a (play-in)
        if (i + 3 < len(r64)
                and int(r64.iloc[i + 2]["TEAM NO"]) == team_a):
            playin_team_2 = int(r64.iloc[i + 3]["TEAM NO"])
            pi_idx = len(playin_games)
            playin_games.append((team_b, playin_team_2))
            slot_idx = len(r64_games)
            r64_games.append((team_a, None))  # placeholder
            playin_slots[slot_idx] = pi_idx
            i += 4
        else:
            r64_games.append((team_a, team_b))
            i += 2

    return playin_games, r64_games, playin_slots, team_names


# ── Tournament Simulation ──────────────────────────────────────────────────────

def simulate_tournament(model, model_name, playin_games, r64_games, playin_slots,
                        feature_lookup, scaler, team_names):
    """Simulate full tournament for one model. Returns round-by-round results."""
    results = {}

    # 1. Play-in games
    playin_winners = []
    playin_results = []
    for team_1, team_2 in playin_games:
        winner, prob = predict_winner(
            model, model_name, team_1, team_2, feature_lookup, scaler, team_names
        )
        playin_winners.append(winner)
        playin_results.append((team_names[team_1], team_names[team_2], team_names[winner]))
    results["First Four"] = playin_results

    # 2. Fill play-in winners into R64 bracket
    bracket = list(r64_games)
    for slot_idx, pi_idx in playin_slots.items():
        opponent = bracket[slot_idx][0]
        bracket[slot_idx] = (opponent, playin_winners[pi_idx])

    # 3. Simulate rounds
    current_matchups = bracket
    for round_name in ROUND_NAMES[1:]:  # R64 through Championship
        round_results = []
        winners = []
        for team_a, team_b in current_matchups:
            winner, prob = predict_winner(
                model, model_name, team_a, team_b, feature_lookup, scaler, team_names
            )
            winners.append(winner)
            round_results.append((team_names[team_a], team_names[team_b], team_names[winner]))
        results[round_name] = round_results

        # Pair consecutive winners for next round
        if len(winners) >= 2:
            current_matchups = [
                (winners[i], winners[i + 1]) for i in range(0, len(winners), 2)
            ]

    return results


# ── Display ─────────────────────────────────────────────────────────────────────

def print_results(model_name, results):
    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    for round_name in ROUND_NAMES:
        games = results.get(round_name, [])
        if not games:
            continue
        print(f"\n  {round_name}")
        print(f"  {'-' * 40}")
        for team_a, team_b, winner in games:
            marker_a = " >>>>" if winner == team_a else ""
            marker_b = " >>>>" if winner == team_b else ""
            print(f"    {team_a:>20s}{marker_a}  vs  {team_b:<20s}{marker_b}")

    # Print champion
    champ = results["Championship"][0][2]
    print(f"\n  CHAMPION: {champ}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bracket", default=os.path.join(DATA_DIR, "matchups.csv"),
                        help="Path to bracket matchups CSV (default: data/matchups.csv)")
    parser.add_argument("--test", default=os.path.join(BASE_DIR, "test.csv"),
                        help="Path to team stats CSV (default: test.csv)")
    args = parser.parse_args()

    # Load saved artifacts
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    medians = joblib.load(os.path.join(MODEL_DIR, "medians.pkl"))

    # Load test data and build feature lookup
    test_df = pd.read_csv(args.test)
    for col in PCT_COLS:
        if col in test_df.columns:
            test_df[col] = (
                test_df[col]
                .astype(str)
                .str.rstrip("%")
                .apply(lambda x: np.nan if x in ("", "nan") else float(x))
            )
    for col in feature_cols:
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
    test_df[feature_cols] = test_df[feature_cols].fillna(medians)

    feature_lookup = {}
    name_lookup = {}
    for _, row in test_df.iterrows():
        tno = int(row["TEAM NO"])
        feature_lookup[tno] = row[feature_cols].values.astype(np.float64)
        name_lookup[tno] = row["TEAM"]

    # Parse bracket
    playin_games, r64_games, playin_slots, team_names = parse_bracket(args.bracket)
    # Merge name lookups
    team_names.update(name_lookup)

    print("2026 March Madness Tournament Simulation")
    print(f"Bracket: {len(r64_games)} R64 slots, {len(playin_games)} play-in games")

    # Load and run each model
    models = {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
        "XGBoost": joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl")),
    }

    # Neural net
    n_features = len(feature_cols) * 2
    nn_model = MatchupNet(n_features)
    nn_model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "neural_net.pt"), map_location="cpu", weights_only=True,
    ))
    nn_model.eval()
    models["NeuralNet"] = nn_model

    # Entropy-regularized neural net (same architecture)
    nn_entropy_model = MatchupNet(n_features)
    nn_entropy_model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "neural_net_entropy.pt"), map_location="cpu", weights_only=True,
    ))
    nn_entropy_model.eval()
    models["NeuralNetEntropy"] = nn_entropy_model

    for model_name, model in models.items():
        results = simulate_tournament(
            model, model_name, playin_games, r64_games, playin_slots,
            feature_lookup, scaler, team_names,
        )
        print_results(model_name, results)


if __name__ == "__main__":
    main()
