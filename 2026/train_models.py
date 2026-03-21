"""
Train matchup-based models for March Madness prediction.

Takes two teams' features as input, predicts 1 if Team A wins, 0 if Team A loses.
Trains: Logistic Regression, Random Forest, XGBoost, PyTorch Neural Net.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Columns to drop (non-numeric, identity, or categorical)
DROP_COLS = [
    "YEAR", "CONF", "CONF ID", "QUAD NO", "QUAD ID",
    "TEAM NO", "TEAM ID", "TEAM", "ROUND",
    "CONF_CONF",
]

# Percentage string columns that need conversion
PCT_COLS = ["HIST_F4%", "HIST_CHAMP%"]


# ── Neural Net ──────────────────────────────────────────────────────────────────

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


def train_neural_net(X_train, y_train, X_val, y_val, input_dim, epochs=100, patience=10):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MatchupNet(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).squeeze()
            val_loss = criterion(val_logits, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model.cpu()


# ── Data Pipeline ───────────────────────────────────────────────────────────────

def build_feature_lookup(train_df):
    """Preprocess train_df and return feature_lookup, feature_cols, medians."""
    # Convert percentage strings → floats
    for col in PCT_COLS:
        if col in train_df.columns:
            train_df[col] = (
                train_df[col]
                .astype(str)
                .str.rstrip("%")
                .apply(lambda x: np.nan if x in ("", "nan") else float(x))
            )

    # Feature columns = everything except DROP_COLS
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]
    # Ensure all feature cols are numeric
    for col in feature_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")

    # Fill NaN with column median
    medians = train_df[feature_cols].median()
    train_df[feature_cols] = train_df[feature_cols].fillna(medians)

    # Build lookup: (year, team_no) → feature vector
    train_df["_key"] = list(zip(train_df["YEAR"], train_df["TEAM NO"]))
    feature_lookup = {
        key: row[feature_cols].values.astype(np.float64)
        for key, row in train_df.set_index("_key")[feature_cols].iterrows()
    }

    return feature_lookup, feature_cols, medians


def build_matchup_samples(completed, feature_lookup):
    """Convert paired matchup rows into feature vectors and labels."""
    X_list, y_list = [], []
    skipped = 0

    for i in range(0, len(completed) - 1, 2):
        row_a = completed.iloc[i]
        row_b = completed.iloc[i + 1]

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
        score_a = row_a["SCORE"]
        score_b = row_b["SCORE"]

        # label = 1 if team A wins, 0 if team B wins
        label = 1 if score_a > score_b else 0

        # Original: [A | B] with label
        X_list.append(np.concatenate([feats_a, feats_b]))
        y_list.append(label)

        # Augmented: [B | A] with flipped label
        X_list.append(np.concatenate([feats_b, feats_a]))
        y_list.append(1 - label)

    return np.array(X_list), np.array(y_list), skipped


def load_and_prepare():
    """Build train (2008-2024) and eval (2025) matchup datasets."""
    train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    matchups_df = pd.read_csv(os.path.join(DATA_DIR, "matchups.csv"))

    feature_lookup, feature_cols, medians = build_feature_lookup(train_df)

    # Filter matchups to completed games (non-empty SCORE)
    matchups_df["SCORE"] = pd.to_numeric(matchups_df["SCORE"], errors="coerce")
    completed = matchups_df.dropna(subset=["SCORE"]).reset_index(drop=True)

    # Split by year: train on 2008-2024, eval on 2025
    train_matchups = completed[completed["YEAR"] <= 2024].reset_index(drop=True)
    eval_matchups = completed[completed["YEAR"] == 2025].reset_index(drop=True)

    X_train, y_train, skip_tr = build_matchup_samples(train_matchups, feature_lookup)
    X_eval, y_eval, skip_ev = build_matchup_samples(eval_matchups, feature_lookup)

    print(f"Train set (2008-2024): {X_train.shape[0]} samples "
          f"({X_train.shape[0] // 2} games, skipped {skip_tr})")
    print(f"Eval set  (2025):      {X_eval.shape[0]} samples "
          f"({X_eval.shape[0] // 2} games, skipped {skip_ev})")
    print(f"Features per sample:   {X_train.shape[1]}")

    return X_train, y_train, X_eval, y_eval, feature_cols, medians


# ── Evaluation on 2025 holdout ──────────────────────────────────────────────────

def evaluate_models(X_train, y_train, X_eval, y_eval):
    """Train on 2008-2024, evaluate on 2025 holdout."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_ev_s = scaler.transform(X_eval)

    results = {}

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_s, y_train)
    p = lr.predict_proba(X_ev_s)
    results["LogReg"] = {
        "acc": accuracy_score(y_eval, lr.predict(X_ev_s)),
        "logloss": log_loss(y_eval, p),
    }

    # Random Forest
    rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_tr_s, y_train)
    p = rf.predict_proba(X_ev_s)
    results["RF"] = {
        "acc": accuracy_score(y_eval, rf.predict(X_ev_s)),
        "logloss": log_loss(y_eval, p),
    }

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    xgb.fit(X_tr_s, y_train)
    p = xgb.predict_proba(X_ev_s)
    results["XGB"] = {
        "acc": accuracy_score(y_eval, xgb.predict(X_ev_s)),
        "logloss": log_loss(y_eval, p),
    }

    # Neural Net (80/20 split within training set for early stopping)
    split = int(0.8 * len(X_tr_s))
    idx = np.random.RandomState(42).permutation(len(X_tr_s))
    X_shuf, y_shuf = X_tr_s[idx], y_train[idx]
    model = train_neural_net(
        X_shuf[:split], y_shuf[:split],
        X_shuf[split:], y_shuf[split:],
        input_dim=X_tr_s.shape[1],
    )
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_ev_s, dtype=torch.float32)).squeeze()
        probs = torch.sigmoid(logits).numpy()
    preds = (probs > 0.5).astype(int)
    results["NeuralNet"] = {
        "acc": accuracy_score(y_eval, preds),
        "logloss": log_loss(y_eval, probs),
    }

    return results


# ── Train Final Models & Save ───────────────────────────────────────────────────

def train_and_save(X, y, feature_cols, medians):
    """Train all models on 2008-2024 data and save to models/."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_s, y)
    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_s, y)
    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.pkl"))

    # XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    xgb.fit(X_s, y)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgboost.pkl"))

    # Neural Net (80/20 split for early stopping)
    print("Training Neural Net...")
    split = int(0.8 * len(X_s))
    idx = np.random.RandomState(42).permutation(len(X_s))
    X_shuf, y_shuf = X_s[idx], y[idx]
    model = train_neural_net(
        X_shuf[:split], y_shuf[:split],
        X_shuf[split:], y_shuf[split:],
        input_dim=X_s.shape[1],
    )
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "neural_net.pt"))

    # Save scaler, feature cols, and medians for inference
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_cols.pkl"))
    joblib.dump(medians, os.path.join(MODEL_DIR, "medians.pkl"))

    print(f"\nAll models saved to {MODEL_DIR}/")


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("March Madness Matchup Model Training")
    print("=" * 60)

    # 1. Build dataset
    print("\n[1/3] Building matchup dataset...")
    X_train, y_train, X_eval, y_eval, feature_cols, medians = load_and_prepare()

    # 2. Evaluate on 2025 holdout
    print("\n[2/3] Evaluating on 2025 holdout set...")
    results = evaluate_models(X_train, y_train, X_eval, y_eval)

    print("\n" + "=" * 60)
    print("Eval on 2025 holdout (trained on 2008-2024)")
    print(f"{'Model':<15} {'Accuracy':>10} {'Log Loss':>10}")
    print("-" * 37)
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['acc']:>10.4f} {metrics['logloss']:>10.4f}")
    print("=" * 60)

    # 3. Train final models on 2008-2024
    print("\n[3/3] Training final models on 2008-2024...")
    train_and_save(X_train, y_train, feature_cols, medians)

    print("\nDone!")


if __name__ == "__main__":
    main()
