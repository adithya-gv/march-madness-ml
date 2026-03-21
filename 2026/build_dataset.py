"""
Build unified training and testing CSVs for March Madness prediction.
- train.csv: 2008-2025 (~1,147 rows), all features + ROUND target
- test.csv: 2026 (68 rows), all features, ROUND=0
"""

import pandas as pd
import numpy as np

DATA_DIR = "data/"
OUTPUT_DIR = "./"


def aggregate_ap_poll(ap_df):
    """Aggregate weekly AP poll data to per-team per-year features."""

    def compute_features(group):
        group = group.sort_values("WEEK")

        # Preseason (week 1)
        week1 = group[group["WEEK"] == group["WEEK"].min()]
        if len(week1) > 0:
            row = week1.iloc[0]
            ap_preseason_rank = int(row["AP RANK"]) if row["AP RANK"] > 0 else 0
            ap_preseason_votes = int(row["AP VOTES"])
        else:
            ap_preseason_rank = 0
            ap_preseason_votes = 0

        # Final week (max week)
        final = group[group["WEEK"] == group["WEEK"].max()]
        if len(final) > 0:
            row = final.iloc[0]
            ap_final_rank = int(row["AP RANK"]) if row["AP RANK"] > 0 else 0
            ap_final_votes = int(row["AP VOTES"])
        else:
            ap_final_rank = 0
            ap_final_votes = 0

        # Best rank (lowest positive AP RANK)
        ranked = group[group["AP RANK"] > 0]["AP RANK"]
        ap_best_rank = int(ranked.min()) if len(ranked) > 0 else 0

        # Weeks ranked (using RANK? column as indicator)
        if "RANK?" in group.columns:
            ap_weeks_ranked = int(group["RANK?"].sum())
        else:
            ap_weeks_ranked = int((group["AP RANK"] > 0).sum())

        return pd.Series(
            {
                "AP_PRESEASON_RANK": ap_preseason_rank,
                "AP_FINAL_RANK": ap_final_rank,
                "AP_BEST_RANK": ap_best_rank,
                "AP_WEEKS_RANKED": ap_weeks_ranked,
                "AP_FINAL_VOTES": ap_final_votes,
                "AP_PRESEASON_VOTES": ap_preseason_votes,
            }
        )

    return (
        ap_df.groupby(["YEAR", "TEAM NO"])
        .apply(compute_features, include_groups=False)
        .reset_index()
    )


def main():
    # --- Load all CSVs ---
    kenpom = pd.read_csv(f"{DATA_DIR}kenpom.csv")
    resumes = pd.read_csv(f"{DATA_DIR}resumes.csv")
    team_rankings = pd.read_csv(f"{DATA_DIR}team_rankings.csv")
    preseason = pd.read_csv(f"{DATA_DIR}preseason.csv")
    z_rating = pd.read_csv(f"{DATA_DIR}z_rating.csv")
    conf_stats = pd.read_csv(f"{DATA_DIR}conf_stats.csv")
    ap_poll = pd.read_csv(f"{DATA_DIR}ap_poll_data.csv")
    team_results = pd.read_csv(f"{DATA_DIR}team_results.csv")

    # --- Aggregate AP poll data ---
    ap_features = aggregate_ap_poll(ap_poll)

    # --- Prepare tables: drop duplicate identity columns ---
    identity_cols = ["TEAM", "SEED", "ROUND"]

    resumes_clean = resumes.drop(columns=identity_cols, errors="ignore")
    rankings_clean = team_rankings.drop(columns=identity_cols, errors="ignore")
    preseason_clean = preseason.drop(columns=identity_cols, errors="ignore")
    # z_rating has two rows per team (NEW and OLD types) — pivot into separate columns
    z_new = z_rating[z_rating["TYPE"] == "NEW"].drop(columns=["TYPE"] + identity_cols, errors="ignore")
    z_old = z_rating[z_rating["TYPE"] == "OLD"].drop(columns=["TYPE"] + identity_cols, errors="ignore")
    z_old = z_old.rename(columns={
        "Z RATING RANK": "Z RATING RANK OLD",
        "Z RATING": "Z RATING OLD",
    })
    # Keep SEED LIST only from NEW (same in both)
    z_old = z_old.drop(columns=["SEED LIST"], errors="ignore")
    z_rating_clean = z_new.merge(z_old, on=["YEAR", "TEAM NO"], how="left")

    # --- Prefix conf_stats columns with CONF_ ---
    conf_key_cols = {"YEAR", "CONF ID"}
    conf_stats_prefixed = conf_stats.rename(
        columns={c: f"CONF_{c}" for c in conf_stats.columns if c not in conf_key_cols}
    )

    # --- Prefix team_results columns with HIST_ ---
    team_results_prefixed = team_results.rename(
        columns={c: f"HIST_{c}" for c in team_results.columns if c != "TEAM"}
    )
    # Drop HIST_TEAM ID (internal ID, not useful after join)
    team_results_prefixed = team_results_prefixed.drop(
        columns=["HIST_TEAM ID"], errors="ignore"
    )

    # --- Perform joins ---
    df = kenpom.copy()

    # Join resumes on (YEAR, TEAM NO) — suffix _RES for any clashes
    df = df.merge(resumes_clean, on=["YEAR", "TEAM NO"], how="left", suffixes=("", "_RES"))

    # Join team_rankings on (YEAR, TEAM NO)
    df = df.merge(rankings_clean, on=["YEAR", "TEAM NO"], how="left", suffixes=("", "_TR"))

    # Join preseason on (YEAR, TEAM NO) — NaN for 2008-2011
    df = df.merge(preseason_clean, on=["YEAR", "TEAM NO"], how="left")

    # Join z_rating on (YEAR, TEAM NO) — NaN for 2008-2014
    df = df.merge(z_rating_clean, on=["YEAR", "TEAM NO"], how="left")

    # Join conf_stats on (YEAR, CONF ID)
    df = df.merge(conf_stats_prefixed, on=["YEAR", "CONF ID"], how="left")

    # Join AP features on (YEAR, TEAM NO)
    df = df.merge(ap_features, on=["YEAR", "TEAM NO"], how="left")

    # Join team_results on TEAM name
    df = df.merge(team_results_prefixed, on="TEAM", how="left")

    # --- Fill AP features with 0 for teams not in AP poll data ---
    ap_cols = [c for c in df.columns if c.startswith("AP_")]
    df[ap_cols] = df[ap_cols].fillna(0).astype(int)

    # --- Encode BID TYPE as binary (1=At-Large, 0=Auto) ---
    if "BID TYPE" in df.columns:
        df["BID TYPE"] = (
            df["BID TYPE"].map({"At-Large": 1, "Auto": 0}).fillna(0).astype(int)
        )

    # --- Check for duplicate column names ---
    dupes = [c for c in df.columns if df.columns.tolist().count(c) > 1]
    if dupes:
        print(f"WARNING: Duplicate columns found: {set(dupes)}")

    # --- Split train/test ---
    train = df[df["YEAR"] <= 2025].copy()
    test = df[df["YEAR"] == 2026].copy()

    # --- Write output ---
    train.to_csv(f"{OUTPUT_DIR}train.csv", index=False)
    test.to_csv(f"{OUTPUT_DIR}test.csv", index=False)

    # --- Summary ---
    print(f"Train: {len(train)} rows, {len(train.columns)} columns")
    print(f"Test:  {len(test)} rows, {len(test.columns)} columns")
    print(f"Years in train: {sorted(train['YEAR'].unique())}")
    print(f"Columns: {list(df.columns)}")

    # Spot checks
    assert len(test) == 68, f"Expected 68 test rows, got {len(test)}"
    assert (test["ROUND"] == 0).all(), "Test set should have ROUND=0 for all rows"
    assert not dupes, f"Duplicate columns: {set(dupes)}"

    uconn = train[(train["YEAR"] == 2024) & (train["TEAM"].str.contains("Connecticut"))]
    if len(uconn) > 0:
        r = uconn["ROUND"].values[0]
        print(f"\nSpot check — 2024 Connecticut ROUND: {r} (expected: 1, champion)")
        assert r == 1, f"Expected ROUND=1 for 2024 UConn, got {r}"

    print("\nDone!")


if __name__ == "__main__":
    main()
