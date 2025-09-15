# src/data/splits.py
from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "data" / "labels.csv"
SPLIT_DIR = REPO / "data" / "splits"

def main():
    if not LABELS.exists():
        print("[ERROR] data/labels.csv not found. Run build_labels_csv first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(LABELS)
    # Require binary target 0/1
    df = df[df["target"].isin([0, 1])].reset_index(drop=True)
    if df["target"].nunique() < 2:
        raise SystemExit("[ERROR] Not enough classes for stratification (need both 0 and 1).")

    # 70 / 15 / 15 split, stratified by target
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(df, df["target"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(sss2.split(temp_df, temp_df["target"]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

    print(f"[OK] Wrote splits to {SPLIT_DIR.relative_to(REPO)}")
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = d["target"].value_counts(normalize=False).to_dict()
        print(f"  {name:5s}: n={len(d)}  class0={counts.get(0,0)}  class1={counts.get(1,0)}")

if __name__ == "__main__":
    main()
