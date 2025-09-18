import pathlib as p
import pandas as pd

for split in ["train", "val", "test"]:
    df = pd.read_csv(p.Path("data/splits") / f"{split}.csv")
    print(f"\n[{split}] total={len(df)}")
    print(df["label"].value_counts(dropna=False))
