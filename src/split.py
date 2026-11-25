"""Step 2: split the dataset into train and test, stratified on price bins."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

IN = Path("data/processed/dataset.csv")
OUT_DIR = Path("data/split")
TARGET = "SalePrice"


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def split(df, test_size, n_bins, seed):
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    bins = pd.qcut(y, q=n_bins, labels=False)
    return train_test_split(X, y, test_size=test_size, stratify=bins, random_state=seed)


def main():
    p = load_params()["split"]
    df = pd.read_csv(IN)
    X_train, X_test, y_train, y_test = split(df, p["test_size"], p["n_bins"], p["seed"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)
    print(f"Train: {len(X_train)} rows, test: {len(X_test)} rows -> {OUT_DIR}")


if __name__ == "__main__":
    main()
