"""Step 1: clean the raw data and keep the columns most linked to the price."""
from pathlib import Path

import pandas as pd
import yaml

RAW = Path("data/raw/data.csv")
OUT = Path("data/processed/dataset.csv")
TARGET = "SalePrice"


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def clean(df):
    """Same cleaning as the notebook: numeric columns only, no Id, no incomplete rows."""
    return df.select_dtypes("number").drop(columns="Id").dropna()


def select_features(df, threshold):
    """New step: keep the columns whose correlation with the price is above the threshold."""
    corr = df.corr()[TARGET].abs()
    keep = corr[corr > threshold].index  # SalePrice stays (its correlation with itself is 1)
    return df[keep]


def main():
    threshold = load_params()["process"]["corr_threshold"]
    df = pd.read_csv(RAW)
    df = select_features(clean(df), threshold)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Kept {df.shape[1] - 1} features and {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
