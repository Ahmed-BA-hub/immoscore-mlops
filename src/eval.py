"""Step 4: measure the error on train and on test, and save it."""
import json
import math
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

SPLIT = Path("data/split")
MODEL = Path("models/model.pkl")
METRICS = Path("reports/metrics.json")


def evaluate(model, X, y):
    pred = model.predict(X)
    rmse = math.sqrt(mean_squared_error(y, pred))
    return {"rmse": round(float(rmse), 2), "r2": round(float(r2_score(y, pred)), 4)}


def main():
    with open(MODEL, "rb") as f:
        model = pickle.load(f)
    metrics = {}
    for part in ("train", "test"):
        X = pd.read_csv(SPLIT / f"X_{part}.csv")
        y = pd.read_csv(SPLIT / f"y_{part}.csv").squeeze("columns")
        metrics[part] = evaluate(model, X, y)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
