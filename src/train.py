"""Step 3: train the model on the training set."""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

SPLIT = Path("data/split")
MODEL = Path("models/model.pkl")


def train(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def main():
    X_train = pd.read_csv(SPLIT / "X_train.csv")
    y_train = pd.read_csv(SPLIT / "y_train.csv").squeeze("columns")
    model = train(X_train, y_train)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved -> {MODEL}")


if __name__ == "__main__":
    main()
