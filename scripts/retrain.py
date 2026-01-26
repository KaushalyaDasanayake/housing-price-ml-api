import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_ORDER = [
    "MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup",
    "Latitude","Longitude","RoomsPerHousehold","BedroomsPerHouse","PopulationPerHousehold"
]

PRED_LOG_PATH = Path("data/predictions.csv")   # production log
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct order + float types."""
    return df[FEATURE_ORDER].astype(float)


def save_training_stats(X: pd.DataFrame):
    stats = {}
    for col in FEATURE_ORDER:
        stats[col] = {
            "mean": float(X[col].mean()),
            "std": float(X[col].std(ddof=0))
        }

    with open(MODEL_DIR / "training_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def main():
    # 1) Load original training data (same source every time)
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    # y is the target (median house value)
    y = df["MedHouseVal"]
    X = df.drop(columns=["MedHouseVal"])

    # Create derived features if used them in API
    # If API expects these 3 extra features, make sure they exist here too
    X["RoomsPerHousehold"] = X["AveRooms"] / X["AveOccup"]
    X["BedroomsPerHouse"] = X["AveBedrms"] / X["AveRooms"]
    X["PopulationPerHousehold"] = X["Population"] / X["AveOccup"]

    X = build_features(X)

    # 2) Load production samples (if available)
    X_all = X
    y_all = y

    if PRED_LOG_PATH.exists() and PRED_LOG_PATH.stat().st_size > 0:
        try:
            prod = pd.read_csv(PRED_LOG_PATH)

            prod_X = build_features(prod)
            X_all = pd.concat([X, prod_X], ignore_index=True)

            print(f"✅ Added {len(prod_X)} production samples")

        except Exception as e:
            print(f"⚠️ Skipping production data due to error: {e}")
        
        else:
            print("ℹ️ No production data found, training on original dataset only")

    # 3) Fit scaler + model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    model = LinearRegression()
    model.fit(X_scaled, y_all)

    # 4) Save artifacts
    joblib.dump(model, MODEL_DIR / "model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")

    # 5) Save drift baseline stats
    save_training_stats(X_all)

    print("✅ Retraining complete")
    print("✅ Saved: model.joblib, scaler.joblib, training_stats.json")


if __name__ == "__main__":
    main()