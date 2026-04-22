"""
train_model.py
==============
End-to-end training pipeline for the Brazilian E-commerce review score predictor.

Usage:
    python -m src.models.train_model

Outputs (saved to outputs/):
    - model.pkl               : Trained RandomForest model (joblib)
    - metrics.json            : MAE, RMSE, R2 on held-out test set
    - feature_importance.png  : Horizontal bar chart of top features
    - prediction_error.png    : Actual vs predicted scatter plot
    - cv_scores.png           : Cross-validation score distribution
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from src.data.load_data import load_all_data
from src.data.merge_data import build_base_table
from src.data.preprocess import preprocess_all_data
from src.data.transform import build_order_level_dataset
from src.features.build_features import build_features, clean_outliers, impute_product_features

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("outputs")
FEATURES = [
    "price", "freight_value", "num_items",
    "avg_product_weight_g", "avg_product_length_cm", "avg_product_height_cm",
    "avg_product_width_cm", "avg_product_photos_qty", "avg_product_name_length",
    "avg_product_description_length", "days_to_approve", "days_to_carrier",
    "days_to_deliver", "delivery_delay_days", "purchase_month",
    "purchase_weekday", "is_weekend_purchase", "is_delivered",
]


def build_dataset() -> pd.DataFrame:
    """Load, clean, merge and engineer features.

    Pipeline: load -> preprocess -> merge -> transform -> feature engineering

    Returns:
        pd.DataFrame: Order-level dataset, NaN review_score rows dropped.
    """
    data = load_all_data()
    data = preprocess_all_data(data)
    df_base = build_base_table(data)
    df_model = build_order_level_dataset(df_base)
    df_model = build_features(df_model)
    df_model = impute_product_features(df_model)
    df_model = clean_outliers(df_model)
    return df_model.dropna(subset=["review_score"]).copy()


def prepare_features(df: pd.DataFrame) -> tuple:
    """Select feature matrix X and target y from dataset.

    Args:
        df: Order-level DataFrame from build_dataset().

    Returns:
        Tuple (X, y).
    """
    df = df.dropna(subset=["review_score"] + FEATURES).copy()
    return df[FEATURES], df["review_score"]


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """Split data into train/test sets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Test fraction (default 0.2).

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    return train_test_split(X, y, test_size=test_size, random_state=42)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    """Fit a RandomForestRegressor (200 trees, max_depth=12).

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Fitted RandomForestRegressor.
    """
    model = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def cross_validate(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> np.ndarray:
    """Run k-fold CV and return per-fold MAE scores.

    Args:
        model: Sklearn estimator.
        X: Feature matrix.
        y: Target vector.
        cv: Number of folds (default 5).

    Returns:
        Array of MAE scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    return -scores


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute MAE, RMSE, R2 on test set.

    Args:
        model: Fitted estimator.
        X_test: Test features.
        y_test: Test target.

    Returns:
        dict with keys mae, rmse, r2.
    """
    preds = model.predict(X_test)
    metrics = {
        "mae":  round(float(mean_absolute_error(y_test, preds)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "r2":   round(float(r2_score(y_test, preds)), 4),
    }
    print(f"  MAE={metrics['mae']}  RMSE={metrics['rmse']}  R2={metrics['r2']}")
    return metrics


def save_metrics(metrics: dict, cv_scores: np.ndarray) -> None:
    """Save metrics dict to outputs/metrics.json.

    Args:
        metrics: Test-set metrics dict.
        cv_scores: Per-fold CV MAE array.
    """
    payload = {**metrics, "cv_mae_mean": round(float(cv_scores.mean()), 4),
               "cv_mae_std": round(float(cv_scores.std()), 4)}
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(payload, f, indent=2)
    print("  metrics.json saved")


def plot_feature_importance(model, X: pd.DataFrame) -> None:
    """Save horizontal bar chart of top-18 feature importances.

    Args:
        model: Fitted RandomForestRegressor.
        X: Feature matrix (provides column names).
    """
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values().tail(18)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3" if v < importance.max() * 0.6 else "#1565C0" for v in importance]
    importance.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
    ax.set_title("Feature Importance - Review Score Predictor", fontsize=13, pad=12)
    ax.set_xlabel("Importance", fontsize=11)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  feature_importance.png saved")


def plot_prediction_error(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save actual vs predicted scatter plot.

    Args:
        model: Fitted estimator.
        X_test: Test features.
        y_test: Test target.
    """
    preds = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.15, s=8, color="#2196F3", rasterized=True)
    ax.plot([1, 5], [1, 5], "r--", lw=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual review score", fontsize=11)
    ax.set_ylabel("Predicted review score", fontsize=11)
    ax.set_title("Actual vs Predicted - Review Score", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "prediction_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  prediction_error.png saved")


def plot_cv_scores(cv_scores: np.ndarray) -> None:
    """Save bar chart of per-fold CV MAE scores.

    Args:
        cv_scores: Array of MAE per fold.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = [f"Fold {i+1}" for i in range(len(cv_scores))]
    bars = ax.bar(folds, cv_scores, color="#2196F3", edgecolor="none", width=0.5)
    ax.axhline(cv_scores.mean(), color="red", linestyle="--", lw=1.2,
               label=f"Mean MAE = {cv_scores.mean():.4f}")
    ax.set_ylabel("MAE", fontsize=11)
    ax.set_title("Cross-Validation MAE (5-Fold)", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, cv_scores.max() * 1.25)
    for bar, val in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "cv_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  cv_scores.png saved")


def main() -> None:
    """Run the full training pipeline end-to-end.

    Steps:
        1. Build dataset
        2. Prepare features and train/test split
        3. Train RandomForestRegressor
        4. Cross-validate (5-fold)
        5. Evaluate on test set
        6. Save model, metrics, and plots to outputs/
    """
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Building dataset...")
    df = build_dataset()
    print(f"  {len(df):,} rows | {df.shape[1]} columns")

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Cross-validation (5-fold)...")
    cv_scores = cross_validate(model, X_train, y_train)
    print(f"  CV MAE: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print("Evaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)

    print("Saving outputs...")
    joblib.dump(model, OUTPUT_DIR / "model.pkl")
    print("  model.pkl saved")
    save_metrics(metrics, cv_scores)
    plot_feature_importance(model, X)
    plot_prediction_error(model, X_test, y_test)
    plot_cv_scores(cv_scores)

    print("Done. All outputs in outputs/")


if __name__ == "__main__":
    main()
