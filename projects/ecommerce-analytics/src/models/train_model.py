#imports 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# pipeline propio
from src.data.load_data import load_all_data
from src.data.preprocess import preprocess_all_data
from src.data.merge_data import build_base_table
from src.data.transform import build_order_level_dataset
from src.features.build_features import (
    build_features,
    impute_product_features,
    clean_outliers
)


def build_dataset():
    data = load_all_data()
    data = preprocess_all_data(data)

    df_base = build_base_table(data)
    df_model = build_order_level_dataset(df_base)

    df_model = build_features(df_model)
    df_model = impute_product_features(df_model)
    df_model = clean_outliers(df_model)

    df_model = df_model.dropna(subset=["review_score"]).copy()

    return df_model

def prepare_features(df):
    features = [
        "price",
        "freight_value",
        "num_items",
        "avg_product_weight_g",
        "avg_product_length_cm",
        "avg_product_height_cm",
        "avg_product_width_cm",
        "avg_product_photos_qty",
        "avg_product_name_length",
        "avg_product_description_length",
        "days_to_approve",
        "days_to_carrier",
        "days_to_deliver",
        "delivery_delay_days",
        "purchase_month",
        "purchase_weekday",
        "is_weekend_purchase",
        "is_delivered",
    ]

    df = df.dropna(subset=["review_score"] + features).copy()

    X = df[features]
    y = df["review_score"]

    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")


def feature_importance(model, X):
    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop Features:")
    print(importance.head(10))

    return importance

def main():
    df = build_dataset()

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    feature_importance(model, X)


if __name__ == "__main__":
    main()