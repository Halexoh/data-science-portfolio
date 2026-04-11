import pandas as pd


def preprocess_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def preprocess_items(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "shipping_limit_date" in df.columns:
        df["shipping_limit_date"] = pd.to_datetime(df["shipping_limit_date"], errors="coerce")

    return df


def preprocess_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df


def preprocess_geo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates(subset=["geolocation_zip_code_prefix"])

    return df


def preprocess_payments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.groupby("order_id", as_index=False).agg({
        "payment_value": "sum"
    })

    return df
def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    date_cols = [
        "review_creation_date",
        "review_answer_timestamp",
    ]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def preprocess_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df


def preprocess_sellers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df


def preprocess_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df


def preprocess_all_data(data: dict) -> dict:
    return {
        "orders": preprocess_orders(data["orders"]),
        "items": preprocess_items(data["items"]),
        "customers": preprocess_customers(data["customers"]),
        "geo": preprocess_geo(data["geo"]),
        "payments": preprocess_payments(data["payments"]),
        "reviews": preprocess_reviews(data["reviews"]),
        "products": preprocess_products(data["products"]),
        "sellers": preprocess_sellers(data["sellers"]),
        "category": preprocess_category(data["category"]),
    }


def preprocess_payments(df):
    df = df.copy()

    df = df.groupby("order_id").agg({
        "payment_value": "sum"
    }).reset_index()

    return df