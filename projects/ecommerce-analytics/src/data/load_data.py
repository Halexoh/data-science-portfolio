import pandas as pd
from src.config import DATA_RAW_DIR


def load_orders() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "olist_orders_dataset.csv")


def load_items() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "olist_order_items_dataset.csv")


def load_customers() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "olist_customers_dataset.csv")


def load_geo() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "olist_geolocation_dataset.csv")


def load_payments() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "olist_order_payments_dataset.csv")


def load_reviews() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "olist_order_reviews_dataset.csv")


def load_products() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "olist_products_dataset.csv")


def load_sellers() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "olist_sellers_dataset.csv")


def load_category() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW_DIR / "product_category_name_translation.csv")


def load_all_data() -> dict:
    return {
        "orders": load_orders(),
        "items": load_items(),
        "customers": load_customers(),
        "geo": load_geo(),
        "payments": load_payments(),
        "reviews": load_reviews(),
        "products": load_products(),
        "sellers": load_sellers(),
        "category": load_category(),
    }