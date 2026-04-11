import pandas as pd


def merge_orders_items(orders: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    return orders.merge(items, on="order_id", how="left")


def merge_with_payments(df: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    return df.merge(payments, on="order_id", how="left")


def merge_with_reviews(df: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    return df.merge(reviews, on="order_id", how="left")


def merge_with_products(df: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    return df.merge(products, on="product_id", how="left")


def merge_with_sellers(df: pd.DataFrame, sellers: pd.DataFrame) -> pd.DataFrame:
    return df.merge(sellers, on="seller_id", how="left")


def merge_with_customers(df: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    return df.merge(customers, on="customer_id", how="left")


def merge_with_category(df: pd.DataFrame, category: pd.DataFrame) -> pd.DataFrame:
    return df.merge(category, on="product_category_name", how="left")


def build_base_table(data: dict) -> pd.DataFrame:
    df = merge_orders_items(data["orders"], data["items"])
    df = merge_with_payments(df, data["payments"])
    df = merge_with_reviews(df, data["reviews"])
    df = merge_with_products(df, data["products"])
    df = merge_with_sellers(df, data["sellers"])
    df = merge_with_customers(df, data["customers"])
    df = merge_with_category(df, data["category"])
    
    return df