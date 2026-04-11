import pandas as pd



def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["days_to_approve"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.days

    df["days_to_carrier"] = (
        df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["days_to_deliver"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_day"] = df["order_purchase_timestamp"].dt.day
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["purchase_weekday"] = df["order_purchase_timestamp"].dt.weekday
    df["is_weekend_purchase"] = df["purchase_weekday"].isin([5, 6]).astype(int)

    return df


def add_business_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_delivered"] = (df["order_status"] == "delivered").astype(int)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_calendar_features(df)
    df = add_business_flags(df)

    return df

def clean_outliers(df):
    q_low = df['delivery_delay_days'].quantile(0.01)
    q_high = df['delivery_delay_days'].quantile(0.99)

    df = df[
        (df['delivery_delay_days'] >= q_low) &
        (df['delivery_delay_days'] <= q_high)
    ].copy()

    return df


def impute_product_features(df):
    cols = [
        'avg_product_photos_qty',
        'avg_product_name_length',
        'avg_product_description_length'
    ]
    
    df[cols] = df[cols].fillna(0)
    
    return df