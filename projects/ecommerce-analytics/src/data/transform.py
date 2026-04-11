import pandas as pd


def build_order_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_order = (
        df.groupby("order_id")
        .agg(
            {
                "customer_id": "first",
                "order_status": "first",
                "order_purchase_timestamp": "first",
                "order_approved_at": "first",
                "order_delivered_carrier_date": "first",
                "order_delivered_customer_date": "first",
                "order_estimated_delivery_date": "first",
                "price": "sum",
                "freight_value": "sum",
                "payment_value": "sum",
                "order_item_id": "max",
                "review_score": "first",
                "product_weight_g": "mean",
                "product_length_cm": "mean",
                "product_height_cm": "mean",
                "product_width_cm": "mean",
                "product_photos_qty": "mean",
                "product_name_lenght": "mean",
                "product_description_lenght": "mean",
            }
        )
        .reset_index()
    )

    df_order = df_order.rename(
        columns={
            "order_item_id": "num_items",
            "product_weight_g": "avg_product_weight_g",
            "product_length_cm": "avg_product_length_cm",
            "product_height_cm": "avg_product_height_cm",
            "product_width_cm": "avg_product_width_cm",
            "product_photos_qty": "avg_product_photos_qty",
            "product_name_lenght": "avg_product_name_length",
            "product_description_lenght": "avg_product_description_length",
        }
    )

    return df_order
    




