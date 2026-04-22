# Brazilian E-commerce Analytics

> **Predicting customer satisfaction from logistics and product data** — an end-to-end data science project using the Olist public dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Business problem

An e-commerce company wants to understand **what drives customer satisfaction** and **why some orders receive low review scores**. Instead of treating reviews as random noise, this project treats them as measurable outcomes of operational decisions: delivery speed, product quality, and logistics efficiency.

**Key question:** Can we predict a customer's review score from the moment their order is placed?

---

## Key insight

> **Delivery delays are the single strongest predictor of low review scores.** An order delivered even one day late is significantly more likely to receive a 1-star review — regardless of product price, category, or payment method.

This finding has direct business implications: improving last-mile logistics has a larger impact on customer satisfaction than product-level improvements.

---

## Dataset

- **Source:** [Brazilian E-commerce Public Dataset (Olist)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
- **Size:** ~100,000 orders across 9 relational CSV files
- **Period:** 2016–2018
- **Note:** Raw data is not included due to size. See [How to Run](#how-to-run).

| File | Description |
|------|-------------|
| `olist_orders_dataset.csv` | Order status and timestamps |
| `olist_order_items_dataset.csv` | Items, prices, freight |
| `olist_order_reviews_dataset.csv` | Customer review scores (1–5) |
| `olist_customers_dataset.csv` | Customer location |
| `olist_order_payments_dataset.csv` | Payment method and value |
| `olist_products_dataset.csv` | Product dimensions and category |
| `olist_sellers_dataset.csv` | Seller location |
| `olist_geolocation_dataset.csv` | ZIP code coordinates |
| `product_category_name_translation.csv` | Category names in English |

---

## Project structure

```
ecommerce-analytics/
├── notebooks/
│   ├── 01_data_understanding.ipynb   # EDA: distributions, missing values, patterns
│   └── cleaning.ipynb                # Data cleaning and transformation logic
├── src/
│   ├── config.py                     # Centralized path configuration
│   ├── data/
│   │   ├── load_data.py              # Load each CSV with typed loaders
│   │   ├── merge_data.py             # Join all datasets into a base table
│   │   ├── preprocess.py             # Type casting, date parsing, nulls
│   │   └── transform.py             # Order-level aggregation
│   ├── features/
│   │   └── build_features.py        # Feature engineering, imputation, outliers
│   └── models/
│       └── train_model.py           # Full training pipeline with CV and outputs
├── outputs/                          # Generated after running train_model.py
│   ├── model.pkl
│   ├── metrics.json
│   ├── feature_importance.png
│   ├── prediction_error.png
│   └── cv_scores.png
├── requirements.txt
└── README.md
```

---

## Methodology

### Data pipeline

The project uses a **fully modular pipeline** where each stage is a separate Python module with typed functions and docstrings:

```
load_all_data()
    → preprocess_all_data()       # parse dates, cast types, handle nulls
    → build_base_table()          # merge 9 datasets on order_id
    → build_order_level_dataset() # aggregate items to order level
    → build_features()            # engineer temporal and product features
    → clean_outliers()            # IQR-based outlier removal
```

### Feature engineering

The 18 features used for modelling fall into four groups:

| Group | Features |
|-------|----------|
| **Logistics timing** | `days_to_approve`, `days_to_carrier`, `days_to_deliver`, `delivery_delay_days` |
| **Product attributes** | `avg_product_weight_g`, dimensions, `avg_product_photos_qty`, description length |
| **Order composition** | `price`, `freight_value`, `num_items` |
| **Temporal** | `purchase_month`, `purchase_weekday`, `is_weekend_purchase`, `is_delivered` |

### Model

A **RandomForestRegressor** trained to predict `review_score` (1–5):

| Hyperparameter | Value |
|---------------|-------|
| `n_estimators` | 200 |
| `max_depth` | 12 |
| `min_samples_leaf` | 5 |
| `random_state` | 42 |

Cross-validation: **5-fold** on the training set to ensure generalization.

---

## Results

| Metric | Value |
|--------|-------|
| MAE (test set) | ~0.72 |
| RMSE (test set) | ~0.98 |
| CV MAE (5-fold mean) | ~0.73 ± 0.01 |

The low variance across folds confirms the model generalizes well without overfitting.

### Top features by importance

The model consistently ranks logistics timing features at the top, confirming that **delivery performance is the primary driver of customer satisfaction**.

```
delivery_delay_days            ████████████████████  0.21
days_to_deliver                ███████████████       0.18
days_to_carrier                ████████              0.09
price                          ███████               0.08
freight_value                  ██████                0.07
avg_product_description_length ████                  0.05
days_to_approve                ████                  0.05
```

---

## How to run

**1. Clone the repository**
```bash
git clone https://github.com/Halexoh/data-science-portfolio.git
cd data-science-portfolio/projects/ecommerce-analytics
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place all CSV files inside `data_raw/`.

**4. Run the training pipeline**
```bash
python -m src.models.train_model
```

This will:
- Load and merge all 9 datasets
- Engineer 18 features from logistics, product, and temporal data
- Train a RandomForestRegressor with 5-fold cross-validation
- Save `model.pkl`, `metrics.json`, and 3 diagnostic plots to `outputs/`

**5. Explore the notebooks** (optional)

Open `notebooks/01_data_understanding.ipynb` for the full EDA, or `notebooks/cleaning.ipynb` to follow the data transformation decisions.

---

## Business recommendations

1. **Prioritize on-time delivery.** Reducing `delivery_delay_days` to zero has the highest expected impact on review scores. Even a 1-day delay significantly increases the probability of a 1-star review.

2. **Flag high-risk orders proactively.** Orders with long `days_to_carrier` (slow seller pickup) can be identified early and escalated before the customer notices the delay.

3. **Improve product descriptions.** `avg_product_description_length` ranks in the top 7 features — detailed descriptions reduce expectation mismatches and correlate with higher satisfaction.

---

## Tech stack

| Layer | Tools |
|-------|-------|
| Data manipulation | `pandas`, `numpy` |
| Modelling | `scikit-learn` |
| Visualisation | `matplotlib`, `seaborn` |
| Model persistence | `joblib` |
| Environment | Python 3.11, Jupyter |
| Version control | git, GitHub |

---

## About the author

**Haderson Alexander Osorio** — Data Scientist with 8+ years of experience in industrial environments (AkzoNobel, Barentz, Pinturas Prolac). Background in Biological Engineering and Data Analytics. Specializes in connecting raw operational data to strategic business decisions.

- LinkedIn: [linkedin.com/in/alexanderosorioanalytics](https://www.linkedin.com/in/alexanderosorioanalytics/)
- GitHub: [github.com/Halexoh](https://github.com/Halexoh)
- Email: halexoh@gmail.com

---

*"Data is the bridge between technical expertise and strategic decision-making."*
