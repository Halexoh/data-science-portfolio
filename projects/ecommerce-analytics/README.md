# E-commerce Analytics & Machine Learning Project

## Overview

This project analyzes a Brazilian e-commerce dataset to identify key drivers of customer satisfaction and operational performance.

It was developed following an end-to-end data science workflow, from raw data ingestion to modeling and business insights.

---

## Objective

Identify the main operational and product-related factors that impact customer satisfaction in an e-commerce environment.

---

## Business Problem

E-commerce companies need to understand:

- What drives customer satisfaction (review scores)
- What factors contribute to delivery delays
- How operational and product variables impact performance

This project aims to transform raw transactional data into actionable insights.

---
## 📊 Model Performance

Model performance was evaluated using MAE and R², showing the model captures a meaningful portion of variability in customer satisfaction.


---

## Dataset

The dataset includes multiple relational tables:

- Orders
- Customers
- Products
- Sellers
- Payments
- Reviews
- Geolocation

These were integrated into a single analytical dataset.

---

## Project Structure

```
src/
│
├── data/
│ ├── load_data.py # Data ingestion
│ ├── merge_data.py # Dataset integration
│ ├── preprocess.py # Cleaning and preparation
│ └── transform.py # Feature transformations
│
├── features/
│ └── build_features.py # Feature engineering
│
├── models/
│ └── train_model.py # Model training
```
---

## Key Steps

### 1. Data Engineering

- Merged multiple datasets into a unified structure
- Handled missing values and inconsistencies
- Standardized formats across tables

### 2. Feature Engineering

- Delivery time metrics
- Product-related variables
- Customer behavior indicators

### 3. Modeling

- Built predictive models (regression/classification)
- Evaluated model performance
- Identified key drivers using feature importance

---

## Key Insights

- Delivery delays are strongly associated with lower review scores
- Product characteristics influence customer satisfaction
- Logistics variables explain a significant portion of variability

---

## Business Impact (Simulated)

Based on the analysis:

- Reducing delivery delays could significantly improve customer satisfaction scores
- Logistics optimization has high potential to reduce negative reviews
- Product-related features can help prioritize high-impact items

These insights can support decision-making in operations, logistics, and customer experience.

---
## How to Run

1. Clone the repository:
```bash
git clone https://github.com/Halexoh/data-science-portfolio.git
cd data-science-portfolio/projects/ecommerce-analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
Dataset: Brazilian E-commerce Public Dataset (Olist)
Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

4. Place the CSV files inside:
```bash
data_raw/
```
Expected structure:

```bash
projects/ecommerce-analytics/
│
├── data_raw/
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   └── olist_geolocation_dataset.csv
```

5. Run the pipeline:
```bash
python src/models/train_model.py
```
### What the Pipeline Does
- Loads raw data from multiple tables
- Cleans and preprocesses datasets
- Merges into a unified analytical dataset
- Builds features related to delivery, products and customers
- Trains a machine learning model (Random Forest)
- Evaluates performance using MAE and R²

---

## Tools & Technologies

- Python (pandas, numpy)
- scikit-learn
- Data visualization (matplotlib, seaborn)

---

## Next Steps

- Improve model performance
- Deploy as an API
- Build a dashboard (Power BI or Streamlit)

---

## Key Takeaway

This project demonstrates the ability to translate raw data into actionable business insights and predictive models using a structured, end-to-end data science workflow.
