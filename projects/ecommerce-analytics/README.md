# E-commerce Analytics & Machine Learning Project

## Overview

This project analyzes a Brazilian e-commerce dataset to identify key drivers of customer satisfaction and operational performance.

It was developed following an end-to-end data science workflow, from raw data ingestion to modeling and business insights.

---

## Business Problem

E-commerce companies need to understand:

- What drives customer satisfaction (review scores)
- What factors contribute to delivery delays
- How operational and product variables impact performance

This project aims to transform raw transactional data into actionable insights.

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

## 📈 Business Impact (Simulated)

Based on the analysis:

- Reducing delivery delays could significantly improve customer satisfaction scores
- Logistics optimization has high potential to reduce negative reviews
- Product-related features can help prioritize high-impact items

These insights can support decision-making in operations, logistics, and customer experience.

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

This project demonstrates the ability to work end-to-end: from raw data to business insights and predictive modeling, using a structured and scalable approach.
