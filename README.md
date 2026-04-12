# Data Science Portfolio – Alexander Osorio

Portfolio of end-to-end data science and machine learning projects focused on solving real-world business problems using data-driven approaches.

---

## About Me

Data Analytics Specialist with experience in industrial processes, combining data science, machine learning, and business understanding to drive impactful decisions.

- Data Analysis & Visualization
- Machine Learning (Regression, Classification)
- Business Problem Solving
- Python (pandas, scikit-learn)
- SQL & Data Modeling

---

## Projects

### E-commerce Analytics Project
**Goal:** Identify key drivers of customer satisfaction and delivery delays in an e-commerce business.
**Key Insight:** Delivery delays are strongly correlated with lower review scores, highlighting logistics as a critical factor for customer satisfaction.


- Data cleaning and transformation (multiple sources)
- Feature engineering
- Exploratory Data Analysis (EDA)
- Business insights and recommendations

Tools: Python, pandas, seaborn, matplotlib

---

### Kaggle Machine Learning Project
**Goal:** Identify key drivers of customer satisfaction and delivery delays in an e-commerce business.

- Model training (Random Forest, CatBoost)
- Cross-validation
- Feature selection
- Performance optimization

Result: Top 48% in competition

---

### Time Series / Forecasting (Coming Soon)
**Goal:** Predict demand / trends using historical data.

---

## Tech Stack

- Python (pandas, numpy, scikit-learn)
- SQL
- Power BI / Visualization tools
- Git & GitHub
- Jupyter Notebooks

---

## What I Bring

- Ability to connect data with business impact
- Experience working with real industrial data
- End-to-end project development
- Strong analytical and problem-solving mindset

---

## Contact

- LinkedIn: https://www.linkedin.com/in/alexanderosorioanalytics/
- GitHub: https://github.com/Halexoh

---

## Dataset

Dataset used: Brazilian E-commerce Public Dataset (Olist)

Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

Note: Raw data is not included due to size limitations.

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

3. Download the dataset and place the CSV files inside:
```bash
data_raw/
```
4. Run the training pipeline:
```bash
python -m src.models.train_model
```
This script performs:

- Data loading from multiple sources
- Data cleaning and preprocessing
- Dataset integration
- Feature engineering
- Model training (Random Forest)
- Model evaluation (MAE, R²)

If you find this portfolio interesting, feel free to connect!
