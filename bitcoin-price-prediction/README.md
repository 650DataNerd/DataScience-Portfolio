# ₿ Bitcoin Price Prediction & Market Pattern Analysis

## Overview

This project explores historical Bitcoin market data using both supervised and unsupervised machine learning techniques.

The objective was to understand historical price behaviour, engineer predictive features, evaluate regression models, and identify hidden market structures through clustering.

Rather than focusing on a single algorithm, the project demonstrates an end-to-end data science workflow including data preparation, feature engineering, model evaluation, visualization, and exploratory pattern discovery.

---

## Objectives

- Predict future Bitcoin closing prices
- Engineer lag-based market features
- Compare multiple regression models
- Discover hidden market clusters
- Visualize long-term Bitcoin price behaviour

---

## Project Structure

```
bitcoin-price-prediction
├── data
├── images
├── scripts
└── README.md
```

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Matplotlib

---

## Machine Learning Techniques

### Supervised Learning

Regression models were trained using historical Bitcoin price data with engineered lag features to predict future closing prices.

Scripts include:

- supervised_learning.py
- lightgbm_prediction.py
- model_evaluation.py

---

### Unsupervised Learning

The project also investigates market behaviour using clustering algorithms including:

- K-Means
- DBSCAN

to identify natural groupings and potential market anomalies.

---

## Visualizations

The repository includes visualizations showing:

- Historical Bitcoin price trends
- Yearly price movement
- Actual vs predicted prices
- Cluster visualizations
- Correlation analysis

---

## Key Skills Demonstrated

- Feature Engineering
- Machine Learning
- Time Series Analysis
- Data Cleaning
- Model Evaluation
- Data Visualization

---

## Future Improvements

Future versions could include:

- LSTM neural networks
- XGBoost & CatBoost comparison
- Prophet forecasting
- Live market data integration
- Model deployment using FastAPI
