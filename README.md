# NDVI-Based Land Cover Classification

## Overview

This project builds a machine learning pipeline for classifying land-cover or crop types using NDVI (Normalized Difference Vegetation Index) time-series data collected from satellite observations.

The workflow includes:

- Missing value interpolation
- Signal smoothing using Savitzky-Golay filtering
- Time-series feature extraction
- Logistic Regression classification
- Hyperparameter tuning using GridSearchCV

---

## Dataset

The dataset consists of:

- `hacktrain.csv` – training data with class labels
- `hacktest.csv` – test data for prediction

NDVI observations are stored in columns ending with `_N`.

Example:

```
20200101_N
20200117_N
20200202_N
...
```

---

## Methodology

### 1. Preprocessing

- Extract NDVI columns
- Sort observations chronologically
- Fill missing values using linear interpolation
- Smooth NDVI curves using Savitzky-Golay filtering

### 2. Feature Engineering

Extracted features include:

- Mean NDVI
- Median NDVI
- Standard deviation
- Minimum NDVI
- Maximum NDVI
- Range
- Trend (slope)

Seasonal statistics:

- Seasonal mean NDVI
- Seasonal standard deviation

### 3. Model Training

Pipeline:

```python
StandardScaler()
LogisticRegression(class_weight="balanced")
```

Hyperparameter tuning:

```python
GridSearchCV
```

with 5-fold cross validation.

### 4. Prediction

The trained model generates predictions for the test dataset and exports:

```text
submission.csv
```

---

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn

---

## Project Structure

```text
├── hacktrain.csv
├── hacktest.csv
├── notebook.ipynb
├── submission.csv
└── README.md
```

---

## Future Improvements

- XGBoost / LightGBM models
- Advanced temporal feature extraction
- Peak NDVI analysis
- Area-under-curve features
- Ensemble learning
- Feature importance analysis

---

## Results

The project provides a complete end-to-end pipeline for NDVI-based classification, including preprocessing, feature engineering, model optimization, and prediction generation.

---

## Author

Developed as part of a Summer Analytics Hackathon project.
