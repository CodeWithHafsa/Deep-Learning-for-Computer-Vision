# Housing in Buenos Aires

This repository contains a data science workflow for predicting apartment prices in Buenos Aires. The project focuses on cleaning real estate data, engineering meaningful features, and training interpretable regression models to estimate price in USD.

## Workflow Overview

### 1. Data Wrangling (`wrangle()` function)
- Filters listings for Buenos Aires apartments under $100,000 USD
- Cleans `lat-lon` coordinates
- Extracts boroughs from location strings
- Removes high-cardinality and low-information columns
- Handles missing and outlier values
- Collapses rare boroughs into `"Other"` for model stability

### 2. Exploratory Analysis
- Histograms and scatter plots via Matplotlib's OOP interface
- Mapbox scatter plot with Plotly to visualize spatial price patterns

### 3. Modeling
- Uses a `Pipeline` with `SimpleImputer` and `OneHotEncoder` via `ColumnTransformer`
- Final model: `Ridge` or `LinearRegression` depending on the task
- Feature importances extracted from model coefficients

### 4. Evaluation
- Baseline MAE (mean absolute error) via mean prediction
- Model MAE from Ridge/LinearRegression
- Visual comparison of features affecting price



## Outputs

- `feature_importances.png`: Horizontal bar chart showing impact of borough and numeric features on predicted price
- Test predictions stored as a pandas Series
- Final model outputs compared against expected values

## 🧠 Key Learnings

- Importance of preprocessing pipelines for reproducibility
- Effects of regularization on coefficient magnitudes
- Feature selection improves generalization
- Geospatial features (lat/lon) help capture neighborhood-level price differences


## 🛠️ Dependencies

- `pandas`, `numpy`, `matplotlib`, `scikit-learn`
- `plotly`, `category_encoders` for extended exploration
- Python ≥ 3.8

Install via:

```bash
pip install -r requirements.txt