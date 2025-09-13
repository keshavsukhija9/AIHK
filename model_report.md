# House Price Prediction Model Report

Date: 2025-09-13 13:21:05

## Model Performance Summary

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Linear Regression | 43.11 | 33.85 | 0.803 |
| Ridge Regression | 43.08 | 33.82 | 0.804 |
| Lasso Regression | 42.79 | 33.64 | 0.806 |
| Random Forest | 20.96 | 15.08 | 0.953 |
| Gradient Boosting | 16.50 | 12.14 | 0.971 |

## Best Model: Gradient Boosting

- **R² Score**: 0.971
- **RMSE**: 16.50 lakhs
- **MAE**: 12.14 lakhs

## Features Used

1. total_sqft
2. bath
3. balcony
4. bhk
5. area_type_encoded
6. location_encoded
7. availability_encoded

## Feature Importance

- **bhk**: 0.525
- **total_sqft**: 0.360
- **location_encoded**: 0.115
- **area_type_encoded**: 0.000
- **balcony**: 0.000
- **bath**: 0.000
- **availability_encoded**: 0.000
