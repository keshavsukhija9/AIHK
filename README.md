# Bengaluru House Price Prediction System

A comprehensive machine learning system for predicting house prices in Bengaluru using multiple algorithms and data preprocessing techniques.

## Project Overview

This project implements a complete house price prediction pipeline that can:
- Process real estate data from various formats (CSV, Numbers files)
- Clean and preprocess housing data automatically
- Train multiple machine learning models
- Compare model performance and select the best one
- Generate predictions for new properties
- Create detailed performance reports and visualizations

## Features

### Data Processing
- Automatic data cleaning and outlier removal
- Handles various data formats and inconsistencies
- Feature engineering (BHK extraction, price per sqft calculation)
- Missing value handling
- Data validation and quality checks

### Machine Learning Models
- **Linear Regression**: Basic linear relationship modeling
- **Ridge Regression**: L2 regularization for better generalization
- **Lasso Regression**: L1 regularization with feature selection
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Advanced ensemble technique

### Evaluation & Reporting
- Comprehensive model comparison using multiple metrics
- R² score, RMSE, and MAE evaluation
- Feature importance analysis
- Residual plots and prediction visualizations
- Automated report generation

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Required packages:
- pandas (>=1.3.0)
- numpy (>=1.21.0)
- scikit-learn (>=1.0.0)
- matplotlib (>=3.4.0)
- seaborn (>=0.11.0)

## Data Requirements

### Using Your Bengaluru House Data

The system is designed to work with the provided `Bengaluru_House_Data.numbers` file:

1. **Convert to CSV format**:
   - Open the Numbers file in Apple Numbers
   - Go to File → Export To → CSV
   - Save as `bengaluru_house_data.csv`

2. **Expected data columns**:
   - `area_type`: Type of area (Super built-up, Plot, Built-up, Carpet)
   - `availability`: Property availability status
   - `location`: Property location in Bengaluru
   - `size`: Number of BHK or bedrooms
   - `society`: Society/building name
   - `total_sqft`: Total square footage
   - `bath`: Number of bathrooms
   - `balcony`: Number of balconies
   - `price`: Property price (in lakhs)

### Sample Data Generation

If no data file is available, the system automatically generates realistic sample data for demonstration.

## Usage

### Basic Usage

Run the complete pipeline:
```bash
python house_price_predictor.py
```

### Data Processing Only

Process and clean your data:
```bash
python data_processor.py
```

### Interactive Demo

Run the interactive demo with sample predictions:
```bash
python demo.py
```

### Manual Usage

```python
from house_price_predictor import HousePricePredictor

# Initialize predictor
predictor = HousePricePredictor()

# Load and process data
df = predictor.load_data('your_data.csv')
df_clean = predictor.clean_data(df)
df_processed = predictor.feature_engineering(df_clean)

# Train models
X = df_processed[predictor.feature_columns]
y = df_processed['price']
predictor.train_models(X, y)

# Make prediction
# Features: [sqft, bath, balcony, bhk, area_type, location, availability]
features = [1200, 2, 1, 2, 0, 0, 0]
predicted_price = predictor.predict_price(features)
print(f"Predicted price: ₹{predicted_price:.2f} lakhs")
```

## Project Structure

```
├── house_price_predictor.py    # Main ML pipeline implementation
├── data_processor.py           # Data cleaning and preprocessing
├── data_converter.py           # File format conversion utilities
├── demo.py                     # Interactive demonstration
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── Bengaluru_House_Data.numbers # Original data file
```

## Output Files

The system generates several output files:

### Visualizations
- `model_performance.png`: Comprehensive model performance charts

### Reports
- `model_report.md`: Detailed model performance metrics
- `data_summary.md`: Dataset analysis and statistics

### Processed Data
- `bengaluru_house_data_cleaned.csv`: Cleaned and processed dataset

## Model Performance

The system trains and compares multiple models:

1. **Model Comparison**: R² scores across all algorithms
2. **Prediction Accuracy**: Actual vs predicted price plots
3. **Error Analysis**: Residual plots and error distributions
4. **Feature Importance**: Which features matter most for predictions

### Typical Performance Metrics
- **R² Score**: 0.80-0.90 (varies with data quality)
- **RMSE**: 15-25 lakhs (typical prediction error)
- **MAE**: 10-20 lakhs (mean absolute error)

## Data Quality Features

### Automatic Data Cleaning
- Removes duplicate entries
- Handles missing values appropriately
- Converts string ranges to numeric values
- Extracts BHK information from size descriptions
- Standardizes location names

### Outlier Detection
- Filters unrealistic property sizes (<300 sqft or >10,000 sqft)
- Removes extreme price outliers (<10 lakhs or >1000 lakhs)
- Validates bathroom and balcony counts

### Feature Engineering
- Creates price per square foot metric
- Encodes categorical variables
- Generates derived features for better predictions

## Advanced Features

### Cross-Validation
Models are evaluated using cross-validation for robust performance estimation.

### Hyperparameter Optimization
Grid search capabilities for model tuning (can be extended).

### Scalable Architecture
Designed to handle datasets of various sizes and can be extended with additional models.

## Troubleshooting

### Common Issues

1. **Numbers file not readable**:
   - Convert to CSV format manually
   - Ensure proper column headers

2. **Missing dependencies**:
   - Run `pip install -r requirements.txt`
   - Check Python version (3.7+ recommended)

3. **Poor model performance**:
   - Check data quality and completeness
   - Verify feature engineering results
   - Consider data size (minimum 100+ samples recommended)

### Data Format Issues

The system handles common data inconsistencies:
- Price formats (lakhs, crores, with/without currency symbols)
- Square footage ranges (e.g., "1000-1200")
- Various BHK representations ("2 BHK", "2BHK", "2 Bedroom")

## Extension Possibilities

### Additional Models
- XGBoost for enhanced gradient boosting
- Neural networks for complex pattern recognition
- Support Vector Regression for non-linear relationships

### Enhanced Features
- Location-based clustering
- Market trend analysis
- Property age considerations
- Amenity scoring systems

### Integration Options
- Web interface for easy predictions
- API endpoints for integration with apps
- Database connectivity for live data
- Real-time market data integration

## Contributing

To extend this project:

1. Add new models in the `train_models()` method
2. Enhance feature engineering in `feature_engineering()`
3. Improve data cleaning in `clean_data()`
4. Add new visualization types in `plot_results()`

## License

This project is for educational and research purposes. Ensure proper data licensing when using real estate data.

## Contact & Support

For issues or questions about the implementation, please review the code comments and error messages for guidance on troubleshooting.