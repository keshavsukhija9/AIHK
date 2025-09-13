import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.feature_columns = []
        
    def load_data(self, file_path='bengaluru_house_data.csv'):
        """Load and return the dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully! Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"File {file_path} not found. Creating sample data for demonstration...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample Bengaluru house data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        locations = ['Electronic City', 'Whitefield', 'Sarjapur Road', 'Marathahalli', 
                    'BTM Layout', 'Koramangala', 'Indiranagar', 'HSR Layout']
        area_types = ['Super built-up Area', 'Plot Area', 'Built-up Area', 'Carpet Area']
        sizes = ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '4+ BHK']
        
        data = {
            'area_type': np.random.choice(area_types, n_samples),
            'availability': ['Ready To Move'] * n_samples,
            'location': np.random.choice(locations, n_samples),
            'size': np.random.choice(sizes, n_samples),
            'total_sqft': np.random.randint(500, 3000, n_samples),
            'bath': np.random.randint(1, 5, n_samples),
            'balcony': np.random.randint(0, 4, n_samples),
        }
        
        # Generate realistic prices based on features
        base_prices = {'1 BHK': 30, '2 BHK': 50, '3 BHK': 80, '4 BHK': 120, '4+ BHK': 150}
        location_multipliers = {'Electronic City': 0.8, 'Whitefield': 1.2, 'Sarjapur Road': 0.9,
                               'Marathahalli': 1.1, 'BTM Layout': 1.0, 'Koramangala': 1.5,
                               'Indiranagar': 1.6, 'HSR Layout': 1.3}
        
        prices = []
        for i in range(n_samples):
            base_price = base_prices[data['size'][i]]
            loc_mult = location_multipliers[data['location'][i]]
            sqft_factor = data['total_sqft'][i] / 1000
            price = base_price * loc_mult * sqft_factor + np.random.normal(0, 10)
            prices.append(max(20, price))  # Minimum price of 20 lakhs
            
        data['price'] = prices
        
        df = pd.DataFrame(data)
        print(f"Sample data created with shape: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Clean and preprocess the data"""
        print("Cleaning data...")
        
        # Handle missing values
        df = df.dropna()
        
        # Clean total_sqft column (handle ranges like "1000-1200")
        def convert_sqft(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, str):
                if '-' in x:
                    try:
                        parts = x.split('-')
                        return (float(parts[0]) + float(parts[1])) / 2
                    except:
                        return np.nan
                else:
                    try:
                        return float(x)
                    except:
                        return np.nan
            return float(x)
        
        df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
        df = df.dropna(subset=['total_sqft'])
        
        # Extract BHK from size column
        def extract_bhk(size):
            if pd.isna(size):
                return np.nan
            if 'BHK' in str(size):
                try:
                    return int(str(size).split()[0])
                except:
                    return np.nan
            elif 'Bedroom' in str(size):
                try:
                    return int(str(size).split()[0])
                except:
                    return np.nan
            return np.nan
        
        df['bhk'] = df['size'].apply(extract_bhk)
        df = df.dropna(subset=['bhk'])
        
        # Remove outliers
        df = df[df['total_sqft'] > 300]  # Minimum 300 sqft
        df = df[df['price'] > 10]  # Minimum 10 lakhs
        df = df[df['price'] < 1000]  # Maximum 1000 lakhs
        
        # Create price per sqft feature
        df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
        
        print(f"Data cleaned. Final shape: {df.shape}")
        return df
    
    def feature_engineering(self, df):
        """Engineer features for the model"""
        print("Engineering features...")
        
        # Encode categorical variables
        categorical_columns = ['area_type', 'location', 'availability']
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Select features for modeling
        feature_cols = ['total_sqft', 'bath', 'balcony', 'bhk']
        for col in categorical_columns:
            if col + '_encoded' in df.columns:
                feature_cols.append(col + '_encoded')
        
        self.feature_columns = feature_cols
        print(f"Features selected: {feature_cols}")
        
        return df
    
    def train_models(self, X, y):
        """Train multiple models and find the best one"""
        print("Training models...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.3f}")
        
        # Find best model based on R² score
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        self.models = results
        
        print(f"\nBest model: {best_model_name} with R² score: {results[best_model_name]['r2']:.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def predict_price(self, features):
        """Predict house price for given features"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Prepare features
        feature_array = np.array(features).reshape(1, -1)
        
        # Check if scaling is needed (for linear models)
        model_name = None
        for name, model_info in self.models.items():
            if model_info['model'] == self.best_model:
                model_name = name
                break
        
        if model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
            feature_array = self.scaler.transform(feature_array)
        
        prediction = self.best_model.predict(feature_array)[0]
        return prediction
    
    def plot_results(self):
        """Plot model performance and feature importance"""
        if not self.models:
            print("No models trained yet.")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Model comparison
        plt.subplot(2, 3, 1)
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        
        plt.bar(model_names, r2_scores)
        plt.title('Model Comparison (R² Score)')
        plt.xticks(rotation=45)
        plt.ylabel('R² Score')
        
        # Plot 2: Best model predictions vs actual
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['r2'])
        best_results = self.models[best_model_name]
        
        plt.subplot(2, 3, 2)
        plt.scatter(best_results['y_test'], best_results['y_pred'], alpha=0.6)
        plt.plot([best_results['y_test'].min(), best_results['y_test'].max()], 
                [best_results['y_test'].min(), best_results['y_test'].max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{best_model_name} - Predictions vs Actual')
        
        # Plot 3: Residuals
        plt.subplot(2, 3, 3)
        residuals = best_results['y_test'] - best_results['y_pred']
        plt.scatter(best_results['y_pred'], residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Plot 4: Feature importance (for tree-based models)
        if hasattr(self.best_model, 'feature_importances_'):
            plt.subplot(2, 3, 4)
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.bar(range(len(importances)), importances[indices])
            plt.title('Feature Importance')
            plt.xticks(range(len(importances)), [self.feature_columns[i] for i in indices], rotation=45)
        
        # Plot 5: Model metrics comparison
        plt.subplot(2, 3, 5)
        metrics = ['rmse', 'mae']
        x = np.arange(len(model_names))
        width = 0.35
        
        rmse_values = [self.models[name]['rmse'] for name in model_names]
        mae_values = [self.models[name]['mae'] for name in model_names]
        
        plt.bar(x - width/2, rmse_values, width, label='RMSE')
        plt.bar(x + width/2, mae_values, width, label='MAE')
        plt.xlabel('Models')
        plt.ylabel('Error')
        plt.title('Error Metrics Comparison')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model_report(self):
        """Save detailed model performance report"""
        if not self.models:
            print("No models trained yet.")
            return
        
        report = "# House Price Prediction Model Report\n\n"
        report += f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Model Performance Summary\n\n"
        report += "| Model | RMSE | MAE | R² Score |\n"
        report += "|-------|------|-----|----------|\n"
        
        for name, results in self.models.items():
            report += f"| {name} | {results['rmse']:.2f} | {results['mae']:.2f} | {results['r2']:.3f} |\n"
        
        # Best model details
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['r2'])
        best_results = self.models[best_model_name]
        
        report += f"\n## Best Model: {best_model_name}\n\n"
        report += f"- **R² Score**: {best_results['r2']:.3f}\n"
        report += f"- **RMSE**: {best_results['rmse']:.2f} lakhs\n"
        report += f"- **MAE**: {best_results['mae']:.2f} lakhs\n\n"
        
        report += "## Features Used\n\n"
        for i, feature in enumerate(self.feature_columns, 1):
            report += f"{i}. {feature}\n"
        
        # Feature importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            report += "\n## Feature Importance\n\n"
            importances = self.best_model.feature_importances_
            feature_importance = list(zip(self.feature_columns, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for feature, importance in feature_importance:
                report += f"- **{feature}**: {importance:.3f}\n"
        
        with open('model_report.md', 'w') as f:
            f.write(report)
        
        print("Model report saved as 'model_report.md'")

def main():
    """Main function to run the house price prediction pipeline"""
    predictor = HousePricePredictor()
    
    # Load data
    df = predictor.load_data()
    print(f"\nOriginal data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Clean data
    df_clean = predictor.clean_data(df)
    
    # Feature engineering
    df_processed = predictor.feature_engineering(df_clean)
    
    # Prepare features and target
    X = df_processed[predictor.feature_columns]
    y = df_processed['price']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Train models
    X_train, X_test, y_train, y_test = predictor.train_models(X, y)
    
    # Generate visualizations and report
    predictor.plot_results()
    predictor.save_model_report()
    
    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    # Example: 1200 sqft, 2 bath, 1 balcony, 2 BHK, area_type=0, location=0, availability=0
    example_features = [1200, 2, 1, 2, 0, 0, 0]  # Adjust based on actual features
    if len(example_features) == len(predictor.feature_columns):
        try:
            predicted_price = predictor.predict_price(example_features)
            print(f"Predicted price for example house: ₹{predicted_price:.2f} lakhs")
        except Exception as e:
            print(f"Error in prediction: {e}")
    
    print("\nTraining completed! Check 'model_performance.png' and 'model_report.md' for results.")

if __name__ == "__main__":
    main()