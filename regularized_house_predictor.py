import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, validation_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class RegularizedHousePricePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = RobustScaler()  # More robust to outliers
        self.feature_selector = None
        self.best_model = None
        self.feature_columns = []
        self.cv_scores = {}
        
    def load_data(self, file_path='bengaluru_house_data.csv'):
        """Load and return the dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully! Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"File {file_path} not found. Creating sample data...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create realistic sample data with proper variance"""
        np.random.seed(42)
        n_samples = 2000  # Larger dataset
        
        locations = ['Electronic City', 'Whitefield', 'Sarjapur Road', 'Marathahalli', 
                    'BTM Layout', 'Koramangala', 'Indiranagar', 'HSR Layout', 'Hebbal',
                    'JP Nagar', 'Jayanagar', 'Rajajinagar', 'Malleshwaram', 'Yelahanka']
        area_types = ['Super built-up Area', 'Plot Area', 'Built-up Area', 'Carpet Area']
        sizes = ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '4+ BHK']
        
        # Create more realistic distributions
        data = {
            'area_type': np.random.choice(area_types, n_samples),
            'availability': ['Ready To Move'] * n_samples,
            'location': np.random.choice(locations, n_samples),
            'size': np.random.choice(sizes, n_samples, p=[0.15, 0.35, 0.3, 0.15, 0.05]),
            'total_sqft': np.random.lognormal(mean=7.2, sigma=0.4, size=n_samples),  # More realistic distribution
            'bath': np.random.poisson(lam=2, size=n_samples) + 1,
            'balcony': np.random.poisson(lam=1, size=n_samples),
        }
        
        # Clip values to realistic ranges
        data['total_sqft'] = np.clip(data['total_sqft'], 400, 4000)
        data['bath'] = np.clip(data['bath'], 1, 5)
        data['balcony'] = np.clip(data['balcony'], 0, 4)
        
        # Generate realistic prices with proper noise and non-linear relationships
        base_prices = {'1 BHK': 25, '2 BHK': 45, '3 BHK': 70, '4 BHK': 110, '4+ BHK': 160}
        location_multipliers = {
            'Electronic City': 0.7, 'Whitefield': 1.1, 'Sarjapur Road': 0.85,
            'Marathahalli': 1.05, 'BTM Layout': 0.95, 'Koramangala': 1.4,
            'Indiranagar': 1.5, 'HSR Layout': 1.25, 'Hebbal': 0.9,
            'JP Nagar': 1.0, 'Jayanagar': 1.15, 'Rajajinagar': 1.2,
            'Malleshwaram': 1.3, 'Yelahanka': 0.8
        }
        
        prices = []
        for i in range(n_samples):
            base_price = base_prices[data['size'][i]]
            loc_mult = location_multipliers[data['location'][i]]
            
            # Non-linear sqft relationship
            sqft_factor = (data['total_sqft'][i] / 1000) ** 0.8
            
            # Additional factors
            bath_factor = 1 + (data['bath'][i] - 2) * 0.05
            balcony_factor = 1 + data['balcony'][i] * 0.03
            
            # Add realistic noise (heteroscedastic)
            noise_level = base_price * 0.15  # 15% noise
            noise = np.random.normal(0, noise_level)
            
            price = base_price * loc_mult * sqft_factor * bath_factor * balcony_factor + noise
            prices.append(max(15, price))  # Minimum price
            
        data['price'] = prices
        
        df = pd.DataFrame(data)
        print(f"Sample data created with shape: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Enhanced data cleaning with outlier detection"""
        print("Cleaning data with robust outlier detection...")
        
        # Handle missing values
        df = df.dropna()
        
        # Convert total_sqft to numeric
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
        
        # Extract BHK
        def extract_bhk(size):
            if pd.isna(size):
                return np.nan
            if 'BHK' in str(size):
                try:
                    return int(str(size).split()[0])
                except:
                    return np.nan
            return np.nan
        
        df['bhk'] = df['size'].apply(extract_bhk)
        df = df.dropna(subset=['bhk'])
        
        # Robust outlier removal using IQR method
        numeric_cols = ['total_sqft', 'price', 'bath', 'balcony']
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Additional reasonable constraints
        df = df[df['total_sqft'] > 300]
        df = df[df['price'] > 10]
        df = df[df['price'] < 500]  # More reasonable upper limit
        
        # Create derived features
        df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
        df['room_to_bath_ratio'] = df['bhk'] / df['bath']
        
        print(f"Data cleaned. Final shape: {df.shape}")
        return df
    
    def feature_engineering(self, df):
        """Enhanced feature engineering with proper encoding"""
        print("Engineering features with proper encoding...")
        
        # One-hot encode categorical variables (prevents overfitting from label encoding)
        categorical_columns = ['area_type', 'location', 'availability']
        
        for col in categorical_columns:
            if col in df.columns:
                # One-hot encoding for categorical variables
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
        
        # Select base numerical features
        numerical_features = ['total_sqft', 'bath', 'balcony', 'bhk', 'price_per_sqft', 'room_to_bath_ratio']
        
        # Add one-hot encoded features
        feature_cols = numerical_features.copy()
        for col in categorical_columns:
            dummy_cols = [c for c in df.columns if c.startswith(f'{col}_')]
            feature_cols.extend(dummy_cols)
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = feature_cols
        
        print(f"Features selected: {len(feature_cols)} features")
        return df
    
    def train_models_with_cv(self, X, y):
        """Train models with proper cross-validation and regularization"""
        print("Training models with cross-validation...")
        
        # Stratified split to maintain price distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection to reduce overfitting
        self.feature_selector = SelectKBest(score_func=f_regression, k=min(15, X_train.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Define models with proper regularization
        models = {
            'Ridge Regression': Ridge(alpha=10.0),  # Increased regularization
            'Lasso Regression': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(
                n_estimators=50,  # Reduced to prevent overfitting
                max_depth=10,     # Limited depth
                min_samples_split=10,  # Increased minimum samples
                min_samples_leaf=5,    # Increased leaf size
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=50,  # Reduced
                max_depth=6,      # Limited depth
                learning_rate=0.1, # Slower learning
                subsample=0.8,    # Subsampling for regularization
                random_state=42
            )
        }
        
        # 5-fold cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        for name, model in models.items():
            print(f"Training {name} with CV...")
            
            if name in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
                # Use selected features for linear models
                cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='r2')
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                X_train_model = X_train_selected
            else:
                # Use original features for tree models
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                X_train_model = X_train
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Store CV scores
            self.cv_scores[name] = cv_scores
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"{name}:")
            print(f"  CV R² Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"  Test R² Score: {r2:.3f}")
            print(f"  RMSE: {rmse:.2f}")
        
        # Select best model based on CV score (more reliable than test score)
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        self.models = results
        
        print(f"\nBest model: {best_model_name}")
        print(f"CV R² Score: {results[best_model_name]['cv_mean']:.3f} (±{results[best_model_name]['cv_std']:.3f})")
        print(f"Test R² Score: {results[best_model_name]['r2']:.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def plot_validation_analysis(self):
        """Plot comprehensive validation analysis"""
        if not self.models:
            print("No models trained yet.")
            return
        
        plt.figure(figsize=(20, 15))
        
        # 1. Cross-validation scores comparison
        plt.subplot(3, 4, 1)
        model_names = list(self.cv_scores.keys())
        cv_means = [self.cv_scores[name].mean() for name in model_names]
        cv_stds = [self.cv_scores[name].std() for name in model_names]
        
        plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5)
        plt.title('Cross-Validation R² Scores')
        plt.xticks(rotation=45)
        plt.ylabel('R² Score')
        plt.ylim(0, 1)
        
        # 2. CV vs Test R² comparison
        plt.subplot(3, 4, 2)
        test_scores = [self.models[name]['r2'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        plt.bar(x - width/2, cv_means, width, label='CV R²', alpha=0.8)
        plt.bar(x + width/2, test_scores, width, label='Test R²', alpha=0.8)
        plt.xlabel('Models')
        plt.ylabel('R² Score')
        plt.title('CV vs Test R² Scores')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.ylim(0, 1)
        
        # 3. Best model predictions vs actual
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['cv_mean'])
        best_results = self.models[best_model_name]
        
        plt.subplot(3, 4, 3)
        plt.scatter(best_results['y_test'], best_results['y_pred'], alpha=0.6)
        plt.plot([best_results['y_test'].min(), best_results['y_test'].max()], 
                [best_results['y_test'].min(), best_results['y_test'].max()], 'r--', lw=2)
        plt.xlabel('Actual Price (lakhs)')
        plt.ylabel('Predicted Price (lakhs)')
        plt.title(f'{best_model_name} - Predictions vs Actual')
        
        # 4. Residuals plot
        plt.subplot(3, 4, 4)
        residuals = best_results['y_test'] - best_results['y_pred']
        plt.scatter(best_results['y_pred'], residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price (lakhs)')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # 5. RMSE comparison
        plt.subplot(3, 4, 5)
        rmse_values = [self.models[name]['rmse'] for name in model_names]
        plt.bar(model_names, rmse_values)
        plt.title('RMSE Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('RMSE')
        
        # 6. MAE comparison
        plt.subplot(3, 4, 6)
        mae_values = [self.models[name]['mae'] for name in model_names]
        plt.bar(model_names, mae_values)
        plt.title('MAE Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('MAE')
        
        # 7. CV score distribution for best model
        plt.subplot(3, 4, 7)
        best_cv_scores = self.cv_scores[best_model_name]
        plt.hist(best_cv_scores, bins=5, alpha=0.7, edgecolor='black')
        plt.axvline(best_cv_scores.mean(), color='red', linestyle='--', 
                   label=f'Mean: {best_cv_scores.mean():.3f}')
        plt.xlabel('R² Score')
        plt.ylabel('Frequency')
        plt.title(f'{best_model_name} - CV Score Distribution')
        plt.legend()
        
        # 8. Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            plt.subplot(3, 4, 8)
            importances = self.best_model.feature_importances_
            
            # Get feature names (need to handle feature selection)
            if hasattr(self, 'feature_selector') and self.feature_selector is not None:
                selected_features = self.feature_selector.get_support()
                feature_names = [self.feature_columns[i] for i, selected in enumerate(selected_features) if selected]
            else:
                feature_names = self.feature_columns
            
            # Ensure we don't exceed available features
            max_features = min(10, len(importances), len(feature_names))
            indices = np.argsort(importances)[::-1][:max_features]
            
            plt.bar(range(len(indices)), importances[indices])
            plt.title(f'Top {len(indices)} Feature Importances')
            if len(indices) > 0 and len(feature_names) > max(indices):
                plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            else:
                plt.xticks(range(len(indices)), [f'Feature_{i}' for i in range(len(indices))], rotation=90)
        
        # 9. Model complexity vs performance
        plt.subplot(3, 4, 9)
        complexity_scores = {
            'Ridge Regression': 1,
            'Lasso Regression': 1,
            'ElasticNet': 1.5,
            'Random Forest': 3,
            'Gradient Boosting': 3.5
        }
        
        model_complexity = [complexity_scores.get(name, 2) for name in model_names]
        plt.scatter(model_complexity, cv_means, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            plt.annotate(name, (model_complexity[i], cv_means[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Model Complexity')
        plt.ylabel('CV R² Score')
        plt.title('Complexity vs Performance')
        
        # 10. Error distribution
        plt.subplot(3, 4, 10)
        plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        
        # 11. Overfitting detection (CV vs Test gap)
        plt.subplot(3, 4, 11)
        gaps = [cv_means[i] - test_scores[i] for i in range(len(model_names))]
        colors = ['red' if gap > 0.05 else 'green' for gap in gaps]
        plt.bar(model_names, gaps, color=colors, alpha=0.7)
        plt.axhline(y=0.05, color='red', linestyle='--', label='Overfitting threshold')
        plt.title('Overfitting Detection (CV - Test R²)')
        plt.xticks(rotation=45)
        plt.ylabel('R² Difference')
        plt.legend()
        
        # 12. Performance summary table
        plt.subplot(3, 4, 12)
        plt.axis('off')
        
        # Create summary table
        summary_data = []
        for name in model_names:
            summary_data.append([
                name,
                f"{cv_means[model_names.index(name)]:.3f}",
                f"{test_scores[model_names.index(name)]:.3f}",
                f"{self.models[name]['rmse']:.1f}"
            ])
        
        table = plt.table(cellText=summary_data,
                         colLabels=['Model', 'CV R²', 'Test R²', 'RMSE'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        plt.title('Performance Summary', pad=20)
        
        plt.tight_layout()
        plt.savefig('regularized_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print overfitting analysis
        print("\n" + "="*60)
        print("OVERFITTING ANALYSIS")
        print("="*60)
        for name in model_names:
            cv_score = cv_means[model_names.index(name)]
            test_score = test_scores[model_names.index(name)]
            gap = cv_score - test_score
            
            if gap > 0.05:
                status = "⚠️  OVERFITTED"
            elif gap > 0.02:
                status = "⚡ SLIGHTLY OVERFITTED"
            else:
                status = "✅ WELL GENERALIZED"
            
            print(f"{name}:")
            print(f"  CV R²: {cv_score:.3f} | Test R²: {test_score:.3f} | Gap: {gap:.3f}")
            print(f"  Status: {status}\n")

def main():
    """Main function with proper overfitting prevention"""
    predictor = RegularizedHousePricePredictor()
    
    # Load and process data
    df = predictor.load_data()
    print(f"Original data shape: {df.shape}")
    
    df_clean = predictor.clean_data(df)
    df_processed = predictor.feature_engineering(df_clean)
    
    # Prepare features and target
    X = df_processed[predictor.feature_columns]
    y = df_processed['price']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Train models with proper validation
    X_train, X_test, y_train, y_test = predictor.train_models_with_cv(X, y)
    
    # Generate comprehensive analysis
    predictor.plot_validation_analysis()
    
    print("\n🎯 Regularized model training completed!")
    print("📊 Check 'regularized_model_analysis.png' for comprehensive analysis")
    print("✅ Overfitting issues have been addressed through:")
    print("   - Proper cross-validation")
    print("   - Regularization parameters")
    print("   - Feature selection")
    print("   - Model complexity control")
    print("   - Robust data preprocessing")

if __name__ == "__main__":
    main()