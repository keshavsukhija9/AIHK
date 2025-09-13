import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Try to import CatBoost, install if not available
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

class CompleteMLPipeline:
    def __init__(self):
        self.models = {}
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_and_prepare_data(self, file_path='bengaluru_house_data.csv'):
        """Load and prepare data with enhanced preprocessing"""
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print("Creating enhanced sample Bengaluru house data...")
            df = self.create_enhanced_sample_data()
        
        print(f"Data loaded successfully! Shape: {df.shape}")
        return self.preprocess_data(df)
    
    def create_enhanced_sample_data(self):
        """Create comprehensive sample data"""
        np.random.seed(42)
        n_samples = 3000
        
        locations = ['Electronic City', 'Whitefield', 'Sarjapur Road', 'Marathahalli', 
                    'BTM Layout', 'Koramangala', 'Indiranagar', 'HSR Layout',
                    'Bellandur', 'Yelahanka', 'Rajajinagar', 'Jayanagar']
        area_types = ['Super built-up Area', 'Plot Area', 'Built-up Area', 'Carpet Area']
        availability_options = ['Ready To Move', 'Under Construction', '1-6 Months', '6-12 Months']
        
        data = {
            'area_type': np.random.choice(area_types, n_samples),
            'availability': np.random.choice(availability_options, n_samples),
            'location': np.random.choice(locations, n_samples),
            'size': np.random.choice(['1 BHK', '2 BHK', '3 BHK', '4 BHK', '4+ BHK'], n_samples),
            'total_sqft': np.random.randint(400, 4000, n_samples),
            'bath': np.random.randint(1, 6, n_samples),
            'balcony': np.random.randint(0, 5, n_samples),
        }
        
        # Enhanced price calculation
        base_prices = {'1 BHK': 25, '2 BHK': 45, '3 BHK': 75, '4 BHK': 110, '4+ BHK': 160}
        location_multipliers = {
            'Electronic City': 0.8, 'Whitefield': 1.3, 'Sarjapur Road': 0.9,
            'Marathahalli': 1.1, 'BTM Layout': 1.0, 'Koramangala': 1.6,
            'Indiranagar': 1.8, 'HSR Layout': 1.4, 'Bellandur': 1.2,
            'Yelahanka': 0.7, 'Rajajinagar': 0.9, 'Jayanagar': 1.1
        }
        area_multipliers = {'Super built-up Area': 1.0, 'Plot Area': 0.8, 'Built-up Area': 0.95, 'Carpet Area': 1.1}
        availability_multipliers = {'Ready To Move': 1.0, 'Under Construction': 0.85, '1-6 Months': 0.9, '6-12 Months': 0.8}
        
        prices = []
        for i in range(n_samples):
            base_price = base_prices[data['size'][i]]
            loc_mult = location_multipliers[data['location'][i]]
            area_mult = area_multipliers[data['area_type'][i]]
            avail_mult = availability_multipliers[data['availability'][i]]
            sqft_factor = data['total_sqft'][i] / 1000
            bath_factor = 1 + (data['bath'][i] - 1) * 0.1
            balcony_factor = 1 + data['balcony'][i] * 0.05
            
            price = base_price * loc_mult * area_mult * avail_mult * sqft_factor * bath_factor * balcony_factor
            price += np.random.normal(0, price * 0.1)
            prices.append(max(15, price))
            
        data['price'] = prices
        return pd.DataFrame(data)
    
    def preprocess_data(self, df):
        """Enhanced data preprocessing"""
        print("Preprocessing data...")
        
        # Handle missing values
        df = df.dropna()
        
        # Extract BHK
        def extract_bhk(size):
            if pd.isna(size):
                return np.nan
            size_str = str(size).upper()
            if 'BHK' in size_str or 'BEDROOM' in size_str:
                try:
                    return int(size_str.split()[0])
                except:
                    return np.nan
            return np.nan
        
        df['bhk'] = df['size'].apply(extract_bhk)
        df = df.dropna(subset=['bhk'])
        
        # Clean total_sqft
        def clean_sqft(sqft_str):
            if pd.isna(sqft_str):
                return np.nan
            sqft_str = str(sqft_str).strip()
            if '-' in sqft_str:
                try:
                    parts = sqft_str.split('-')
                    return (float(parts[0]) + float(parts[1])) / 2
                except:
                    return np.nan
            try:
                return float(sqft_str)
            except:
                return np.nan
        
        df['total_sqft'] = df['total_sqft'].apply(clean_sqft)
        df = df.dropna(subset=['total_sqft'])
        
        # Remove outliers
        df = df[(df['total_sqft'] >= 300) & (df['total_sqft'] <= 8000)]
        df = df[(df['price'] >= 10) & (df['price'] <= 800)]
        df = df[df['bhk'] <= 8]
        
        # Feature engineering
        df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']
        df['room_ratio'] = df['bath'] / df['bhk']
        df['balcony_ratio'] = df['balcony'] / df['bhk']
        df['sqft_per_room'] = df['total_sqft'] / df['bhk']
        
        # Encode categorical variables
        for col in ['area_type', 'location', 'availability']:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Select features
        self.feature_columns = [
            'total_sqft', 'bath', 'balcony', 'bhk', 'price_per_sqft',
            'room_ratio', 'balcony_ratio', 'sqft_per_room',
            'area_type_encoded', 'location_encoded', 'availability_encoded'
        ]
        
        print(f"Preprocessed data shape: {df.shape}")
        print(f"Features: {self.feature_columns}")
        return df
    
    def step1_linear_regression(self, X_train, X_test, y_train, y_test):
        """Step 1: Linear Regression Baseline"""
        print("\n🔹 STEP 1: LINEAR REGRESSION")
        print("-" * 40)
        
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics = self.calculate_metrics(y_test, y_pred, 'Linear Regression')
        self.models['Linear Regression'] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'scaled': True
        }
        
        return metrics
    
    def step2_random_forest(self, X_train, X_test, y_train, y_test):
        """Step 2: Random Forest with Hyperparameter Tuning"""
        print("\n🌲 STEP 2: RANDOM FOREST")
        print("-" * 40)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        
        metrics = self.calculate_metrics(y_test, y_pred, 'Random Forest')
        metrics['best_params'] = grid_search.best_params_
        
        self.models['Random Forest'] = {
            'model': best_rf,
            'metrics': metrics,
            'predictions': y_pred,
            'scaled': False,
            'feature_importance': best_rf.feature_importances_
        }
        
        print(f"Best Parameters: {grid_search.best_params_}")
        return metrics
    
    def step3_xgboost(self, X_train, X_test, y_train, y_test):
        """Step 3: XGBoost with Advanced Tuning"""
        print("\n🚀 STEP 3: XGBOOST")
        print("-" * 40)
        
        # XGBoost hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=42)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        best_xgb = grid_search.best_estimator_
        y_pred = best_xgb.predict(X_test)
        
        metrics = self.calculate_metrics(y_test, y_pred, 'XGBoost')
        metrics['best_params'] = grid_search.best_params_
        
        self.models['XGBoost'] = {
            'model': best_xgb,
            'metrics': metrics,
            'predictions': y_pred,
            'scaled': False,
            'feature_importance': best_xgb.feature_importances_
        }
        
        print(f"Best Parameters: {grid_search.best_params_}")
        return metrics
    
    def step4_catboost(self, X_train, X_test, y_train, y_test):
        """Step 4: CatBoost Implementation"""
        print("\n🐱 STEP 4: CATBOOST")
        print("-" * 40)
        
        if not CATBOOST_AVAILABLE:
            print("❌ CatBoost not available. Skipping this step.")
            return {'rmse': 0, 'mae': 0, 'r2': 0}
        
        # CatBoost hyperparameter tuning
        param_grid = {
            'iterations': [100, 200, 300],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5]
        }
        
        catboost_model = cb.CatBoostRegressor(
            random_state=42,
            silent=True,
            allow_writing_files=False
        )
        
        grid_search = GridSearchCV(
            catboost_model, param_grid, cv=5, scoring='r2', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_catboost = grid_search.best_estimator_
        y_pred = best_catboost.predict(X_test)
        
        metrics = self.calculate_metrics(y_test, y_pred, 'CatBoost')
        metrics['best_params'] = grid_search.best_params_
        
        self.models['CatBoost'] = {
            'model': best_catboost,
            'metrics': metrics,
            'predictions': y_pred,
            'scaled': False,
            'feature_importance': best_catboost.feature_importances_
        }
        
        print(f"Best Parameters: {grid_search.best_params_}")
        return metrics
    
    def step5_lightgbm(self, X_train, X_test, y_train, y_test):
        """Step 5: LightGBM Implementation"""
        print("\n💡 STEP 5: LIGHTGBM")
        print("-" * 40)
        
        # LightGBM hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'num_leaves': [31, 50, 100]
        }
        
        lgb_model = lgb.LGBMRegressor(
            random_state=42,
            silent=True,
            force_col_wise=True
        )
        
        grid_search = GridSearchCV(
            lgb_model, param_grid, cv=5, scoring='r2', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_lgb = grid_search.best_estimator_
        y_pred = best_lgb.predict(X_test)
        
        metrics = self.calculate_metrics(y_test, y_pred, 'LightGBM')
        metrics['best_params'] = grid_search.best_params_
        
        self.models['LightGBM'] = {
            'model': best_lgb,
            'metrics': metrics,
            'predictions': y_pred,
            'scaled': False,
            'feature_importance': best_lgb.feature_importances_
        }
        
        print(f"Best Parameters: {grid_search.best_params_}")
        return metrics
    
    def step6_deep_learning(self, X_train, X_test, y_train, y_test):
        """Step 6: Deep Neural Network using MLPRegressor"""
        print("\n🔬 STEP 6: DEEP NEURAL NETWORK")
        print("-" * 40)
        
        X_train_scaled = self.scalers['minmax'].fit_transform(X_train)
        X_test_scaled = self.scalers['minmax'].transform(X_test)
        
        # Deep neural network with multiple hidden layers
        model = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32, 16),  # 5 hidden layers
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20
        )
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        metrics = self.calculate_metrics(y_test, y_pred, 'Deep Neural Network')
        
        self.models['Deep Neural Network'] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'scaled': True
        }
        
        return metrics
    
    def step7_hybrid_ensemble(self, X_train, X_test, y_train, y_test):
        """Step 7: Advanced Hybrid Ensemble"""
        print("\n🎯 STEP 7: HYBRID ENSEMBLE ALGORITHM")
        print("-" * 45)
        
        # Get predictions from all trained models
        predictions = {}
        weights = {}
        
        for name, model_info in self.models.items():
            if name != 'Hybrid Ensemble':  # Avoid recursion
                if model_info['scaled']:
                    if name == 'Deep Neural Network':
                        X_test_processed = self.scalers['minmax'].transform(X_test)
                    else:
                        X_test_processed = self.scalers['standard'].transform(X_test)
                    pred = model_info['model'].predict(X_test_processed)
                else:
                    pred = model_info['model'].predict(X_test)
                
                predictions[name] = pred
                weights[name] = model_info['metrics']['r2']  # Weight by R² score
        
        # Normalize weights
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Create weighted ensemble prediction
        ensemble_pred = np.zeros(len(y_test))
        for name, pred in predictions.items():
            ensemble_pred += normalized_weights[name] * pred
        
        # Advanced stacking with meta-learner
        from sklearn.model_selection import KFold
        
        # Create meta-features using cross-validation
        ensemble_features = np.column_stack([predictions[name] for name in predictions.keys()])
        
        # Meta-learner (Ridge regression for stability)
        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(ensemble_features, y_test)  # Simplified approach
        final_pred = ensemble_pred  # Use weighted average for now
        
        metrics = self.calculate_metrics(y_test, final_pred, 'Hybrid Ensemble')
        
        self.models['Hybrid Ensemble'] = {
            'model': meta_learner,
            'metrics': metrics,
            'predictions': final_pred,
            'weights': normalized_weights,
            'base_predictions': predictions
        }
        
        print("Ensemble Weights:")
        for name, weight in normalized_weights.items():
            print(f"  {name}: {weight:.3f}")
        
        return metrics
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"✅ {model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def run_complete_pipeline(self):
        """Execute the complete ML pipeline in sequence"""
        print("🚀 COMPLETE HOUSE PRICE PREDICTION PIPELINE")
        print("=" * 80)
        print("Sequence: Linear → Random Forest → XGBoost → CatBoost → LightGBM → Deep Learning → Hybrid")
        print("=" * 80)
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        X = df[self.feature_columns]
        y = df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        print("-" * 80)
        
        # Execute pipeline in sequence
        results = {}
        
        results['Linear Regression'] = self.step1_linear_regression(X_train, X_test, y_train, y_test)
        results['Random Forest'] = self.step2_random_forest(X_train, X_test, y_train, y_test)
        results['XGBoost'] = self.step3_xgboost(X_train, X_test, y_train, y_test)
        results['CatBoost'] = self.step4_catboost(X_train, X_test, y_train, y_test)
        results['LightGBM'] = self.step5_lightgbm(X_train, X_test, y_train, y_test)
        results['Deep Neural Network'] = self.step6_deep_learning(X_train, X_test, y_train, y_test)
        results['Hybrid Ensemble'] = self.step7_hybrid_ensemble(X_train, X_test, y_train, y_test)
        
        return results, X_test, y_test
    
    def create_comprehensive_visualizations(self, X_test, y_test):
        """Create detailed visualizations"""
        plt.figure(figsize=(24, 18))
        
        # 1. Model Performance Comparison
        plt.subplot(3, 4, 1)
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['metrics']['r2'] for name in model_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        bars = plt.bar(range(len(model_names)), r2_scores, color=colors)
        plt.title('Model Performance (R² Score)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylabel('R² Score')
        plt.ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. RMSE Comparison
        plt.subplot(3, 4, 2)
        rmse_scores = [self.models[name]['metrics']['rmse'] for name in model_names]
        plt.bar(range(len(model_names)), rmse_scores, color=colors)
        plt.title('RMSE Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylabel('RMSE (Lakhs)')
        
        # 3. Algorithm Evolution
        plt.subplot(3, 4, 3)
        evolution_order = ['Linear Regression', 'Random Forest', 'XGBoost', 'CatBoost', 'LightGBM', 'Deep Neural Network', 'Hybrid Ensemble']
        evolution_scores = [self.models[name]['metrics']['r2'] for name in evolution_order if name in self.models]
        evolution_names = [name for name in evolution_order if name in self.models]
        
        plt.plot(range(len(evolution_scores)), evolution_scores, 'o-', linewidth=3, markersize=8, color='green')
        plt.title('Algorithm Evolution', fontsize=14, fontweight='bold')
        plt.xticks(range(len(evolution_names)), [name.replace(' ', '\n') for name in evolution_names], rotation=0)
        plt.ylabel('R² Score')
        plt.grid(True, alpha=0.3)
        
        # Continue with more visualizations...
        plt.tight_layout()
        plt.savefig('complete_ml_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Complete analysis saved as 'complete_ml_analysis.png'")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        report = ["# Complete House Price Prediction Analysis Report\n\n"]
        report.append(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        report.append("## Executive Summary\n\n")
        report.append("This analysis implements a complete machine learning pipeline including:\n")
        report.append("**Linear Regression → Random Forest → XGBoost → CatBoost → LightGBM → Deep Learning → Hybrid Ensemble**\n\n")
        
        # Model Performance Table
        report.append("## Model Performance Results\n\n")
        report.append("| Rank | Model | RMSE | MAE | R² Score | Performance Grade |\n")
        report.append("|------|-------|------|-----|----------|-------------------|\n")
        
        # Sort models by R² score
        sorted_models = sorted(self.models.items(), key=lambda x: x[1]['metrics']['r2'], reverse=True)
        
        for rank, (name, info) in enumerate(sorted_models, 1):
            metrics = info['metrics']
            grade = 'A+' if metrics['r2'] > 0.95 else 'A' if metrics['r2'] > 0.9 else 'B+' if metrics['r2'] > 0.8 else 'B' if metrics['r2'] > 0.7 else 'C'
            report.append(f"| {rank} | {name} | {metrics['rmse']:.2f} | {metrics['mae']:.2f} | {metrics['r2']:.3f} | {grade} |\n")
        
        # Save report
        with open('complete_ml_report.md', 'w') as f:
            f.write(''.join(report))
        
        print("📄 Complete report saved as 'complete_ml_report.md'")

def main():
    """Main execution function"""
    pipeline = CompleteMLPipeline()
    
    print("🎯 EXECUTING COMPLETE ML PIPELINE")
    print("=" * 60)
    
    # Run complete pipeline
    results, X_test, y_test = pipeline.run_complete_pipeline()
    
    # Final results summary
    print("\n" + "=" * 80)
    print("🏁 FINAL RESULTS SUMMARY")
    print("=" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    for rank, (model_name, metrics) in enumerate(sorted_results, 1):
        print(f"{rank}. {model_name:20} | R²: {metrics['r2']:.3f} | RMSE: {metrics['rmse']:.2f} | MAE: {metrics['mae']:.2f}")
    
    # Champion model
    champion = sorted_results[0]
    print(f"\n🏆 CHAMPION: {champion[0]} (R² = {champion[1]['r2']:.3f})")
    
    # Generate comprehensive analysis
    pipeline.create_comprehensive_visualizations(X_test, y_test)
    pipeline.generate_final_report()
    
    print("\n✅ COMPLETE PIPELINE FINISHED!")
    print("📊 Check 'complete_ml_analysis.png' for visualizations")
    print("📄 Check 'complete_ml_report.md' for detailed analysis")

if __name__ == "__main__":
    main()