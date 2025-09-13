import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import tensorflow as tf
import keras
from keras import layers
import warnings
warnings.filterwarnings('ignore')

class AdvancedHousePricePredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        self.label_encoders = {}
        self.feature_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self, file_path='bengaluru_house_data.csv'):
        """Load and prepare data with enhanced preprocessing"""
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print("Creating sample Bengaluru house data...")
            df = self.create_enhanced_sample_data()
        
        print(f"Data loaded successfully! Shape: {df.shape}")
        return self.preprocess_data(df)
    
    def create_enhanced_sample_data(self):
        """Create more comprehensive sample data"""
        np.random.seed(42)
        n_samples = 2000
        
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
        
        # Enhanced price calculation with more factors
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
            price += np.random.normal(0, price * 0.1)  # Add realistic noise
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
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Step 1: Linear Regression"""
        print("\n1. Training Linear Regression...")
        
        # Scale features for linear regression
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
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Step 2: Random Forest"""
        print("\n2. Training Random Forest...")
        
        # Hyperparameter tuning for Random Forest
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
        
        return metrics
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Step 3: XGBoost"""
        print("\n3. Training XGBoost...")
        
        # XGBoost with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
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
        
        return metrics
    
    def train_cnn(self, X_train, X_test, y_train, y_test):
        """Step 4: Convolutional Neural Network (1D CNN for tabular data)"""
        print("\n4. Training 1D CNN...")
        
        # Scale data for neural networks
        X_train_scaled = self.scalers['minmax'].fit_transform(X_train)
        X_test_scaled = self.scalers['minmax'].transform(X_test)
        
        # Reshape for 1D CNN
        X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        
        # Build 1D CNN model
        model = keras.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
            layers.BatchNormalization(),
            layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train with early stopping
        early_stopping = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
        history = model.fit(
            X_train_cnn, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        y_pred = model.predict(X_test_cnn).flatten()
        
        metrics = self.calculate_metrics(y_test, y_pred, '1D CNN')
        
        self.models['1D CNN'] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'scaled': True,
            'history': history
        }
        
        return metrics
    
    def train_deep_learning(self, X_train, X_test, y_train, y_test):
        """Step 5: Deep Neural Network"""
        print("\n5. Training Deep Neural Network...")
        
        X_train_scaled = self.scalers['minmax'].transform(X_train)
        X_test_scaled = self.scalers['minmax'].transform(X_test)
        
        # Build deep neural network
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        # Custom learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=100,
            decay_rate=0.96
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='mse',
            metrics=['mae']
        )
        
        # Train with callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
        ]
        
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        y_pred = model.predict(X_test_scaled).flatten()
        
        metrics = self.calculate_metrics(y_test, y_pred, 'Deep Neural Network')
        
        self.models['Deep Neural Network'] = {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'scaled': True,
            'history': history
        }
        
        return metrics
    
    def train_hybrid_ensemble(self, X_train, X_test, y_train, y_test):
        """Step 6: Hybrid Ensemble Algorithm"""
        print("\n6. Training Hybrid Ensemble...")
        
        # Get predictions from all models
        predictions = {}
        weights = {}
        
        for name, model_info in self.models.items():
            if name not in ['Hybrid Ensemble']:  # Avoid recursion
                if model_info['scaled']:
                    if 'CNN' in name:
                        X_test_reshaped = self.scalers['minmax'].transform(X_test).reshape(X_test.shape[0], X_test.shape[1], 1)
                        pred = model_info['model'].predict(X_test_reshaped).flatten()
                    else:
                        pred = model_info['model'].predict(self.scalers['standard'].transform(X_test))
                else:
                    pred = model_info['model'].predict(X_test)
                
                predictions[name] = pred
                # Weight by R² score
                weights[name] = model_info['metrics']['r2']
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Create weighted ensemble prediction
        ensemble_pred = np.zeros(len(y_test))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        # Advanced ensemble with stacking
        ensemble_features = np.column_stack([predictions[name] for name in predictions.keys()])
        
        # Meta-learner (Ridge regression)
        meta_learner = Ridge(alpha=0.1)
        
        # Create meta-training data using cross-validation
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_X = []
        meta_y = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            fold_predictions = []
            for name, model_info in self.models.items():
                if name not in ['Hybrid Ensemble']:
                    temp_model = model_info['model'].__class__(**model_info['model'].get_params() if hasattr(model_info['model'], 'get_params') else {})
                    
                    if model_info['scaled']:
                        if 'CNN' in name or 'Neural' in name:
                            # Retrain neural networks (simplified for meta-learning)
                            continue
                        else:
                            X_fold_train_scaled = self.scalers['standard'].fit_transform(X_fold_train)
                            temp_model.fit(X_fold_train_scaled, y_fold_train)
                            pred = temp_model.predict(self.scalers['standard'].transform(X_fold_val))
                    else:
                        temp_model.fit(X_fold_train, y_fold_train)
                        pred = temp_model.predict(X_fold_val)
                    
                    fold_predictions.append(pred)
            
            if fold_predictions:
                meta_X.extend(np.column_stack(fold_predictions))
                meta_y.extend(y_fold_val)
        
        if meta_X:
            meta_X = np.array(meta_X)
            meta_y = np.array(meta_y)
            meta_learner.fit(meta_X, meta_y)
            
            # Final ensemble prediction using meta-learner
            stacked_pred = meta_learner.predict(ensemble_features)
        else:
            stacked_pred = ensemble_pred
        
        metrics = self.calculate_metrics(y_test, stacked_pred, 'Hybrid Ensemble')
        
        self.models['Hybrid Ensemble'] = {
            'model': meta_learner,
            'metrics': metrics,
            'predictions': stacked_pred,
            'weights': weights,
            'base_predictions': predictions
        }
        
        return metrics
    
    def calculate_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"{model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def train_all_models(self):
        """Train all models in sequence"""
        print("Starting comprehensive model training pipeline...")
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        X = df[self.feature_columns]
        y = df['price']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        # Train models in sequence
        results = {}
        
        results['Linear Regression'] = self.train_linear_regression(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        results['Random Forest'] = self.train_random_forest(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        results['XGBoost'] = self.train_xgboost(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        results['1D CNN'] = self.train_cnn(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        results['Deep Neural Network'] = self.train_deep_learning(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        results['Hybrid Ensemble'] = self.train_hybrid_ensemble(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        
        return results
    
    def plot_comprehensive_results(self):
        """Create comprehensive visualizations"""
        plt.figure(figsize=(20, 15))
        
        # 1. Model comparison
        plt.subplot(3, 4, 1)
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['metrics']['r2'] for name in model_names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        
        bars = plt.bar(range(len(model_names)), r2_scores, color=colors)
        plt.title('Model Comparison (R² Score)', fontsize=14, fontweight='bold')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylabel('R² Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. RMSE comparison
        plt.subplot(3, 4, 2)
        rmse_scores = [self.models[name]['metrics']['rmse'] for name in model_names]
        plt.bar(range(len(model_names)), rmse_scores, color=colors)
        plt.title('RMSE Comparison', fontsize=14, fontweight='bold')
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylabel('RMSE')
        
        # 3. Predictions vs Actual for best model
        best_model_name = max(model_names, key=lambda x: self.models[x]['metrics']['r2'])
        plt.subplot(3, 4, 3)
        best_predictions = self.models[best_model_name]['predictions']
        plt.scatter(self.y_test, best_predictions, alpha=0.6, color='blue')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{best_model_name} - Predictions vs Actual', fontsize=14, fontweight='bold')
        
        # 4. Residuals plot
        plt.subplot(3, 4, 4)
        residuals = self.y_test - best_predictions
        plt.scatter(best_predictions, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title(f'{best_model_name} - Residuals', fontsize=14, fontweight='bold')
        
        # 5. Feature importance (if available)
        if 'feature_importance' in self.models[best_model_name]:
            plt.subplot(3, 4, 5)
            importance = self.models[best_model_name]['feature_importance']
            indices = np.argsort(importance)[::-1]
            plt.bar(range(len(importance)), importance[indices])
            plt.title(f'{best_model_name} - Feature Importance', fontsize=14, fontweight='bold')
            plt.xticks(range(len(importance)), [self.feature_columns[i] for i in indices], rotation=45, ha='right')
        
        # 6. Training history for neural networks
        if 'history' in self.models.get('Deep Neural Network', {}):
            plt.subplot(3, 4, 6)
            history = self.models['Deep Neural Network']['history']
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Neural Network Training History', fontsize=14, fontweight='bold')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
        
        # 7. Error distribution
        plt.subplot(3, 4, 7)
        errors = np.abs(self.y_test - best_predictions)
        plt.hist(errors, bins=30, alpha=0.7, color='orange')
        plt.title('Error Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        
        # 8. Model complexity vs performance
        plt.subplot(3, 4, 8)
        complexity_scores = [1, 3, 4, 5, 6, 2]  # Rough complexity ranking
        performance_scores = r2_scores
        plt.scatter(complexity_scores, performance_scores, s=100, alpha=0.7, color=colors)
        for i, name in enumerate(model_names):
            plt.annotate(name, (complexity_scores[i], performance_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Model Complexity')
        plt.ylabel('R² Score')
        plt.title('Complexity vs Performance', fontsize=14, fontweight='bold')
        
        # 9. Ensemble weights (if hybrid model exists)
        if 'Hybrid Ensemble' in self.models and 'weights' in self.models['Hybrid Ensemble']:
            plt.subplot(3, 4, 9)
            weights = self.models['Hybrid Ensemble']['weights']
            plt.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%', startangle=90)
            plt.title('Ensemble Model Weights', fontsize=14, fontweight='bold')
        
        # 10. Price range analysis
        plt.subplot(3, 4, 10)
        price_ranges = pd.cut(self.y_test, bins=5)
        range_errors = []
        range_labels = []
        for range_val in price_ranges.cat.categories:
            mask = price_ranges == range_val
            if mask.sum() > 0:
                range_error = np.mean(np.abs(self.y_test[mask] - best_predictions[mask]))
                range_errors.append(range_error)
                range_labels.append(f'{range_val.left:.0f}-{range_val.right:.0f}')
        
        plt.bar(range(len(range_errors)), range_errors, color='purple', alpha=0.7)
        plt.title('Error by Price Range', fontsize=14, fontweight='bold')
        plt.xticks(range(len(range_labels)), range_labels, rotation=45)
        plt.ylabel('Mean Absolute Error')
        
        plt.tight_layout()
        plt.savefig('advanced_model_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self):
        """Generate detailed report"""
        report = ["# Advanced House Price Prediction Analysis\n"]
        report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        report.append("## Model Performance Summary\n\n")
        report.append("| Model | RMSE | MAE | R² Score | Ranking |\n")
        report.append("|-------|------|-----|----------|----------|\n")
        
        # Sort models by R² score
        sorted_models = sorted(self.models.items(), key=lambda x: x[1]['metrics']['r2'], reverse=True)
        
        for rank, (name, info) in enumerate(sorted_models, 1):
            metrics = info['metrics']
            report.append(f"| {name} | {metrics['rmse']:.2f} | {metrics['mae']:.2f} | {metrics['r2']:.3f} | {rank} |\n")
        
        # Best model analysis
        best_name, best_info = sorted_models[0]
        report.append(f"\n## Best Model: {best_name}\n\n")
        report.append(f"- **R² Score**: {best_info['metrics']['r2']:.3f}\n")
        report.append(f"- **RMSE**: {best_info['metrics']['rmse']:.2f} lakhs\n")
        report.append(f"- **MAE**: {best_info['metrics']['mae']:.2f} lakhs\n\n")
        
        # Model progression analysis
        report.append("## Model Progression Analysis\n\n")
        model_order = ['Linear Regression', 'Random Forest', 'XGBoost', '1D CNN', 'Deep Neural Network', 'Hybrid Ensemble']
        
        for model in model_order:
            if model in self.models:
                r2 = self.models[model]['metrics']['r2']
                report.append(f"- **{model}**: R² = {r2:.3f}\n")
        
        # Feature importance
        if 'feature_importance' in best_info:
            report.append("\n## Feature Importance Analysis\n\n")
            importance = best_info['feature_importance']
            feature_importance = list(zip(self.feature_columns, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for feature, imp in feature_importance:
                report.append(f"- **{feature}**: {imp:.3f}\n")
        
        # Recommendations
        report.append("\n## Recommendations\n\n")
        
        if best_info['metrics']['r2'] > 0.9:
            report.append("- Excellent model performance achieved (R² > 0.9)\n")
        elif best_info['metrics']['r2'] > 0.8:
            report.append("- Good model performance achieved (R² > 0.8)\n")
        else:
            report.append("- Model performance can be improved\n")
        
        report.append("- Consider ensemble methods for production deployment\n")
        report.append("- Regular model retraining recommended as market conditions change\n")
        report.append("- Feature engineering showed significant impact on performance\n")
        
        # Save report
        with open('advanced_model_report.md', 'w') as f:
            f.write(''.join(report))
        
        print("Comprehensive report saved as 'advanced_model_report.md'")

def main():
    """Main execution function"""
    predictor = AdvancedHousePricePredictor()
    
    print("🚀 Advanced House Price Prediction Pipeline")
    print("=" * 60)
    
    # Train all models
    results = predictor.train_all_models()
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    # Display results
    for model_name, metrics in results.items():
        print(f"{model_name:20} | R²: {metrics['r2']:.3f} | RMSE: {metrics['rmse']:.2f}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['r2'])
    print(f"\n🏆 Best Model: {best_model} (R² = {results[best_model]['r2']:.3f})")
    
    # Generate visualizations and report
    predictor.plot_comprehensive_results()
    predictor.generate_comprehensive_report()
    
    print("\n✅ Analysis complete! Check 'advanced_model_analysis.png' and 'advanced_model_report.md'")

if __name__ == "__main__":
    main()