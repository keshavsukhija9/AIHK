import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class TestBasedComparisonSuite:
    def __init__(self):
        self.df = None
        self.test_results = {}
        self.cv_results = {}
        self.feature_columns = []
        self.X_test = None
        self.y_test = None
        
    def create_sample_data(self):
        """Create comprehensive sample data"""
        np.random.seed(42)
        n_samples = 2000
        
        locations = ['Electronic City', 'Whitefield', 'Sarjapur Road', 'Marathahalli', 
                    'BTM Layout', 'Koramangala', 'Indiranagar', 'HSR Layout', 'Hebbal',
                    'JP Nagar', 'Jayanagar', 'Rajajinagar', 'Malleshwaram', 'Yelahanka']
        area_types = ['Super built-up Area', 'Plot Area', 'Built-up Area', 'Carpet Area']
        sizes = ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '4+ BHK']
        
        data = {
            'area_type': np.random.choice(area_types, n_samples),
            'availability': ['Ready To Move'] * n_samples,
            'location': np.random.choice(locations, n_samples),
            'size': np.random.choice(sizes, n_samples, p=[0.15, 0.35, 0.3, 0.15, 0.05]),
            'total_sqft': np.random.lognormal(mean=7.2, sigma=0.4, size=n_samples),
            'bath': np.random.poisson(lam=2, size=n_samples) + 1,
            'balcony': np.random.poisson(lam=1, size=n_samples),
        }
        
        # Clip values
        data['total_sqft'] = np.clip(data['total_sqft'], 400, 4000)
        data['bath'] = np.clip(data['bath'], 1, 5)
        data['balcony'] = np.clip(data['balcony'], 0, 4)
        
        # Generate realistic prices
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
            sqft_factor = (data['total_sqft'][i] / 1000) ** 0.8
            bath_factor = 1 + (data['bath'][i] - 2) * 0.05
            balcony_factor = 1 + data['balcony'][i] * 0.03
            noise = np.random.normal(0, base_price * 0.15)
            price = base_price * loc_mult * sqft_factor * bath_factor * balcony_factor + noise
            prices.append(max(15, price))
            
        data['price'] = prices
        
        # Add derived features
        df = pd.DataFrame(data)
        df['bhk'] = df['size'].str.extract('(\d+)').astype(int)
        df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
        df['room_to_bath_ratio'] = df['bhk'] / df['bath']
        
        self.df = df
        print(f"Sample data created with shape: {df.shape}")
        return df
    
    def prepare_data_and_train_models(self):
        """Prepare data and train models with proper test evaluation"""
        # Prepare data
        le_location = LabelEncoder()
        le_area_type = LabelEncoder()
        
        self.df['location_encoded'] = le_location.fit_transform(self.df['location'])
        self.df['area_type_encoded'] = le_area_type.fit_transform(self.df['area_type'])
        
        feature_cols = ['total_sqft', 'bath', 'balcony', 'bhk', 'location_encoded', 
                       'area_type_encoded', 'price_per_sqft', 'room_to_bath_ratio']
        
        X = self.df[feature_cols]
        y = self.df['price']
        
        # Split data - store test set for all evaluations
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        # Store test data for consistent evaluation
        self.X_test = X_test
        self.y_test = y_test
        self.feature_columns = feature_cols
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models with regularization
        models = {
            'Ridge Regression': Ridge(alpha=10.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(
                n_estimators=50, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=50, max_depth=6, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
        }
        
        # Train models and evaluate ONLY on test set
        test_results = {}
        cv_results = {}
        
        # 5-fold CV for overfitting detection
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"Training and evaluating {name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
            
            if name in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
                # Train on scaled data
                model.fit(X_train_scaled, y_train)
                # Predict on TEST SET ONLY
                y_test_pred = model.predict(X_test_scaled)
            else:
                # Train on original data
                model.fit(X_train, y_train)
                # Predict on TEST SET ONLY
                y_test_pred = model.predict(X_test)
            
            # Calculate TEST SET metrics only
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            test_results[name] = {
                'model': model,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'y_test_pred': y_test_pred
            }
            
            cv_results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'overfitting_gap': cv_scores.mean() - test_r2
            }
            
            print(f"  CV R² Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"  TEST R² Score: {test_r2:.3f}")
            print(f"  TEST RMSE: {test_rmse:.2f}")
            print(f"  Overfitting Gap: {cv_scores.mean() - test_r2:.3f}")
        
        self.test_results = test_results
        self.cv_results = cv_results
        
        return test_results, cv_results
    
    def create_test_based_performance_dashboard(self):
        """Create dashboard based ONLY on test set performance"""
        if not self.test_results:
            print("Training models first...")
            self.prepare_data_and_train_models()
        
        fig = plt.figure(figsize=(24, 20))
        
        model_names = list(self.test_results.keys())
        test_r2_scores = [self.test_results[name]['test_r2'] for name in model_names]
        test_rmse_scores = [self.test_results[name]['test_rmse'] for name in model_names]
        test_mae_scores = [self.test_results[name]['test_mae'] for name in model_names]
        cv_means = [self.cv_results[name]['cv_mean'] for name in model_names]
        overfitting_gaps = [self.cv_results[name]['overfitting_gap'] for name in model_names]
        
        # 1. TEST SET Performance Metrics Comparison
        ax1 = plt.subplot(3, 4, (1, 2))
        x = np.arange(len(model_names))
        width = 0.25
        
        bars1 = ax1.bar(x - width, test_r2_scores, width, label='Test R² Score', alpha=0.8, color='darkblue')
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x, test_rmse_scores, width, label='Test RMSE', alpha=0.8, color='darkred')
        bars3 = ax1_twin.bar(x + width, test_mae_scores, width, label='Test MAE', alpha=0.8, color='darkgreen')
        
        ax1.set_xlabel('Models', fontweight='bold')
        ax1.set_ylabel('Test R² Score', fontweight='bold', color='darkblue')
        ax1_twin.set_ylabel('Test Error Metrics', fontweight='bold', color='darkred')
        ax1.set_title('🧪 TEST SET Performance Comparison\n(All metrics based on unseen test data)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # 2. TEST SET R² Ranking
        ax2 = plt.subplot(3, 4, (3, 4))
        sorted_indices = np.argsort(test_r2_scores)[::-1]
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_test_r2 = [test_r2_scores[i] for i in sorted_indices]
        
        colors = ['gold', 'silver', '#CD7F32', 'lightcoral', 'lightgray']
        bars = ax2.barh(sorted_names, sorted_test_r2, color=colors[:len(sorted_names)])
        ax2.set_xlabel('Test R² Score', fontweight='bold')
        ax2.set_title('🏆 Model Ranking (Test R² Score)\nBased on unseen test data', 
                     fontsize=14, fontweight='bold')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2., 
                    f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        # 3. Best model TEST predictions vs actual
        best_model_name = max(model_names, key=lambda x: self.test_results[x]['test_r2'])
        best_results = self.test_results[best_model_name]
        
        ax3 = plt.subplot(3, 4, 5)
        scatter = ax3.scatter(self.y_test, best_results['y_test_pred'], alpha=0.6, color='purple')
        min_val = min(self.y_test.min(), best_results['y_test_pred'].min())
        max_val = max(self.y_test.max(), best_results['y_test_pred'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax3.set_xlabel('Actual Price (lakhs)', fontweight='bold')
        ax3.set_ylabel('Predicted Price (lakhs)', fontweight='bold')
        ax3.set_title(f'🎯 {best_model_name}\nTest Set: Predictions vs Actual', fontsize=12, fontweight='bold')
        ax3.legend()
        
        # 4. TEST residuals plot
        ax4 = plt.subplot(3, 4, 6)
        test_residuals = self.y_test - best_results['y_test_pred']
        ax4.scatter(best_results['y_test_pred'], test_residuals, alpha=0.6, color='orange')
        ax4.axhline(y=0, color='red', linestyle='--')
        ax4.set_xlabel('Predicted Price (lakhs)', fontweight='bold')
        ax4.set_ylabel('Test Residuals', fontweight='bold')
        ax4.set_title('📊 Test Set Residual Analysis', fontsize=12, fontweight='bold')
        
        # 5. Overfitting Detection Chart
        ax5 = plt.subplot(3, 4, 7)
        gap_colors = ['red' if gap > 0.05 else 'orange' if gap > 0.02 else 'green' for gap in overfitting_gaps]
        bars = ax5.bar(model_names, overfitting_gaps, color=gap_colors, alpha=0.7)
        ax5.axhline(y=0.05, color='red', linestyle='--', label='Overfitting threshold (0.05)')
        ax5.axhline(y=0.02, color='orange', linestyle='--', label='Caution threshold (0.02)')
        ax5.set_title('🚨 Overfitting Detection\n(CV R² - Test R²)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Performance Gap', fontweight='bold')
        ax5.set_xticklabels(model_names, rotation=45, ha='right')
        ax5.legend()
        
        # Add status labels
        for i, (bar, gap) in enumerate(zip(bars, overfitting_gaps)):
            status = "OVERFITTED" if gap > 0.05 else "CAUTION" if gap > 0.02 else "GOOD"
            ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    status, ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # 6. TEST vs CV R² comparison
        ax6 = plt.subplot(3, 4, 8)
        x = np.arange(len(model_names))
        width = 0.35
        bars1 = ax6.bar(x - width/2, cv_means, width, label='CV R²', alpha=0.8, color='lightblue')
        bars2 = ax6.bar(x + width/2, test_r2_scores, width, label='Test R²', alpha=0.8, color='darkblue')
        
        ax6.set_xlabel('Models', fontweight='bold')
        ax6.set_ylabel('R² Score', fontweight='bold')
        ax6.set_title('📊 CV vs Test R² Comparison', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(model_names, rotation=45, ha='right')
        ax6.legend()
        
        # 7. TEST error distribution
        ax7 = plt.subplot(3, 4, 9)
        ax7.hist(test_residuals, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax7.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
        ax7.set_xlabel('Test Residuals', fontweight='bold')
        ax7.set_ylabel('Frequency', fontweight='bold')
        ax7.set_title('📈 Test Error Distribution', fontsize=12, fontweight='bold')
        ax7.legend()
        
        # 8. TEST performance summary table
        ax8 = plt.subplot(3, 4, 10)
        ax8.axis('off')
        
        table_data = []
        for name in model_names:
            table_data.append([
                name,
                f"{self.test_results[name]['test_r2']:.3f}",
                f"{self.test_results[name]['test_rmse']:.2f}",
                f"{self.test_results[name]['test_mae']:.2f}",
                f"{self.cv_results[name]['overfitting_gap']:.3f}"
            ])
        
        table = ax8.table(cellText=table_data,
                         colLabels=['Model', 'Test R²', 'Test RMSE', 'Test MAE', 'Overfitting\nGap'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax8.set_title('📋 Test Performance Summary', fontsize=12, fontweight='bold', pad=20)
        
        # 9. Feature importance (if available) for best model
        ax9 = plt.subplot(3, 4, 11)
        if hasattr(self.test_results[best_model_name]['model'], 'feature_importances_'):
            importances = self.test_results[best_model_name]['model'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            ax9.bar(range(len(importances)), importances[indices])
            ax9.set_title(f'🔍 Feature Importance\n{best_model_name} (Best Test Performance)', 
                         fontsize=12, fontweight='bold')
            ax9.set_xticks(range(len(importances)))
            ax9.set_xticklabels([self.feature_columns[i] for i in indices], rotation=90)
        else:
            ax9.text(0.5, 0.5, 'Feature importance\nnot available for\nlinear models', 
                    ha='center', va='center', transform=ax9.transAxes, fontsize=12)
            ax9.set_title('🔍 Feature Importance', fontsize=12, fontweight='bold')
        
        # 10. Final recommendation based on TEST performance
        ax10 = plt.subplot(3, 4, 12)
        ax10.axis('off')
        
        best_test_r2 = max(test_r2_scores)
        best_model_idx = test_r2_scores.index(best_test_r2)
        best_gap = overfitting_gaps[best_model_idx]
        
        # Determine generalization status
        if best_gap > 0.05:
            status = "⚠️ OVERFITTED"
            color = 'lightcoral'
        elif best_gap > 0.02:
            status = "⚡ SLIGHTLY OVERFITTED"
            color = 'lightyellow'
        else:
            status = "✅ WELL GENERALIZED"
            color = 'lightgreen'
        
        recommendation_text = f"""
🏆 WINNER (Test Performance):
{model_names[best_model_idx]}

📊 TEST SET Performance:
   R² Score: {best_test_r2:.3f}
   RMSE: {test_rmse_scores[best_model_idx]:.2f}
   MAE: {test_mae_scores[best_model_idx]:.2f}

🎯 Generalization Status:
   CV-Test Gap: {best_gap:.3f}
   {status}

💡 All metrics based on
   unseen test data only!
        """
        
        ax10.text(0.1, 0.9, recommendation_text, fontsize=11, fontweight='bold',
                 verticalalignment='top', transform=ax10.transAxes,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('test_based_performance_dashboard.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: test_based_performance_dashboard.png")
        
        # Print detailed TEST-based summary
        print("\n" + "="*80)
        print("🧪 TEST SET PERFORMANCE ANALYSIS (Unseen Data Only)")
        print("="*80)
        for i, name in enumerate(model_names):
            gap_status = "OVERFITTED" if overfitting_gaps[i] > 0.05 else "CAUTION" if overfitting_gaps[i] > 0.02 else "GOOD"
            print(f"{i+1}. {name}")
            print(f"   Test R² Score: {test_r2_scores[i]:.3f}")
            print(f"   Test RMSE: {test_rmse_scores[i]:.2f}")
            print(f"   Test MAE: {test_mae_scores[i]:.2f}")
            print(f"   CV-Test Gap: {overfitting_gaps[i]:.3f} ({gap_status})")
            print()
        
        best_test_model = model_names[test_r2_scores.index(max(test_r2_scores))]
        print(f"🏆 BEST TEST PERFORMANCE: {best_test_model} with {max(test_r2_scores):.3f} R² score")
        print("📊 All evaluations based exclusively on unseen test data")
        print("="*80)

def main():
    """Main function for test-based evaluation"""
    print("🧪 Starting Test-Based Performance Analysis...")
    print("📊 All comparisons based EXCLUSIVELY on test set performance")
    
    suite = TestBasedComparisonSuite()
    
    # Create data and train models
    suite.create_sample_data()
    
    # Generate test-based performance dashboard
    suite.create_test_based_performance_dashboard()
    
    print("\n🎉 TEST-BASED ANALYSIS COMPLETED!")
    print("\n📁 Generated File:")
    print("   🧪 test_based_performance_dashboard.png - Complete test set evaluation")
    
    print("\n✅ KEY INSIGHTS:")
    print("   📊 All performance metrics calculated on unseen test data")
    print("   🎯 Overfitting detection through CV vs Test comparison") 
    print("   🏆 Model ranking based on true generalization performance")

if __name__ == "__main__":
    main()