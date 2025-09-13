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

class RealisticModelValidation:
    def __init__(self):
        self.df = None
        self.test_results = {}
        self.feature_columns = []
        
    def create_realistic_data(self):
        """Create data with realistic complexity and noise"""
        np.random.seed(42)
        n_samples = 1500  # Smaller dataset for more realistic scenario
        
        locations = ['Electronic City', 'Whitefield', 'Sarjapur Road', 'Marathahalli', 
                    'BTM Layout', 'Koramangala', 'Indiranagar', 'HSR Layout', 'Hebbal',
                    'JP Nagar', 'Jayanagar', 'Rajajinagar', 'Malleshwaram', 'Yelahanka',
                    'Banashankari', 'Basavanagudi', 'RT Nagar', 'Kammanahalli']
        
        area_types = ['Super built-up Area', 'Plot Area', 'Built-up Area', 'Carpet Area']
        sizes = ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '4+ BHK']
        
        # More realistic distributions
        data = {
            'area_type': np.random.choice(area_types, n_samples),
            'location': np.random.choice(locations, n_samples),
            'size': np.random.choice(sizes, n_samples, p=[0.10, 0.40, 0.30, 0.15, 0.05]),
            'total_sqft': np.random.lognormal(mean=7.0, sigma=0.5, size=n_samples),
            'bath': np.random.poisson(lam=2, size=n_samples) + 1,
            'balcony': np.random.poisson(lam=1, size=n_samples),
        }
        
        # Clip to realistic ranges
        data['total_sqft'] = np.clip(data['total_sqft'], 300, 5000)
        data['bath'] = np.clip(data['bath'], 1, 6)
        data['balcony'] = np.clip(data['balcony'], 0, 4)
        
        # More complex and realistic price generation
        base_prices = {'1 BHK': 30, '2 BHK': 50, '3 BHK': 75, '4 BHK': 120, '4+ BHK': 180}
        
        # Market factors (more variation)
        location_multipliers = {
            'Electronic City': 0.65, 'Whitefield': 1.15, 'Sarjapur Road': 0.80,
            'Marathahalli': 1.05, 'BTM Layout': 0.90, 'Koramangala': 1.60,
            'Indiranagar': 1.70, 'HSR Layout': 1.35, 'Hebbal': 0.85,
            'JP Nagar': 1.00, 'Jayanagar': 1.20, 'Rajajinagar': 1.10,
            'Malleshwaram': 1.25, 'Yelahanka': 0.75, 'Banashankari': 0.95,
            'Basavanagudi': 1.15, 'RT Nagar': 0.85, 'Kammanahalli': 0.95
        }
        
        area_type_multipliers = {
            'Super built-up Area': 1.0, 'Plot Area': 1.3, 
            'Built-up Area': 0.85, 'Carpet Area': 0.75
        }
        
        prices = []
        for i in range(n_samples):
            base_price = base_prices[data['size'][i]]
            loc_mult = location_multipliers[data['location'][i]]
            area_mult = area_type_multipliers[data['area_type'][i]]
            
            # Complex non-linear relationships
            sqft_factor = (data['total_sqft'][i] / 1000) ** 0.7  # Diminishing returns
            bath_factor = 1 + (data['bath'][i] - 2) * 0.08
            balcony_factor = 1 + data['balcony'][i] * 0.04
            
            # Market volatility and external factors (high noise)
            market_noise = np.random.normal(0, base_price * 0.25)  # 25% noise
            economic_factor = np.random.uniform(0.85, 1.15)  # Economic fluctuation
            
            # Age/condition factor (hidden complexity)
            condition_factor = np.random.uniform(0.8, 1.2)
            
            price = (base_price * loc_mult * area_mult * sqft_factor * 
                    bath_factor * balcony_factor * economic_factor * 
                    condition_factor + market_noise)
            
            prices.append(max(20, price))  # Minimum price
            
        data['price'] = prices
        
        # Add derived features
        df = pd.DataFrame(data)
        df['bhk'] = df['size'].str.extract('(\d+)').astype(int)
        df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
        
        # Add some missing values for realism
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, 'balcony'] = np.nan
        
        self.df = df.dropna()  # Clean dataset
        print(f"Realistic data created with shape: {self.df.shape}")
        print(f"Price range: ₹{self.df['price'].min():.1f}L - ₹{self.df['price'].max():.1f}L")
        print(f"Price std deviation: ₹{self.df['price'].std():.1f}L")
        return self.df
    
    def prepare_and_evaluate_models(self):
        """Prepare data and evaluate models with realistic expectations"""
        # Encode categorical variables
        le_location = LabelEncoder()
        le_area_type = LabelEncoder()
        
        self.df['location_encoded'] = le_location.fit_transform(self.df['location'])
        self.df['area_type_encoded'] = le_area_type.fit_transform(self.df['area_type'])
        
        # Simple feature set to avoid overfitting
        feature_cols = ['total_sqft', 'bath', 'balcony', 'bhk', 'location_encoded', 'area_type_encoded']
        
        X = self.df[feature_cols]
        y = self.df['price']
        
        # Larger test set for more reliable evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # More conservative models
        models = {
            'Ridge Regression': Ridge(alpha=50.0),  # Higher regularization
            'Lasso Regression': Lasso(alpha=5.0),
            'ElasticNet': ElasticNet(alpha=5.0, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(
                n_estimators=30,      # Fewer trees
                max_depth=8,          # Shallower trees
                min_samples_split=20, # More conservative
                min_samples_leaf=10,
                max_features='sqrt',  # Feature subsampling
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=30,      # Fewer iterations
                max_depth=4,          # Shallow trees
                learning_rate=0.05,   # Slower learning
                subsample=0.7,        # More subsampling
                max_features='sqrt',
                random_state=42
            )
        }
        
        # Evaluate with cross-validation
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            
            # Cross-validation on training set
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
            
            if name in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
                model.fit(X_train_scaled, y_train)
                y_test_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
            
            # Test set metrics
            test_mse = mean_squared_error(y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Calculate MAPE for percentage error
            mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
            
            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_mape': mape,
                'overfitting_gap': cv_scores.mean() - test_r2,
                'y_test': y_test,
                'y_test_pred': y_test_pred
            }
            
            print(f"  CV R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"  Test R²: {test_r2:.3f}")
            print(f"  Test RMSE: ₹{test_rmse:.1f}L")
            print(f"  Test MAPE: {mape:.1f}%")
            print(f"  Gap: {cv_scores.mean() - test_r2:.3f}")
            print()
        
        self.test_results = results
        self.feature_columns = feature_cols
        return results
    
    def create_realistic_evaluation_dashboard(self):
        """Create evaluation dashboard with realistic performance expectations"""
        if not self.test_results:
            self.prepare_and_evaluate_models()
        
        fig = plt.figure(figsize=(20, 16))
        
        model_names = list(self.test_results.keys())
        test_r2_scores = [self.test_results[name]['test_r2'] for name in model_names]
        test_rmse_scores = [self.test_results[name]['test_rmse'] for name in model_names]
        test_mape_scores = [self.test_results[name]['test_mape'] for name in model_names]
        overfitting_gaps = [self.test_results[name]['overfitting_gap'] for name in model_names]
        
        # 1. Realistic Performance Metrics
        ax1 = plt.subplot(3, 3, 1)
        x = np.arange(len(model_names))
        width = 0.3
        
        bars1 = ax1.bar(x - width, test_r2_scores, width, label='Test R²', alpha=0.8, color='darkblue')
        ax1.set_ylabel('R² Score', fontweight='bold', color='darkblue')
        ax1.set_title('🎯 Realistic Test Performance\n(R² scores in practical range)', fontweight='bold')
        ax1.set_ylim(0, 1)
        
        # Add realistic performance bands
        ax1.axhspan(0.7, 0.8, alpha=0.3, color='orange', label='Good (70-80%)')
        ax1.axhspan(0.8, 0.9, alpha=0.3, color='lightgreen', label='Very Good (80-90%)')
        ax1.axhspan(0.9, 1.0, alpha=0.3, color='red', label='Suspicious (>90%)')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Error Metrics in Real Units
        ax2 = plt.subplot(3, 3, 2)
        ax2_twin = ax2.twinx()
        
        bars2 = ax2.bar(x - width/2, test_rmse_scores, width, label='RMSE (lakhs)', alpha=0.8, color='red')
        bars3 = ax2_twin.bar(x + width/2, test_mape_scores, width, label='MAPE (%)', alpha=0.8, color='orange')
        
        ax2.set_ylabel('RMSE (lakhs)', fontweight='bold', color='red')
        ax2_twin.set_ylabel('MAPE (%)', fontweight='bold', color='orange')
        ax2.set_title('📊 Real-World Error Metrics', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        # 3. Overfitting Detection
        ax3 = plt.subplot(3, 3, 3)
        gap_colors = ['red' if gap > 0.05 else 'orange' if gap > 0.02 else 'green' for gap in overfitting_gaps]
        bars = ax3.bar(model_names, overfitting_gaps, color=gap_colors, alpha=0.7)
        ax3.axhline(y=0.05, color='red', linestyle='--', label='Overfitting threshold')
        ax3.axhline(y=0.02, color='orange', linestyle='--', label='Caution threshold')
        ax3.set_title('🚨 Overfitting Detection', fontweight='bold')
        ax3.set_ylabel('CV R² - Test R²', fontweight='bold')
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.legend()
        
        # 4. Best model analysis
        best_model_name = max(model_names, key=lambda x: self.test_results[x]['test_r2'])
        best_results = self.test_results[best_model_name]
        
        ax4 = plt.subplot(3, 3, 4)
        y_test = best_results['y_test']
        y_pred = best_results['y_test_pred']
        
        scatter = ax4.scatter(y_test, y_pred, alpha=0.6, color='purple')
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        ax4.set_xlabel('Actual Price (lakhs)')
        ax4.set_ylabel('Predicted Price (lakhs)')
        ax4.set_title(f'🎯 {best_model_name}\nTest Predictions vs Actual')
        
        # 5. Residual analysis
        ax5 = plt.subplot(3, 3, 5)
        residuals = y_test - y_pred
        ax5.scatter(y_pred, residuals, alpha=0.6, color='orange')
        ax5.axhline(y=0, color='red', linestyle='--')
        ax5.set_xlabel('Predicted Price (lakhs)')
        ax5.set_ylabel('Residuals (lakhs)')
        ax5.set_title('📈 Residual Analysis')
        
        # 6. Error distribution
        ax6 = plt.subplot(3, 3, 6)
        ax6.hist(residuals, bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
        ax6.axvline(0, color='red', linestyle='--')
        ax6.set_xlabel('Residuals (lakhs)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('📊 Error Distribution')
        
        # 7. Performance summary
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')
        
        table_data = []
        for name in model_names:
            table_data.append([
                name,
                f"{self.test_results[name]['test_r2']:.3f}",
                f"{self.test_results[name]['test_rmse']:.1f}",
                f"{self.test_results[name]['test_mape']:.1f}%"
            ])
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Model', 'Test R²', 'RMSE (L)', 'MAPE'],
                         cellLoc='center',
                         loc='center')
        table.set_fontsize(10)
        table.scale(1, 2)
        ax7.set_title('📋 Performance Summary', fontweight='bold', pad=20)
        
        # 8. Realistic expectations text
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        
        best_r2 = max(test_r2_scores)
        best_mape = self.test_results[best_model_name]['test_mape']
        
        if best_r2 > 0.9:
            status = "⚠️ SUSPICIOUSLY HIGH"
            color = 'lightcoral'
            advice = "Consider adding more noise\nor complexity to data"
        elif best_r2 > 0.8:
            status = "✅ VERY GOOD"
            color = 'lightgreen'
            advice = "Excellent performance\nfor real-world scenario"
        elif best_r2 > 0.7:
            status = "✅ GOOD"
            color = 'lightblue'
            advice = "Acceptable performance\nfor production use"
        else:
            status = "⚠️ NEEDS IMPROVEMENT"
            color = 'lightyellow'
            advice = "Consider feature engineering\nor different algorithms"
        
        summary_text = f"""
🏆 BEST MODEL: {best_model_name}

📊 Performance:
   R² Score: {best_r2:.3f}
   MAPE: {best_mape:.1f}%

📈 Status: {status}

💡 {advice}

🎯 Realistic Range:
   R² 0.70-0.85 = Good
   R² 0.85+ = Excellent
   MAPE <15% = Good
        """
        
        ax8.text(0.05, 0.95, summary_text, fontsize=10, fontweight='bold',
                verticalalignment='top', transform=ax8.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.8))
        
        # 9. Reality check
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        reality_text = """
🔍 REALITY CHECK:

Real Estate ML Performance:
• R² 0.70-0.85: Industry Standard
• R² 0.85-0.90: Very Good
• R² >0.90: Potentially Overfitted

⚠️ Red Flags:
• R² >0.95: Almost certainly overfitted
• Very low RMSE with complex data
• Perfect separation in residuals

✅ Good Signs:
• Reasonable error rates
• Random residual patterns
• CV-Test gap <0.05
        """
        
        ax9.text(0.05, 0.95, reality_text, fontsize=9, fontweight='bold',
                verticalalignment='top', transform=ax9.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('realistic_model_evaluation.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: realistic_model_evaluation.png")
        
        # Print realistic summary
        print("\n" + "="*80)
        print("🎯 REALISTIC MODEL EVALUATION SUMMARY")
        print("="*80)
        
        for name in model_names:
            r2 = self.test_results[name]['test_r2']
            mape = self.test_results[name]['test_mape']
            gap = self.test_results[name]['overfitting_gap']
            
            if r2 > 0.9:
                assessment = "⚠️  SUSPICIOUSLY HIGH - Check for overfitting"
            elif r2 > 0.8:
                assessment = "✅ EXCELLENT - Industry leading"
            elif r2 > 0.7:
                assessment = "✅ GOOD - Production ready"
            else:
                assessment = "⚠️  NEEDS WORK - Below industry standard"
            
            print(f"{name}:")
            print(f"   Test R²: {r2:.3f} | MAPE: {mape:.1f}% | Gap: {gap:.3f}")
            print(f"   Assessment: {assessment}")
            print()
        
        print(f"🏆 RECOMMENDED MODEL: {best_model_name}")
        print(f"📊 Expected accuracy range for real estate: 70-85% R²")
        print(f"📈 Your best model: {max(test_r2_scores):.1%} R²")
        
        if max(test_r2_scores) > 0.9:
            print("⚠️  WARNING: Accuracy seems too high for real estate prediction!")
            print("💡 Consider: More noise, missing values, or external factors")
        
        print("="*80)

def main():
    """Main function for realistic model evaluation"""
    print("🎯 Creating Realistic Model Evaluation...")
    print("🔍 Focusing on practical, industry-standard performance")
    
    validator = RealisticModelValidation()
    
    # Create realistic data with proper complexity
    validator.create_realistic_data()
    
    # Evaluate models
    validator.create_realistic_evaluation_dashboard()
    
    print("\n✅ REALISTIC EVALUATION COMPLETED!")
    print("\n📁 Generated File:")
    print("   🎯 realistic_model_evaluation.png - Industry-standard evaluation")
    
    print("\n🎯 KEY INSIGHTS:")
    print("   📊 Real estate ML typically achieves 70-85% R²")
    print("   ⚠️  Scores >90% often indicate overfitting")
    print("   💡 MAPE <15% is considered good for price prediction")

if __name__ == "__main__":
    main()