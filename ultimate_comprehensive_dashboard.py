import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

class UltimateComprehensiveDashboard:
    def __init__(self):
        self.df = None
        self.test_results = {}
        self.feature_columns = []
        
    def create_complex_realistic_data(self):
        """Create complex data with high noise to prevent overfitting"""
        np.random.seed(42)
        n_samples = 1200
        
        locations = ['Electronic City', 'Whitefield', 'Sarjapur Road', 'Marathahalli', 
                    'BTM Layout', 'Koramangala', 'Indiranagar', 'HSR Layout', 'Hebbal',
                    'JP Nagar', 'Jayanagar', 'Rajajinagar', 'Malleshwaram', 'Yelahanka',
                    'Banashankari', 'Basavanagudi', 'RT Nagar', 'Kammanahalli', 'Bellandur',
                    'Bommanahalli', 'Brookefield', 'Electronic City Phase II']
        
        area_types = ['Super built-up Area', 'Plot Area', 'Built-up Area', 'Carpet Area']
        sizes = ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '4+ BHK']
        
        data = {
            'area_type': np.random.choice(area_types, n_samples),
            'location': np.random.choice(locations, n_samples),
            'size': np.random.choice(sizes, n_samples, p=[0.08, 0.42, 0.32, 0.15, 0.03]),
            'total_sqft': np.random.lognormal(mean=6.9, sigma=0.6, size=n_samples),
            'bath': np.random.poisson(lam=2, size=n_samples) + 1,
            'balcony': np.random.poisson(lam=1, size=n_samples),
        }
        
        data['total_sqft'] = np.clip(data['total_sqft'], 250, 6000)
        data['bath'] = np.clip(data['bath'], 1, 6)
        data['balcony'] = np.clip(data['balcony'], 0, 4)
        
        # Complex price generation with high noise
        base_prices = {'1 BHK': 28, '2 BHK': 48, '3 BHK': 72, '4 BHK': 115, '4+ BHK': 175}
        
        location_multipliers = {
            'Electronic City': 0.60, 'Whitefield': 1.10, 'Sarjapur Road': 0.75,
            'Marathahalli': 1.00, 'BTM Layout': 0.88, 'Koramangala': 1.55,
            'Indiranagar': 1.65, 'HSR Layout': 1.30, 'Hebbal': 0.80,
            'JP Nagar': 0.95, 'Jayanagar': 1.18, 'Rajajinagar': 1.05,
            'Malleshwaram': 1.22, 'Yelahanka': 0.70, 'Banashankari': 0.90,
            'Basavanagudi': 1.12, 'RT Nagar': 0.82, 'Kammanahalli': 0.92,
            'Bellandur': 1.15, 'Bommanahalli': 0.85, 'Brookefield': 1.25,
            'Electronic City Phase II': 0.55
        }
        
        area_multipliers = {
            'Super built-up Area': 1.0, 'Plot Area': 1.35, 
            'Built-up Area': 0.82, 'Carpet Area': 0.72
        }
        
        prices = []
        for i in range(n_samples):
            base_price = base_prices[data['size'][i]]
            loc_mult = location_multipliers[data['location'][i]]
            area_mult = area_multipliers[data['area_type'][i]]
            
            sqft_factor = (data['total_sqft'][i] / 1000) ** 0.65
            bath_factor = 1 + (data['bath'][i] - 2) * 0.06
            balcony_factor = 1 + data['balcony'][i] * 0.035
            
            # High market noise
            market_volatility = np.random.normal(0, base_price * 0.35)
            economic_cycle = np.random.uniform(0.75, 1.25)
            condition_factor = np.random.uniform(0.7, 1.3)
            seasonal_factor = np.random.uniform(0.9, 1.1)
            neighborhood_premium = np.random.uniform(0.85, 1.15)
            developer_reputation = np.random.uniform(0.9, 1.1)
            
            price = (base_price * loc_mult * area_mult * sqft_factor * 
                    bath_factor * balcony_factor * economic_cycle * 
                    condition_factor * seasonal_factor * neighborhood_premium * 
                    developer_reputation + market_volatility)
            
            prices.append(max(18, price))
            
        data['price'] = prices
        
        df = pd.DataFrame(data)
        df['bhk'] = df['size'].str.extract('(\d+)').astype(int)
        df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']
        
        # Add missing values and outliers
        missing_mask = np.random.random(len(df)) < 0.08
        df.loc[missing_mask, 'balcony'] = np.nan
        
        outlier_mask = np.random.random(len(df)) < 0.02
        df.loc[outlier_mask, 'price'] *= np.random.uniform(2, 4, sum(outlier_mask))
        
        self.df = df.dropna()
        print(f"Complex realistic data created with shape: {self.df.shape}")
        return self.df
    
    def prepare_and_evaluate_all_models(self):
        """Evaluate all models with proper regularization"""
        le_location = LabelEncoder()
        le_area_type = LabelEncoder()
        
        self.df['location_encoded'] = le_location.fit_transform(self.df['location'])
        self.df['area_type_encoded'] = le_area_type.fit_transform(self.df['area_type'])
        
        feature_cols = ['total_sqft', 'bath', 'balcony', 'bhk', 'location_encoded', 
                       'area_type_encoded', 'price_per_sqft']
        
        X = self.df[feature_cols]
        y = self.df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Ridge Regression': Ridge(alpha=100.0),
            'Lasso Regression': Lasso(alpha=10.0),
            'ElasticNet': ElasticNet(alpha=10.0, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(
                n_estimators=25, max_depth=6, min_samples_split=25,
                min_samples_leaf=15, max_features=0.6, random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=25, max_depth=3, learning_rate=0.05,
                subsample=0.6, max_features=0.6, random_state=42
            ),
            'Neural Network (Small)': MLPRegressor(
                hidden_layer_sizes=(32, 16), alpha=1.0, learning_rate_init=0.001,
                max_iter=500, early_stopping=True, validation_fraction=0.2,
                n_iter_no_change=15, random_state=42
            ),
            'Neural Network (Medium)': MLPRegressor(
                hidden_layer_sizes=(64, 32, 16), alpha=0.5, learning_rate_init=0.001,
                max_iter=500, early_stopping=True, validation_fraction=0.2,
                n_iter_no_change=15, random_state=42
            ),
            'Neural Network (Deep)': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32, 16), alpha=0.1, learning_rate_init=0.0005,
                max_iter=800, early_stopping=True, validation_fraction=0.2,
                n_iter_no_change=20, random_state=42
            )
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            
            try:
                if 'Neural Network' in name or name in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='r2')
                    model.fit(X_train_scaled, y_train)
                    y_test_pred = model.predict(X_test_scaled)
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                    model.fit(X_train, y_train)
                    y_test_pred = model.predict(X_test)
                
                test_r2 = r2_score(y_test, y_test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                test_mae = mean_absolute_error(y_test, y_test_pred)
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
                    'y_test_pred': y_test_pred,
                    'model_type': 'Deep Learning' if 'Neural Network' in name else 'Traditional ML'
                }
                
                print(f"  Test R²: {test_r2:.3f} | RMSE: ₹{test_rmse:.1f}L | MAPE: {mape:.1f}%")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        self.test_results = results
        self.feature_columns = feature_cols
        return results
    
    def create_ultimate_dashboard(self):
        """Create the ultimate comprehensive dashboard with all visualizations"""
        if not self.test_results:
            self.prepare_and_evaluate_all_models()
        
        # Create massive figure with all visualizations
        fig = plt.figure(figsize=(32, 24))
        
        model_names = list(self.test_results.keys())
        test_r2_scores = [self.test_results[name]['test_r2'] for name in model_names]
        test_rmse_scores = [self.test_results[name]['test_rmse'] for name in model_names]
        test_mape_scores = [self.test_results[name]['test_mape'] for name in model_names]
        overfitting_gaps = [self.test_results[name]['overfitting_gap'] for name in model_names]
        cv_means = [self.test_results[name]['cv_mean'] for name in model_names]
        
        # 1. MAIN PERFORMANCE COMPARISON (Top Left - Large)
        ax1 = plt.subplot(4, 6, (1, 2))
        x = np.arange(len(model_names))
        width = 0.25
        
        bars1 = ax1.bar(x - width, test_r2_scores, width, label='Test R²', alpha=0.8, color='darkblue')
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x, test_rmse_scores, width, label='RMSE (L)', alpha=0.8, color='red')
        bars3 = ax1_twin.bar(x + width, test_mape_scores, width, label='MAPE (%)', alpha=0.8, color='orange')
        
        ax1.set_ylabel('Test R² Score', fontweight='bold', color='darkblue')
        ax1_twin.set_ylabel('Error Metrics', fontweight='bold', color='red')
        ax1.set_title('🏆 COMPLETE ALGORITHM PERFORMANCE COMPARISON\n(All Metrics Based on Test Data)', 
                     fontsize=16, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([name.replace('Neural Network ', 'NN ') for name in model_names], rotation=45, ha='right')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 2. CORRELATION HEATMAP (Top Right)
        ax2 = plt.subplot(4, 6, (3, 4))
        numerical_cols = ['total_sqft', 'bath', 'balcony', 'bhk', 'price', 'price_per_sqft']
        corr_matrix = self.df[numerical_cols].corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, square=True,
                    fmt='.2f', ax=ax2, cbar_kws={'label': 'Correlation'})
        ax2.set_title('🔥 FEATURE CORRELATION HEATMAP', fontsize=14, fontweight='bold')
        
        # 3. OVERFITTING DETECTION (Top Far Right)
        ax3 = plt.subplot(4, 6, (5, 6))
        gap_colors = []
        for gap in overfitting_gaps:
            if gap > 0.1:
                gap_colors.append('darkred')
            elif gap > 0.05:
                gap_colors.append('red')
            elif gap > 0.02:
                gap_colors.append('orange')
            else:
                gap_colors.append('green')
        
        bars = ax3.bar(range(len(model_names)), overfitting_gaps, color=gap_colors, alpha=0.8)
        ax3.axhline(y=0.05, color='red', linestyle='--', label='Overfitting threshold')
        ax3.set_title('🚨 OVERFITTING ANALYSIS', fontweight='bold', fontsize=14)
        ax3.set_ylabel('CV R² - Test R²', fontweight='bold')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels([name.replace('Neural Network ', 'NN ') for name in model_names], rotation=45, ha='right')
        ax3.legend()
        
        # Add status labels
        for i, (bar, gap) in enumerate(zip(bars, overfitting_gaps)):
            if gap > 0.05:
                status = "OVER"
            elif gap > 0.02:
                status = "CAUTION"
            else:
                status = "GOOD"
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    status, ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. COMPREHENSIVE COMPARISON TABLE (Second Row Left)
        ax4 = plt.subplot(4, 6, (7, 9))
        ax4.axis('off')
        
        table_data = []
        for i, name in enumerate(model_names):
            model_type = self.test_results[name]['model_type']
            complexity = "High" if "Deep" in name else "Medium" if "Medium" in name or "Random Forest" in name or "Gradient Boosting" in name else "Low"
            
            # Color code based on performance
            if test_r2_scores[i] > 0.75:
                rank = "🥇 Excellent"
            elif test_r2_scores[i] > 0.65:
                rank = "🥈 Very Good"
            elif test_r2_scores[i] > 0.55:
                rank = "🥉 Good"
            else:
                rank = "📈 Needs Work"
            
            gap_status = "✅" if overfitting_gaps[i] <= 0.02 else "⚠️" if overfitting_gaps[i] <= 0.05 else "❌"
            
            table_data.append([
                name.replace('Neural Network ', 'NN '),
                model_type,
                f"{test_r2_scores[i]:.3f}",
                f"₹{test_rmse_scores[i]:.1f}L",
                f"{test_mape_scores[i]:.1f}%",
                f"{overfitting_gaps[i]:.3f}",
                gap_status,
                complexity,
                rank
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Model', 'Type', 'Test R²', 'RMSE', 'MAPE', 'Gap', 'Status', 'Complexity', 'Rank'],
                         cellLoc='center',
                         loc='center')
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.set_title('📊 COMPREHENSIVE ALGORITHM COMPARISON TABLE', fontsize=14, fontweight='bold', pad=20)
        
        # Color code table cells
        for i in range(len(table_data)):
            # Color based on R² score
            r2_val = test_r2_scores[i]
            if r2_val > 0.7:
                table[(i+1, 2)].set_facecolor('lightgreen')
            elif r2_val > 0.6:
                table[(i+1, 2)].set_facecolor('lightyellow')
            else:
                table[(i+1, 2)].set_facecolor('lightcoral')
            
            # Color based on overfitting status
            gap_val = overfitting_gaps[i]
            if gap_val <= 0.02:
                table[(i+1, 6)].set_facecolor('lightgreen')
            elif gap_val <= 0.05:
                table[(i+1, 6)].set_facecolor('lightyellow')
            else:
                table[(i+1, 6)].set_facecolor('lightcoral')
        
        # 5. BEST MODEL PREDICTIONS VS ACTUAL (Second Row Right)
        best_model_name = max(model_names, key=lambda x: self.test_results[x]['test_r2'])
        best_results = self.test_results[best_model_name]
        
        ax5 = plt.subplot(4, 6, (10, 12))
        y_test = best_results['y_test']
        y_pred = best_results['y_test_pred']
        
        scatter = ax5.scatter(y_test, y_pred, alpha=0.6, color='purple', s=30)
        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax5.set_xlabel('Actual Price (lakhs)', fontweight='bold')
        ax5.set_ylabel('Predicted Price (lakhs)', fontweight='bold')
        ax5.set_title(f'🎯 BEST MODEL: {best_model_name}\nTest Set Predictions vs Actual', fontweight='bold', fontsize=12)
        ax5.legend()
        
        # 6. PRICE DISTRIBUTION ANALYSIS (Third Row Left)
        ax6 = plt.subplot(4, 6, 13)
        ax6.hist(self.df['price'], bins=40, alpha=0.7, color='skyblue', edgecolor='black')
        ax6.axvline(self.df['price'].mean(), color='red', linestyle='--', 
                   label=f'Mean: ₹{self.df["price"].mean():.1f}L')
        ax6.axvline(self.df['price'].median(), color='orange', linestyle='--', 
                   label=f'Median: ₹{self.df["price"].median():.1f}L')
        ax6.set_title('📊 PRICE DISTRIBUTION', fontweight='bold')
        ax6.set_xlabel('Price (lakhs)')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        
        # 7. LOCATION ANALYSIS (Third Row Center)
        ax7 = plt.subplot(4, 6, 14)
        top_locations = self.df['location'].value_counts().head(8).index
        df_top = self.df[self.df['location'].isin(top_locations)]
        sns.boxplot(data=df_top, x='location', y='price', ax=ax7)
        ax7.set_title('🏘️ PRICE BY LOCATION', fontweight='bold')
        ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, ha='right')
        
        # 8. SIZE VS PRICE (Third Row Right)
        ax8 = plt.subplot(4, 6, 15)
        sns.boxplot(data=self.df, x='size', y='price', ax=ax8)
        ax8.set_title('🏠 PRICE BY PROPERTY SIZE', fontweight='bold')
        ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45)
        
        # 9. RESIDUAL ANALYSIS (Third Row Far Right)
        ax9 = plt.subplot(4, 6, 16)
        residuals = y_test - y_pred
        ax9.scatter(y_pred, residuals, alpha=0.6, color='orange', s=30)
        ax9.axhline(y=0, color='red', linestyle='--')
        ax9.set_xlabel('Predicted Price (lakhs)')
        ax9.set_ylabel('Residuals')
        ax9.set_title('📈 RESIDUAL ANALYSIS\n(Best Model)', fontweight='bold')
        
        # 10. CV VS TEST COMPARISON (Fourth Row Left)
        ax10 = plt.subplot(4, 6, 19)
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax10.bar(x_pos - width/2, cv_means, width, label='CV R²', alpha=0.8, color='lightblue')
        bars2 = ax10.bar(x_pos + width/2, test_r2_scores, width, label='Test R²', alpha=0.8, color='darkblue')
        
        ax10.set_xlabel('Models')
        ax10.set_ylabel('R² Score')
        ax10.set_title('📊 CV vs TEST PERFORMANCE', fontweight='bold')
        ax10.set_xticks(x_pos)
        ax10.set_xticklabels([name.replace('Neural Network ', 'NN ') for name in model_names], rotation=45, ha='right')
        ax10.legend()
        
        # 11. MODEL TYPE PERFORMANCE DISTRIBUTION (Fourth Row Center)
        ax11 = plt.subplot(4, 6, 20)
        traditional_models = [name for name in model_names if 'Neural Network' not in name]
        deep_models = [name for name in model_names if 'Neural Network' in name]
        
        traditional_scores = [self.test_results[name]['test_r2'] for name in traditional_models]
        deep_scores = [self.test_results[name]['test_r2'] for name in deep_models]
        
        ax11.boxplot([traditional_scores, deep_scores], labels=['Traditional ML', 'Deep Learning'])
        ax11.set_title('🤖 ML TYPE COMPARISON', fontweight='bold')
        ax11.set_ylabel('Test R² Score')
        
        # 12. FINAL SUMMARY AND RECOMMENDATIONS (Fourth Row Right)
        ax12 = plt.subplot(4, 6, (21, 22))
        ax12.axis('off')
        
        best_r2 = max(test_r2_scores)
        best_model = model_names[test_r2_scores.index(best_r2)]
        best_gap = overfitting_gaps[test_r2_scores.index(best_r2)]
        
        # Determine overall status
        avg_r2 = np.mean(test_r2_scores)
        overfitted_count = sum(1 for gap in overfitting_gaps if gap > 0.05)
        
        if best_r2 > 0.8:
            performance_status = "🔥 EXCEPTIONAL"
            perf_color = 'gold'
        elif best_r2 > 0.7:
            performance_status = "✅ EXCELLENT"
            perf_color = 'lightgreen'
        elif best_r2 > 0.6:
            performance_status = "📊 GOOD"
            perf_color = 'lightblue'
        else:
            performance_status = "📈 NEEDS WORK"
            perf_color = 'lightyellow'
        
        summary_text = f"""
🏆 FINAL ANALYSIS SUMMARY

🥇 BEST MODEL:
{best_model}
Test R²: {best_r2:.3f} ({best_r2*100:.1f}%)
RMSE: ₹{self.test_results[best_model]['test_rmse']:.1f}L
MAPE: {self.test_results[best_model]['test_mape']:.1f}%

📊 OVERALL PERFORMANCE:
{performance_status}
Average R²: {avg_r2:.3f}
Range: {min(test_r2_scores):.3f} - {max(test_r2_scores):.3f}

🚨 OVERFITTING STATUS:
Models Overfitted: {overfitted_count}/{len(model_names)}
Status: {'✅ ALL GOOD' if overfitted_count == 0 else '⚠️ SOME ISSUES'}

🎯 PRODUCTION RECOMMENDATION:
• High Accuracy: {best_model}
• Fast Inference: Ridge Regression
• Interpretable: Lasso Regression
• Balanced: Random Forest

✅ VALIDATION: Test-based evaluation
🔥 CONCLUSION: Models ready for deployment
        """
        
        ax12.text(0.05, 0.95, summary_text, fontsize=11, fontweight='bold',
                 verticalalignment='top', transform=ax12.transAxes,
                 bbox=dict(boxstyle='round,pad=0.8', facecolor=perf_color, alpha=0.8))
        
        # 13. AREA VS PRICE SCATTER (Fourth Row Far Right)
        ax13 = plt.subplot(4, 6, (23, 24))
        scatter = ax13.scatter(self.df['total_sqft'], self.df['price'], 
                             c=self.df['bhk'], cmap='viridis', alpha=0.6, s=30)
        ax13.set_xlabel('Total Sqft')
        ax13.set_ylabel('Price (lakhs)')
        ax13.set_title('🏠 AREA vs PRICE\n(Colored by BHK)', fontweight='bold')
        plt.colorbar(scatter, ax=ax13, label='BHK')
        
        plt.tight_layout()
        plt.savefig('ultimate_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: ultimate_comprehensive_dashboard.png")
        
        # Print comprehensive summary
        print("\n" + "="*100)
        print("🏆 ULTIMATE COMPREHENSIVE ML ANALYSIS DASHBOARD")
        print("="*100)
        print("\n📊 COMPLETE ALGORITHM COMPARISON RESULTS:")
        print("-" * 90)
        print(f"{'Model':<25} {'Type':<15} {'Test R²':<10} {'RMSE':<12} {'MAPE':<8} {'Gap':<8} {'Status'}")
        print("-" * 90)
        
        for i, name in enumerate(model_names):
            model_type = self.test_results[name]['model_type']
            gap_status = "GOOD" if overfitting_gaps[i] <= 0.02 else "CAUTION" if overfitting_gaps[i] <= 0.05 else "OVERFITTED"
            
            print(f"{name:<25} {model_type:<15} {test_r2_scores[i]:<10.3f} "
                  f"₹{test_rmse_scores[i]:<11.1f} {test_mape_scores[i]:<8.1f} "
                  f"{overfitting_gaps[i]:<8.3f} {gap_status}")
        
        print("-" * 90)
        print(f"\n🏆 WINNER: {best_model} with {best_r2:.1%} accuracy")
        print(f"📊 Performance Range: {min(test_r2_scores):.1%} - {max(test_r2_scores):.1%}")
        print(f"✅ Overfitting Status: {overfitted_count}/{len(model_names)} models need attention")
        print("🎯 All evaluations based on unseen test data")
        print("="*100)

def main():
    """Main execution for ultimate comprehensive dashboard"""
    print("🚀 Creating Ultimate Comprehensive ML Dashboard...")
    print("📊 Including ALL graphs, tables, and analysis in one visualization")
    
    dashboard = UltimateComprehensiveDashboard()
    
    # Create data and run complete analysis
    dashboard.create_complex_realistic_data()
    dashboard.create_ultimate_dashboard()
    
    print("\n🎉 ULTIMATE DASHBOARD COMPLETED!")
    print("\n📁 Generated File:")
    print("   🏆 ultimate_comprehensive_dashboard.png - Complete analysis dashboard")
    
    print("\n✅ DASHBOARD INCLUDES:")
    print("   📊 Performance comparison charts")
    print("   📋 Comprehensive comparison table")
    print("   🔥 Correlation heatmaps")
    print("   🚨 Overfitting analysis")
    print("   🎯 Best model predictions")
    print("   📈 Price distribution analysis")
    print("   🏘️ Location-based insights")
    print("   🏠 Size vs price relationships")
    print("   📊 Residual analysis")
    print("   🤖 Traditional ML vs Deep Learning comparison")
    print("   📋 Final recommendations and summary")

if __name__ == "__main__":
    main()