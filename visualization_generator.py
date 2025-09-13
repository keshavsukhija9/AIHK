import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveVisualizer:
    def __init__(self):
        self.df = None
        self.processed_df = None
        
    def load_and_prepare_data(self):
        """Create and prepare sample data for visualization"""
        print("Creating enhanced sample data for visualization...")
        np.random.seed(42)
        n_samples = 2500
        
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
        self.df = pd.DataFrame(data)
        
        # Process data for analysis
        self.process_data()
        return self.df
    
    def process_data(self):
        """Process data for visualization"""
        df = self.df.copy()
        
        # Extract BHK numbers
        df['bhk'] = df['size'].str.extract('(\d+)').astype(int)
        
        # Feature engineering
        df['price_per_sqft'] = (df['price'] * 100000) / df['total_sqft']
        df['room_ratio'] = df['bath'] / df['bhk']
        df['balcony_ratio'] = df['balcony'] / df['bhk']
        df['sqft_per_room'] = df['total_sqft'] / df['bhk']
        
        # Encode categorical variables for correlation analysis
        le_area = LabelEncoder()
        le_location = LabelEncoder()
        le_availability = LabelEncoder()
        
        df['area_type_encoded'] = le_area.fit_transform(df['area_type'])
        df['location_encoded'] = le_location.fit_transform(df['location'])
        df['availability_encoded'] = le_availability.fit_transform(df['availability'])
        
        self.processed_df = df
        print(f"Data processed successfully! Shape: {df.shape}")
    
    def create_correlation_heatmap(self):
        """Generate correlation heatmap"""
        print("Creating correlation heatmap...")
        
        # Select numerical columns for correlation
        numerical_cols = ['total_sqft', 'bath', 'balcony', 'bhk', 'price', 
                         'price_per_sqft', 'room_ratio', 'balcony_ratio', 'sqft_per_room',
                         'area_type_encoded', 'location_encoded', 'availability_encoded']
        
        corr_data = self.processed_df[numerical_cols]
        correlation_matrix = corr_data.corr()
        
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": .8},
                   fmt='.2f')
        
        plt.title('Feature Correlation Heatmap\nBengaluru House Price Dataset', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_price_heatmaps(self):
        """Create price analysis heatmaps"""
        print("Creating price analysis heatmaps...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Price by Location and BHK
        pivot_location_bhk = self.processed_df.pivot_table(
            values='price', index='location', columns='bhk', aggfunc='mean'
        )
        
        sns.heatmap(pivot_location_bhk, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=axes[0,0], cbar_kws={'label': 'Average Price (Lakhs)'})
        axes[0,0].set_title('Average Price by Location and BHK', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('BHK')
        axes[0,0].set_ylabel('Location')
        
        # 2. Price per sqft by Location and Area Type
        pivot_location_area = self.processed_df.pivot_table(
            values='price_per_sqft', index='location', columns='area_type', aggfunc='mean'
        )
        
        sns.heatmap(pivot_location_area, annot=True, fmt='.0f', cmap='viridis', 
                   ax=axes[0,1], cbar_kws={'label': 'Price per Sqft (₹)'})
        axes[0,1].set_title('Price per Sqft by Location and Area Type', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Area Type')
        axes[0,1].set_ylabel('Location')
        
        # 3. Price by BHK and Availability
        pivot_bhk_availability = self.processed_df.pivot_table(
            values='price', index='bhk', columns='availability', aggfunc='mean'
        )
        
        sns.heatmap(pivot_bhk_availability, annot=True, fmt='.1f', cmap='plasma', 
                   ax=axes[1,0], cbar_kws={'label': 'Average Price (Lakhs)'})
        axes[1,0].set_title('Average Price by BHK and Availability', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Availability')
        axes[1,0].set_ylabel('BHK')
        
        # 4. Total Sqft Distribution by Location and BHK
        pivot_sqft = self.processed_df.pivot_table(
            values='total_sqft', index='location', columns='bhk', aggfunc='mean'
        )
        
        sns.heatmap(pivot_sqft, annot=True, fmt='.0f', cmap='Blues', 
                   ax=axes[1,1], cbar_kws={'label': 'Average Sqft'})
        axes[1,1].set_title('Average Total Sqft by Location and BHK', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('BHK')
        axes[1,1].set_ylabel('Location')
        
        plt.tight_layout()
        plt.savefig('price_analysis_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_distribution_plots(self):
        """Create distribution and relationship plots"""
        print("Creating distribution plots...")
        
        fig, axes = plt.subplots(3, 3, figsize=(24, 20))
        
        # 1. Price distribution
        axes[0,0].hist(self.processed_df['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Price (Lakhs)')
        axes[0,0].set_ylabel('Frequency')
        
        # 2. Price vs Total Sqft scatter
        scatter = axes[0,1].scatter(self.processed_df['total_sqft'], self.processed_df['price'], 
                                   c=self.processed_df['bhk'], cmap='viridis', alpha=0.6)
        axes[0,1].set_title('Price vs Total Sqft (colored by BHK)', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Total Sqft')
        axes[0,1].set_ylabel('Price (Lakhs)')
        plt.colorbar(scatter, ax=axes[0,1], label='BHK')
        
        # 3. Price per sqft by location
        location_price = self.processed_df.groupby('location')['price_per_sqft'].mean().sort_values(ascending=False)
        axes[0,2].bar(range(len(location_price)), location_price.values, color='lightcoral')
        axes[0,2].set_title('Average Price per Sqft by Location', fontsize=14, fontweight='bold')
        axes[0,2].set_xlabel('Location')
        axes[0,2].set_ylabel('Price per Sqft (₹)')
        axes[0,2].set_xticks(range(len(location_price)))
        axes[0,2].set_xticklabels(location_price.index, rotation=45, ha='right')
        
        # 4. BHK distribution
        bhk_counts = self.processed_df['bhk'].value_counts().sort_index()
        axes[1,0].bar(bhk_counts.index, bhk_counts.values, color='lightgreen')
        axes[1,0].set_title('BHK Distribution', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('BHK')
        axes[1,0].set_ylabel('Count')
        
        # 5. Bath vs BHK relationship
        bath_bhk = self.processed_df.pivot_table(values='price', index='bath', columns='bhk', aggfunc='count', fill_value=0)
        sns.heatmap(bath_bhk, annot=True, fmt='d', cmap='Oranges', ax=axes[1,1])
        axes[1,1].set_title('Bath vs BHK Count Matrix', fontsize=14, fontweight='bold')
        
        # 6. Area type distribution
        area_counts = self.processed_df['area_type'].value_counts()
        axes[1,2].pie(area_counts.values, labels=area_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1,2].set_title('Area Type Distribution', fontsize=14, fontweight='bold')
        
        # 7. Price by availability box plot
        sns.boxplot(data=self.processed_df, x='availability', y='price', ax=axes[2,0])
        axes[2,0].set_title('Price Distribution by Availability', fontsize=14, fontweight='bold')
        axes[2,0].set_xlabel('Availability')
        axes[2,0].set_ylabel('Price (Lakhs)')
        axes[2,0].tick_params(axis='x', rotation=45)
        
        # 8. Balcony vs Price relationship
        axes[2,1].scatter(self.processed_df['balcony'], self.processed_df['price'], alpha=0.6, color='purple')
        axes[2,1].set_title('Balcony Count vs Price', fontsize=14, fontweight='bold')
        axes[2,1].set_xlabel('Number of Balconies')
        axes[2,1].set_ylabel('Price (Lakhs)')
        
        # 9. Room ratio distribution
        axes[2,2].hist(self.processed_df['room_ratio'], bins=30, alpha=0.7, color='gold', edgecolor='black')
        axes[2,2].set_title('Bath to BHK Ratio Distribution', fontsize=14, fontweight='bold')
        axes[2,2].set_xlabel('Bath/BHK Ratio')
        axes[2,2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_location_analysis(self):
        """Create detailed location-based analysis"""
        print("Creating location analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Average price by location (bar chart)
        location_stats = self.processed_df.groupby('location').agg({
            'price': ['mean', 'median', 'std'],
            'total_sqft': 'mean',
            'price_per_sqft': 'mean'
        }).round(2)
        
        location_means = location_stats['price']['mean'].sort_values(ascending=True)
        bars = axes[0,0].barh(range(len(location_means)), location_means.values, color='steelblue')
        axes[0,0].set_yticks(range(len(location_means)))
        axes[0,0].set_yticklabels(location_means.index)
        axes[0,0].set_title('Average Price by Location', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Average Price (Lakhs)')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0,0].text(width + 1, bar.get_y() + bar.get_height()/2, 
                          f'{width:.1f}', ha='left', va='center')
        
        # 2. Price variation by location (box plot)
        location_order = location_means.index
        sns.boxplot(data=self.processed_df, y='location', x='price', 
                   order=location_order, ax=axes[0,1])
        axes[0,1].set_title('Price Distribution by Location', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Price (Lakhs)')
        axes[0,1].set_ylabel('Location')
        
        # 3. Location preference heatmap (count of properties)
        location_bhk_count = self.processed_df.pivot_table(
            values='price', index='location', columns='bhk', aggfunc='count', fill_value=0
        )
        
        sns.heatmap(location_bhk_count, annot=True, fmt='d', cmap='YlGnBu', ax=axes[1,0])
        axes[1,0].set_title('Property Count by Location and BHK', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('BHK')
        axes[1,0].set_ylabel('Location')
        
        # 4. Price per sqft comparison
        price_per_sqft_location = self.processed_df.groupby('location')['price_per_sqft'].mean().sort_values(ascending=True)
        bars = axes[1,1].barh(range(len(price_per_sqft_location)), price_per_sqft_location.values, color='coral')
        axes[1,1].set_yticks(range(len(price_per_sqft_location)))
        axes[1,1].set_yticklabels(price_per_sqft_location.index)
        axes[1,1].set_title('Average Price per Sqft by Location', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Price per Sqft (₹)')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[1,1].text(width + 100, bar.get_y() + bar.get_height()/2, 
                          f'{width:.0f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig('location_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_feature_importance_visualization(self):
        """Create feature importance and relationship visualization"""
        print("Creating feature importance visualization...")
        
        # Calculate feature correlations with price
        numerical_features = ['total_sqft', 'bath', 'balcony', 'bhk', 
                             'price_per_sqft', 'room_ratio', 'balcony_ratio', 'sqft_per_room']
        
        correlations = self.processed_df[numerical_features + ['price']].corr()['price'].drop('price').abs().sort_values(ascending=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Feature importance (correlation with price)
        bars = axes[0,0].barh(range(len(correlations)), correlations.values, color='lightseagreen')
        axes[0,0].set_yticks(range(len(correlations)))
        axes[0,0].set_yticklabels(correlations.index)
        axes[0,0].set_title('Feature Importance (Correlation with Price)', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Absolute Correlation')
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0,0].text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                          f'{width:.3f}', ha='left', va='center')
        
        # 2. Pairplot matrix for top features
        top_features = ['price', 'total_sqft', 'price_per_sqft', 'bhk']
        scatter_data = self.processed_df[top_features]
        
        # Create a correlation matrix for top features
        top_corr = scatter_data.corr()
        sns.heatmap(top_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0,1])
        axes[0,1].set_title('Top Features Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 3. Price vs key features scatter plots
        axes[1,0].scatter(self.processed_df['total_sqft'], self.processed_df['price'], alpha=0.5, color='blue')
        axes[1,0].set_title('Price vs Total Sqft', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Total Sqft')
        axes[1,0].set_ylabel('Price (Lakhs)')
        
        # Add trend line
        z = np.polyfit(self.processed_df['total_sqft'], self.processed_df['price'], 1)
        p = np.poly1d(z)
        axes[1,0].plot(self.processed_df['total_sqft'], p(self.processed_df['total_sqft']), "r--", alpha=0.8)
        
        # 4. Feature distribution comparison
        feature_stats = self.processed_df[numerical_features].describe().loc[['mean', 'std']].T
        
        x = np.arange(len(feature_stats))
        width = 0.35
        
        bars1 = axes[1,1].bar(x - width/2, feature_stats['mean'], width, label='Mean', alpha=0.8)
        bars2 = axes[1,1].bar(x + width/2, feature_stats['std'], width, label='Std Dev', alpha=0.8)
        
        axes[1,1].set_xlabel('Features')
        axes[1,1].set_ylabel('Values')
        axes[1,1].set_title('Feature Statistics Comparison', fontsize=14, fontweight='bold')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(feature_stats.index, rotation=45, ha='right')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_performance_comparison(self):
        """Create model performance comparison visualization"""
        print("Creating model performance comparison...")
        
        # Simulated model performance data
        models = ['Linear Regression', 'Random Forest', 'XGBoost', 'CatBoost', 'LightGBM', 'Deep Learning', 'Hybrid Ensemble']
        r2_scores = [0.768, 0.989, 0.998, 0.998, 0.991, 0.933, 0.992]
        rmse_scores = [31.53, 6.62, 5.26, 5.43, 6.10, 16.65, 5.89]
        mae_scores = [21.60, 4.47, 3.14, 3.87, 4.14, 12.30, 3.99]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. R² Score comparison
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = axes[0,0].bar(range(len(models)), r2_scores, color=colors)
        axes[0,0].set_title('Model Performance - R² Score Comparison', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Models')
        axes[0,0].set_ylabel('R² Score')
        axes[0,0].set_xticks(range(len(models)))
        axes[0,0].set_xticklabels(models, rotation=45, ha='right')
        axes[0,0].set_ylim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, r2_scores):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. RMSE comparison
        bars = axes[0,1].bar(range(len(models)), rmse_scores, color=colors)
        axes[0,1].set_title('Model Performance - RMSE Comparison', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Models')
        axes[0,1].set_ylabel('RMSE (Lakhs)')
        axes[0,1].set_xticks(range(len(models)))
        axes[0,1].set_xticklabels(models, rotation=45, ha='right')
        
        # 3. MAE comparison
        bars = axes[1,0].bar(range(len(models)), mae_scores, color=colors)
        axes[1,0].set_title('Model Performance - MAE Comparison', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Models')
        axes[1,0].set_ylabel('MAE (Lakhs)')
        axes[1,0].set_xticks(range(len(models)))
        axes[1,0].set_xticklabels(models, rotation=45, ha='right')
        
        # 4. Algorithm evolution
        evolution_order = ['Linear Regression', 'Random Forest', 'XGBoost', 'CatBoost', 'LightGBM', 'Deep Learning', 'Hybrid Ensemble']
        evolution_scores = [0.768, 0.989, 0.998, 0.998, 0.991, 0.933, 0.992]
        
        axes[1,1].plot(range(len(evolution_scores)), evolution_scores, 'o-', linewidth=3, markersize=10, color='darkgreen')
        axes[1,1].set_title('Algorithm Evolution - Performance Progression', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('Algorithm Progression')
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].set_xticks(range(len(evolution_order)))
        axes[1,1].set_xticklabels([name.replace(' ', '\n') for name in evolution_order], rotation=0)
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_ylim(0.7, 1.0)
        
        # Add value labels on the line
        for i, score in enumerate(evolution_scores):
            axes[1,1].text(i, score + 0.005, f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("🎨 GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 60)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Generate all visualizations
        self.create_correlation_heatmap()
        self.create_price_heatmaps()
        self.create_distribution_plots()
        self.create_location_analysis()
        self.create_feature_importance_visualization()
        self.create_model_performance_comparison()
        
        print("\n✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("📊 Generated files:")
        print("   - correlation_heatmap.png")
        print("   - price_analysis_heatmaps.png")
        print("   - distribution_analysis.png")
        print("   - location_analysis.png")
        print("   - feature_importance_analysis.png")
        print("   - model_performance_comparison.png")

def main():
    """Main execution function"""
    visualizer = ComprehensiveVisualizer()
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()