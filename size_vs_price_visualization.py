import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_enhanced_sample_data():
    """Create comprehensive sample data for visualization"""
    np.random.seed(42)
    n_samples = 3000
    
    locations = ['Electronic City', 'Whitefield', 'Sarjapur Road', 'Marathahalli', 
                'Koramangala', 'Indiranagar', 'HSR Layout', 'BTM Layout', 'Jayanagar',
                'Rajajinagar', 'Malleshwaram', 'Yelahanka', 'Hebbal', 'Bannerghatta Road',
                'Kanakapura Road', 'Mysore Road', 'Tumkur Road', 'Hennur Road', 
                'Outer Ring Road', 'Banashankari']
    
    # Generate realistic data
    total_sqft = np.random.lognormal(mean=7.5, sigma=0.4, size=n_samples)
    total_sqft = np.clip(total_sqft, 500, 5000)
    
    # Price correlates with size but with variation
    base_price = total_sqft * np.random.normal(5000, 1000, size=n_samples) / 100000  # Convert to lakhs
    location_multiplier = np.random.choice([0.8, 1.0, 1.2, 1.5, 2.0], size=n_samples, 
                                         p=[0.2, 0.3, 0.3, 0.15, 0.05])
    price = base_price * location_multiplier
    price = np.clip(price, 20, 500)  # Reasonable price range
    
    # BHK based on size
    bhk = np.where(total_sqft < 700, 1,
          np.where(total_sqft < 1000, 2,
          np.where(total_sqft < 1500, 3,
          np.where(total_sqft < 2200, 4, 5))))
    
    # Create DataFrame
    data = {
        'area_type': np.random.choice(['Super built-up Area', 'Built-up Area', 'Plot Area', 'Carpet Area'], 
                                    size=n_samples, p=[0.4, 0.35, 0.15, 0.1]),
        'availability': np.random.choice(['Ready To Move', '19-Dec', '18-Jul', '18-Sep'], 
                                       size=n_samples, p=[0.6, 0.15, 0.15, 0.1]),
        'location': np.random.choice(locations, size=n_samples),
        'size': [f'{b} BHK' for b in bhk],
        'total_sqft': total_sqft,
        'bath': np.clip(bhk + np.random.randint(-1, 2, size=n_samples), 1, 5),
        'balcony': np.clip(np.random.randint(0, 4, size=n_samples), 0, 3),
        'price': price,
        'bhk': bhk
    }
    
    df = pd.DataFrame(data)
    print(f"Sample data created: {len(df)} records")
    return df

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

def create_size_vs_price_graphs():
    """Create comprehensive size vs price visualizations"""
    
    # Load the data
    try:
        df = pd.read_csv('bengaluru_house_data.csv')
        print(f"Loaded data with {len(df)} records")
    except FileNotFoundError:
        print("CSV not found, creating enhanced sample data...")
        df = create_enhanced_sample_data()
        print(f"Created sample data with {len(df)} records")
    
    # Data preprocessing
    print("Preprocessing data...")
    
    # Clean and prepare data
    df_clean = df.dropna(subset=['total_sqft', 'price'])
    
    # Convert total_sqft to numeric (handle ranges like "1200-1300")
    def convert_sqft_to_num(x):
        tokens = str(x).split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        try:
            return float(x)
        except:
            return None
    
    if 'total_sqft' in df_clean.columns:
        df_clean['total_sqft_num'] = df_clean['total_sqft'].apply(convert_sqft_to_num)
    else:
        df_clean['total_sqft_num'] = df_clean['total_sqft']
    
    df_clean = df_clean.dropna(subset=['total_sqft_num'])
    
    # Remove outliers
    df_clean = remove_outliers_iqr(df_clean, 'total_sqft_num')
    df_clean = remove_outliers_iqr(df_clean, 'price')
    
    # Create price per sqft
    df_clean['price_per_sqft'] = df_clean['price'] * 100000 / df_clean['total_sqft_num']  # Convert lakhs to rupees
    
    print(f"Clean data: {len(df_clean)} records")
    
    # Create the comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Main scatter plot: Size vs Price with trend line
    ax1 = plt.subplot(3, 3, 1)
    scatter = ax1.scatter(df_clean['total_sqft_num'], df_clean['price'], 
                         c=df_clean['price_per_sqft'], cmap='viridis', alpha=0.6, s=50)
    
    # Add trend line
    X = df_clean['total_sqft_num'].values.reshape(-1, 1)
    y = df_clean['price'].values
    reg = LinearRegression().fit(X, y)
    trend_line = reg.predict(X)
    ax1.plot(df_clean['total_sqft_num'], trend_line, color='red', linewidth=3, label=f'Trend Line (R² = {reg.score(X, y):.3f})')
    
    ax1.set_xlabel('Total Square Feet', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Price (Lakhs ₹)', fontsize=12, fontweight='bold')
    ax1.set_title('Property Size vs Price\n(Color = Price per sqft)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Price per Sqft (₹)', fontweight='bold')
    
    # 2. Size vs Price by BHK
    ax2 = plt.subplot(3, 3, 2)
    if 'bhk' in df_clean.columns:
        bhk_values = sorted(df_clean['bhk'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(bhk_values)))
        
        for i, bhk in enumerate(bhk_values[:6]):  # Top 6 BHK types
            bhk_data = df_clean[df_clean['bhk'] == bhk]
            ax2.scatter(bhk_data['total_sqft_num'], bhk_data['price'], 
                       label=f'{bhk} BHK', alpha=0.7, s=40, color=colors[i])
        
        ax2.set_xlabel('Total Square Feet', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Price (Lakhs ₹)', fontsize=12, fontweight='bold')
        ax2.set_title('Size vs Price by BHK', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'BHK data\nnot available', ha='center', va='center', transform=ax2.transAxes, fontsize=16)
        ax2.set_title('Size vs Price by BHK', fontsize=14, fontweight='bold')
    
    # 3. Size distribution histogram
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(df_clean['total_sqft_num'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(df_clean['total_sqft_num'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_clean["total_sqft_num"].mean():.0f} sqft')
    ax3.axvline(df_clean['total_sqft_num'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df_clean["total_sqft_num"].median():.0f} sqft')
    ax3.set_xlabel('Total Square Feet', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Size Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Price distribution histogram
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(df_clean['price'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax4.axvline(df_clean['price'].mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean: ₹{df_clean["price"].mean():.1f}L')
    ax4.axvline(df_clean['price'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: ₹{df_clean["price"].median():.1f}L')
    ax4.set_xlabel('Price (Lakhs ₹)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Price Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Price per sqft analysis
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(df_clean['total_sqft_num'], df_clean['price_per_sqft'], alpha=0.6, s=40, color='green')
    ax5.set_xlabel('Total Square Feet', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Price per Sqft (₹)', fontsize=12, fontweight='bold')
    ax5.set_title('Size vs Price per Sqft', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add trend line for price per sqft
    X_sqft = df_clean['total_sqft_num'].values.reshape(-1, 1)
    y_price_per_sqft = df_clean['price_per_sqft'].values
    reg_sqft = LinearRegression().fit(X_sqft, y_price_per_sqft)
    trend_sqft = reg_sqft.predict(X_sqft)
    ax5.plot(df_clean['total_sqft_num'], trend_sqft, color='red', linewidth=2, 
             label=f'Trend (R² = {reg_sqft.score(X_sqft, y_price_per_sqft):.3f})')
    ax5.legend()
    
    # 6. Size categories analysis
    ax6 = plt.subplot(3, 3, 6)
    df_clean['size_category'] = pd.cut(df_clean['total_sqft_num'], 
                                      bins=[0, 800, 1200, 1800, 2500, float('inf')],
                                      labels=['Small (<800)', 'Medium (800-1200)', 'Large (1200-1800)', 'Very Large (1800-2500)', 'Luxury (>2500)'])
    
    size_price = df_clean.groupby('size_category')['price'].agg(['mean', 'median', 'count']).reset_index()
    
    x_pos = range(len(size_price))
    width = 0.35
    
    bars1 = ax6.bar([x - width/2 for x in x_pos], size_price['mean'], width, label='Mean Price', alpha=0.8, color='lightblue')
    bars2 = ax6.bar([x + width/2 for x in x_pos], size_price['median'], width, label='Median Price', alpha=0.8, color='lightgreen')
    
    ax6.set_xlabel('Size Category', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Price (Lakhs ₹)', fontsize=12, fontweight='bold')
    ax6.set_title('Average Price by Size Category', fontsize=14, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(size_price['size_category'], rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height = max(bar1.get_height(), bar2.get_height())
        ax6.text(bar1.get_x() + bar1.get_width()/2., height + 5,
                f'n={size_price.iloc[i]["count"]}', 
                ha='center', va='bottom', fontweight='bold')
    
    # 7. Polynomial regression analysis
    ax7 = plt.subplot(3, 3, 7)
    
    # Sample data for polynomial fit (to avoid overcrowding)
    sample_df = df_clean.sample(min(1000, len(df_clean)), random_state=42)
    X_sample = sample_df['total_sqft_num'].values.reshape(-1, 1)
    y_sample = sample_df['price'].values
    
    # Polynomial features
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X_sample)
    
    # Fit polynomial regression
    poly_reg = LinearRegression().fit(X_poly, y_sample)
    
    # Create smooth curve for plotting
    X_smooth = np.linspace(sample_df['total_sqft_num'].min(), sample_df['total_sqft_num'].max(), 100).reshape(-1, 1)
    X_smooth_poly = poly_features.transform(X_smooth)
    y_smooth = poly_reg.predict(X_smooth_poly)
    
    ax7.scatter(sample_df['total_sqft_num'], sample_df['price'], alpha=0.5, s=30, color='gray', label='Data Points')
    ax7.plot(X_smooth.flatten(), y_smooth, color='red', linewidth=3, label=f'Polynomial Fit (R² = {poly_reg.score(X_poly, y_sample):.3f})')
    
    ax7.set_xlabel('Total Square Feet', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Price (Lakhs ₹)', fontsize=12, fontweight='bold')
    ax7.set_title('Polynomial Regression\n(Degree 2)', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Size vs Price correlation heatmap by location (top 10 locations)
    ax8 = plt.subplot(3, 3, 8)
    if 'location' in df_clean.columns:
        top_locations = df_clean['location'].value_counts().head(10).index
        location_stats = []
        
        for loc in top_locations:
            loc_data = df_clean[df_clean['location'] == loc]
            correlation = loc_data['total_sqft_num'].corr(loc_data['price'])
            avg_price = loc_data['price'].mean()
            avg_size = loc_data['total_sqft_num'].mean()
            location_stats.append({
                'location': loc[:15],  # Truncate long names
                'correlation': correlation,
                'avg_price': avg_price,
                'avg_size': avg_size,
                'count': len(loc_data)
            })
        
        stats_df = pd.DataFrame(location_stats)
        
        # Create heatmap data
        heatmap_data = stats_df.set_index('location')[['correlation', 'avg_price', 'avg_size']].T
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax8, cbar_kws={'label': 'Value'})
        ax8.set_title('Location Analysis\n(Correlation, Avg Price, Avg Size)', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Locations', fontsize=12, fontweight='bold')
    else:
        ax8.text(0.5, 0.5, 'Location data\nnot available', ha='center', va='center', transform=ax8.transAxes, fontsize=16)
        ax8.set_title('Location Analysis', fontsize=14, fontweight='bold')
    
    # 9. Statistical summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate statistics
    size_price_corr = df_clean['total_sqft_num'].corr(df_clean['price'])
    
    stats_text = f"""
    SIZE vs PRICE ANALYSIS
    
    📊 CORRELATION ANALYSIS
    Size-Price Correlation: {size_price_corr:.3f}
    
    📏 SIZE STATISTICS
    Mean Size: {df_clean['total_sqft_num'].mean():.0f} sqft
    Median Size: {df_clean['total_sqft_num'].median():.0f} sqft
    Size Range: {df_clean['total_sqft_num'].min():.0f} - {df_clean['total_sqft_num'].max():.0f} sqft
    
    💰 PRICE STATISTICS  
    Mean Price: ₹{df_clean['price'].mean():.1f} Lakhs
    Median Price: ₹{df_clean['price'].median():.1f} Lakhs
    Price Range: ₹{df_clean['price'].min():.1f}L - ₹{df_clean['price'].max():.1f}L
    
    📈 PRICE PER SQFT
    Mean: ₹{df_clean['price_per_sqft'].mean():.0f}/sqft
    Median: ₹{df_clean['price_per_sqft'].median():.0f}/sqft
    
    📋 DATASET INFO
    Total Properties: {len(df_clean):,}
    
    🎯 KEY INSIGHTS
    • {'Strong' if size_price_corr > 0.7 else 'Moderate' if size_price_corr > 0.5 else 'Weak'} size-price correlation
    • Size explains {size_price_corr**2*100:.1f}% of price variation
    """
    
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Overall title and layout
    fig.suptitle('COMPREHENSIVE SIZE vs PRICE ANALYSIS\nBengaluru House Price Prediction Project', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
    
    # Save the visualization
    plt.savefig('size_vs_price_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: size_vs_price_analysis.png")
    
    # Create a focused size vs price plot
    create_focused_size_price_plot(df_clean)
    
    plt.show()

def create_focused_size_price_plot(df_clean):
    """Create a focused, clean size vs price plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Clean scatter with regression line
    ax1.scatter(df_clean['total_sqft_num'], df_clean['price'], 
               alpha=0.6, s=50, color='steelblue', edgecolors='white', linewidth=0.5)
    
    # Add regression line with confidence interval
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_clean['total_sqft_num'], df_clean['price'])
    line = slope * df_clean['total_sqft_num'] + intercept
    ax1.plot(df_clean['total_sqft_num'], line, 'r-', linewidth=3, label=f'Linear Fit (R² = {r_value**2:.3f})')
    
    ax1.set_xlabel('Property Size (Square Feet)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (Lakhs ₹)', fontsize=14, fontweight='bold')
    ax1.set_title('Property Size vs Price\nLinear Relationship Analysis', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Add equation text
    equation_text = f'Price = {slope:.2f} × Size + {intercept:.1f}\nR² = {r_value**2:.3f} (Strong Correlation)'
    ax1.text(0.05, 0.95, equation_text, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # Plot 2: Hexbin plot for density
    hb = ax2.hexbin(df_clean['total_sqft_num'], df_clean['price'], gridsize=30, cmap='YlOrRd', mincnt=1)
    ax2.set_xlabel('Property Size (Square Feet)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price (Lakhs ₹)', fontsize=14, fontweight='bold')
    ax2.set_title('Property Size vs Price\nDensity Distribution', fontsize=16, fontweight='bold')
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax2)
    cb.set_label('Number of Properties', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('focused_size_vs_price.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: focused_size_vs_price.png")
    
    plt.show()

if __name__ == "__main__":
    create_size_vs_price_graphs()