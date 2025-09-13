import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for professional visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_algorithm_comparison():
    """Create comprehensive algorithm comparison visualization"""
    
    # Algorithm performance data from your project results
    algorithms_data = {
        'Algorithm': [
            'Deep Neural Network',
            'XGBoost', 
            'CatBoost',
            'Random Forest',
            'LightGBM',
            'Hybrid Ensemble',
            'Linear Regression'
        ],
        'R² Score': [0.999, 0.998, 0.998, 0.998, 0.998, 0.997, 0.907],
        'RMSE': [4.85, 5.26, 5.43, 6.34, 6.46, 6.73, 40.96],
        'MAE': [3.31, 3.14, 3.87, 3.67, 3.72, 4.71, 28.84],
        'Rank': [1, 2, 3, 4, 5, 6, 7],
        'Algorithm_Type': ['Deep Learning', 'Ensemble', 'Ensemble', 'Ensemble', 'Ensemble', 'Hybrid', 'Linear'],
        'Training_Time': ['High', 'Medium', 'Medium', 'Low', 'Low', 'High', 'Very Low'],
        'Interpretability': ['Low', 'Medium', 'High', 'High', 'Medium', 'Medium', 'Very High']
    }
    
    df = pd.DataFrame(algorithms_data)
    
    # Create comprehensive comparison visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance Metrics Bar Chart
    ax1 = plt.subplot(3, 3, 1)
    x = np.arange(len(df))
    width = 0.25
    
    bars1 = ax1.bar(x - width, df['R² Score'], width, label='R² Score', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x, df['RMSE']/50, width, label='RMSE (scaled /50)', alpha=0.8, color='lightcoral')  # Scale RMSE for visibility
    bars3 = ax1.bar(x + width, df['MAE']/30, width, label='MAE (scaled /30)', alpha=0.8, color='lightgreen')  # Scale MAE for visibility
    
    ax1.set_xlabel('Algorithms', fontweight='bold')
    ax1.set_ylabel('Score/Metric', fontweight='bold')
    ax1.set_title('Algorithm Performance Comparison\n(RMSE & MAE scaled for visibility)', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['Algorithm'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
        ax1.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.01,
                f'{df.iloc[i]["R² Score"]:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 2. R² Score Ranking
    ax2 = plt.subplot(3, 3, 2)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))
    bars = ax2.barh(df['Algorithm'], df['R² Score'], color=colors, alpha=0.8)
    ax2.set_xlabel('R² Score', fontweight='bold')
    ax2.set_title('R² Score Comparison\n(Higher is Better)', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2., 
                f'{width:.3f}', ha='left', va='center', fontweight='bold')
    
    # 3. RMSE Comparison
    ax3 = plt.subplot(3, 3, 3)
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(df)))  # Reverse color for RMSE (lower is better)
    bars = ax3.barh(df['Algorithm'], df['RMSE'], color=colors, alpha=0.8)
    ax3.set_xlabel('RMSE (Lower is Better)', fontweight='bold')
    ax3.set_title('Root Mean Square Error\n(Lower is Better)', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2., 
                f'{width:.2f}', ha='left', va='center', fontweight='bold')
    
    # 4. MAE Comparison
    ax4 = plt.subplot(3, 3, 4)
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(df)))
    bars = ax4.barh(df['Algorithm'], df['MAE'], color=colors, alpha=0.8)
    ax4.set_xlabel('MAE (Lower is Better)', fontweight='bold')
    ax4.set_title('Mean Absolute Error\n(Lower is Better)', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width + 0.3, bar.get_y() + bar.get_height()/2., 
                f'{width:.2f}', ha='left', va='center', fontweight='bold')
    
    # 5. Algorithm Type Distribution
    ax5 = plt.subplot(3, 3, 5)
    type_counts = df['Algorithm_Type'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    wedges, texts, autotexts = ax5.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    ax5.set_title('Algorithm Types Distribution', fontweight='bold', fontsize=14)
    
    # 6. Performance vs Complexity Matrix
    ax6 = plt.subplot(3, 3, 6)
    
    # Create complexity scores
    complexity_scores = {'Very Low': 1, 'Low': 2, 'Medium': 3, 'High': 4}
    interpretability_scores = {'Very High': 5, 'High': 4, 'Medium': 3, 'Low': 2}
    
    x_complexity = [complexity_scores[x] for x in df['Training_Time']]
    y_interpretability = [interpretability_scores[x] for x in df['Interpretability']]
    
    # Size represents R² score
    sizes = (df['R² Score'] * 1000)
    
    scatter = ax6.scatter(x_complexity, y_interpretability, s=sizes, 
                         c=df['R² Score'], cmap='RdYlGn', alpha=0.7, edgecolors='black')
    
    # Add algorithm labels
    for i, txt in enumerate(df['Algorithm']):
        ax6.annotate(txt[:8], (x_complexity[i], y_interpretability[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax6.set_xlabel('Training Complexity →', fontweight='bold')
    ax6.set_ylabel('Interpretability →', fontweight='bold')
    ax6.set_title('Performance vs Complexity\n(Size = R² Score)', fontweight='bold', fontsize=14)
    ax6.set_xticks([1, 2, 3, 4])
    ax6.set_xticklabels(['Very Low', 'Low', 'Medium', 'High'])
    ax6.set_yticks([2, 3, 4, 5])
    ax6.set_yticklabels(['Low', 'Medium', 'High', 'Very High'])
    ax6.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('R² Score', fontweight='bold')
    
    # 7. Ranking Comparison
    ax7 = plt.subplot(3, 3, 7)
    
    # Create medal colors for ranking
    medal_colors = ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen', 'orange', 'red']
    
    for i, (alg, rank) in enumerate(zip(df['Algorithm'], df['Rank'])):
        ax7.barh(i, 8-rank, color=medal_colors[rank-1], alpha=0.8)
        ax7.text(8-rank+0.1, i, f'#{rank}', va='center', fontweight='bold')
    
    ax7.set_yticks(range(len(df)))
    ax7.set_yticklabels(df['Algorithm'])
    ax7.set_xlabel('Performance Ranking (Left to Right: Best to Worst)', fontweight='bold')
    ax7.set_title('Algorithm Ranking\n(#1 = Best Performance)', fontweight='bold', fontsize=14)
    ax7.grid(True, alpha=0.3, axis='x')
    
    # 8. Detailed Performance Table
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    # Create performance table
    table_data = df[['Algorithm', 'R² Score', 'RMSE', 'MAE', 'Rank']].round(3)
    
    # Create table
    table = ax8.table(cellText=table_data.values, 
                     colLabels=table_data.columns,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color code the table
    for i in range(len(table_data)):
        rank = table_data.iloc[i]['Rank']
        if rank == 1:
            color = 'lightgreen'
        elif rank <= 3:
            color = 'lightyellow'
        else:
            color = 'lightcoral'
        
        for j in range(len(table_data.columns)):
            table[(i+1, j)].set_facecolor(color)
    
    # Header styling
    for j in range(len(table_data.columns)):
        table[(0, j)].set_facecolor('lightblue')
        table[(0, j)].set_text_props(weight='bold')
    
    ax8.set_title('Performance Summary Table\n(Green=Best, Yellow=Good, Red=Needs Improvement)', 
                 fontweight='bold', fontsize=14)
    
    # 9. Key Insights and Recommendations
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    insights_text = f"""
    🏆 ALGORITHM COMPARISON INSIGHTS
    
    🥇 CHAMPION: Deep Neural Network
    • R² Score: {df.iloc[0]['R² Score']:.3f} (99.9% accuracy)
    • Lowest RMSE: {df.iloc[0]['RMSE']:.2f}
    • Best overall performance
    
    🥈 RUNNER-UP: XGBoost  
    • Excellent R² Score: {df.iloc[1]['R² Score']:.3f}
    • Good balance of performance/speed
    • Industry standard for tabular data
    
    📊 KEY FINDINGS:
    • All ensemble methods achieved >99.7% accuracy
    • Deep Learning excels but requires more resources
    • Linear Regression shows significant limitations
    • Ensemble methods provide robust performance
    
    💡 RECOMMENDATIONS:
    • Production: Use XGBoost (best speed/performance)
    • Research: Deep Neural Network (highest accuracy)
    • Interpretability: CatBoost or Random Forest
    • Quick prototyping: Random Forest
    
    🎯 PERFORMANCE RANGE:
    • Best: 99.9% accuracy (Deep NN)
    • Worst: 90.7% accuracy (Linear Reg)
    • Average ensemble: 99.8% accuracy
    """
    
    ax9.text(0.05, 0.95, insights_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
    
    # Overall title and layout
    fig.suptitle('COMPREHENSIVE ALGORITHM COMPARISON\nBengaluru House Price Prediction Project', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, hspace=0.3, wspace=0.3)
    
    # Save the visualization
    plt.savefig('algorithm_comparison_dashboard.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: algorithm_comparison_dashboard.png")
    
    # Create a separate detailed table
    create_detailed_comparison_table(df)
    
    plt.show()

def create_detailed_comparison_table(df):
    """Create a detailed comparison table with additional metrics"""
    
    # Expand the data with additional characteristics
    detailed_data = {
        'Algorithm': df['Algorithm'],
        'R² Score': df['R² Score'],
        'RMSE': df['RMSE'],
        'MAE': df['MAE'],
        'Rank': df['Rank'],
        'Training Speed': ['Slow', 'Fast', 'Medium', 'Fast', 'Very Fast', 'Slow', 'Very Fast'],
        'Prediction Speed': ['Fast', 'Fast', 'Fast', 'Medium', 'Very Fast', 'Medium', 'Very Fast'],
        'Memory Usage': ['High', 'Medium', 'Medium', 'Low', 'Low', 'High', 'Very Low'],
        'Hyperparameter Tuning': ['Complex', 'Medium', 'Easy', 'Easy', 'Medium', 'Complex', 'Minimal'],
        'Overfitting Risk': ['Medium', 'Low', 'Low', 'Medium', 'Low', 'Low', 'Very Low'],
        'Best Use Case': ['Research/High Accuracy', 'Production', 'Production', 'Prototyping', 'Large Data', 'Ensemble', 'Baseline']
    }
    
    detailed_df = pd.DataFrame(detailed_data)
    
    # Create detailed table visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=detailed_df.values, 
                    colLabels=detailed_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color coding based on rank
    colors = ['#FFD700', '#C0C0C0', '#CD7F32', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C']
    
    for i in range(len(detailed_df)):
        rank = detailed_df.iloc[i]['Rank']
        base_color = colors[rank-1]
        
        for j in range(len(detailed_df.columns)):
            table[(i+1, j)].set_facecolor(base_color)
            if j < 4:  # Performance metrics
                table[(i+1, j)].set_text_props(weight='bold')
    
    # Header styling
    for j in range(len(detailed_df.columns)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    plt.title('DETAILED ALGORITHM COMPARISON TABLE\nBengaluru House Price Prediction Project', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_text = """
    🏆 Color Legend:
    🥇 Gold: Rank #1    🥈 Silver: Rank #2    🥉 Bronze: Rank #3
    🔵 Blue: Rank #4    🟢 Green: Rank #5     🟣 Purple: Rank #6    🟡 Yellow: Rank #7
    """
    
    plt.figtext(0.5, 0.02, legend_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('detailed_algorithm_table.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: detailed_algorithm_table.png")
    
    plt.show()
    
    # Print summary to console
    print("\n" + "="*80)
    print("📊 ALGORITHM COMPARISON SUMMARY")
    print("="*80)
    for i, row in detailed_df.iterrows():
        print(f"\n{row['Rank']}. {row['Algorithm']}")
        print(f"   R² Score: {row['R² Score']:.3f} | RMSE: {row['RMSE']:.2f} | MAE: {row['MAE']:.2f}")
        print(f"   Best for: {row['Best Use Case']}")
        print(f"   Training: {row['Training Speed']} | Memory: {row['Memory Usage']}")
    
    print(f"\n🏆 WINNER: {detailed_df.iloc[0]['Algorithm']} with {detailed_df.iloc[0]['R² Score']:.1%} accuracy!")
    print("="*80)

if __name__ == "__main__":
    create_algorithm_comparison()