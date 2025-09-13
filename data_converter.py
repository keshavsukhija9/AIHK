import pandas as pd
import numpy as np
from pathlib import Path
import os

def convert_numbers_to_csv():
    """
    Convert Numbers file to CSV format
    Since Numbers format is proprietary, we'll provide instructions for manual conversion
    """
    numbers_file = "Bengaluru_House_Data.numbers"
    
    if os.path.exists(numbers_file):
        print(f"Found {numbers_file}")
        print("\nTo convert this Numbers file to CSV:")
        print("1. Open the file in Apple Numbers")
        print("2. Go to File > Export To > CSV...")
        print("3. Save as 'bengaluru_house_data.csv'")
        print("4. Run this script again after conversion")
        return False
    
    # Check if CSV already exists
    csv_file = "bengaluru_house_data.csv"
    if os.path.exists(csv_file):
        print(f"Found {csv_file} - proceeding with data analysis")
        return True
    
    print(f"Neither {numbers_file} nor {csv_file} found")
    return False

def analyze_sample_data():
    """
    Create sample data structure for demonstration if no data file exists
    """
    print("Creating sample Bengaluru house data structure...")
    
    sample_data = {
        'area_type': ['Super built-up Area', 'Plot Area', 'Built-up Area', 'Carpet Area'],
        'availability': ['Ready To Move', '19-Dec', '18-Jul', 'Ready To Move'],
        'location': ['Electronic City Phase II', 'Chikka Tirupathi', 'Uttarahalli', 'Lingadheeranahalli'],
        'size': ['2 BHK', '4 Bedroom', '3 BHK', '3 BHK'],
        'society': ['Coomee', 'Theanmp', 'Deans Ace', 'Soiewre'],
        'total_sqft': ['1056', '2600', '1440', '1521'],
        'bath': [2, 2, 3, 3],
        'balcony': [1, 2, 2, 1],
        'price': [39.07, 120.00, 62.00, 95.00]
    }
    
    df = pd.DataFrame(sample_data)
    print("\nSample data structure:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    if not convert_numbers_to_csv():
        analyze_sample_data()