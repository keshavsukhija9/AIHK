import pandas as pd
import numpy as np
import os
import re

def convert_and_process_data():
    """
    Convert Numbers file to CSV and process the Bengaluru house data
    """
    print("Bengaluru House Price Prediction - Data Processing")
    print("=" * 50)
    
    # Check for existing CSV file
    csv_file = "bengaluru_house_data.csv"
    numbers_file = "Bengaluru_House_Data.numbers"
    
    if not os.path.exists(csv_file):
        if os.path.exists(numbers_file):
            print(f"Found {numbers_file}")
            print("\nTo use this data, please convert it to CSV format:")
            print("1. Open the Numbers file in Apple Numbers")
            print("2. File > Export To > CSV...")
            print("3. Save as 'bengaluru_house_data.csv' in this directory")
            print("4. Run this script again")
            print("\nAlternatively, you can run the main prediction script which includes sample data generation.")
            return None
        else:
            print("No data file found. Using sample data generation.")
            return None
    
    # Load and process the CSV data
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded {csv_file}")
        print(f"Original shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Data quality check
        print(f"\nData Quality Summary:")
        print(f"- Total rows: {len(df)}")
        print(f"- Missing values per column:")
        for col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
        
        # Clean the data
        df_clean = clean_bengaluru_data(df)
        
        # Save cleaned data
        cleaned_file = "bengaluru_house_data_cleaned.csv"
        df_clean.to_csv(cleaned_file, index=False)
        print(f"\nCleaned data saved as '{cleaned_file}'")
        print(f"Cleaned data shape: {df_clean.shape}")
        
        return df_clean
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None

def clean_bengaluru_data(df):
    """
    Clean and preprocess the Bengaluru house data
    """
    print("\nCleaning data...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Handle common column name variations
    column_mapping = {
        'area_type': 'area_type',
        'availability': 'availability', 
        'location': 'location',
        'size': 'size',
        'society': 'society',
        'total_sqft': 'total_sqft',
        'bath': 'bath',
        'balcony': 'balcony',
        'price': 'price'
    }
    
    # Rename columns if they exist with different cases
    for old_name in df_clean.columns:
        for standard_name in column_mapping:
            if old_name.lower().replace('_', '').replace(' ', '') == standard_name.replace('_', ''):
                df_clean = df_clean.rename(columns={old_name: standard_name})
                break
    
    # Remove duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"Removed {initial_rows - len(df_clean)} duplicate rows")
    
    # Clean total_sqft column
    def clean_sqft(sqft_str):
        if pd.isna(sqft_str):
            return np.nan
        
        sqft_str = str(sqft_str).strip()
        
        # Handle ranges like "1000-1200"
        if '-' in sqft_str:
            try:
                parts = sqft_str.split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            except:
                return np.nan
        
        # Extract numbers from strings like "1200 sqft"
        numbers = re.findall(r'\d+\.?\d*', sqft_str)
        if numbers:
            try:
                return float(numbers[0])
            except:
                return np.nan
        
        return np.nan
    
    if 'total_sqft' in df_clean.columns:
        df_clean['total_sqft'] = df_clean['total_sqft'].apply(clean_sqft)
        df_clean = df_clean.dropna(subset=['total_sqft'])
        print(f"Cleaned total_sqft column")
    
    # Extract BHK from size column
    def extract_bhk(size_str):
        if pd.isna(size_str):
            return np.nan
        
        size_str = str(size_str).upper()
        
        # Look for patterns like "2 BHK", "3BHK", "4 Bedroom"
        bhk_match = re.search(r'(\d+)\s*(?:BHK|BEDROOM)', size_str)
        if bhk_match:
            return int(bhk_match.group(1))
        
        return np.nan
    
    if 'size' in df_clean.columns:
        df_clean['bhk'] = df_clean['size'].apply(extract_bhk)
        df_clean = df_clean.dropna(subset=['bhk'])
        print(f"Extracted BHK information from size column")
    
    # Clean price column
    def clean_price(price_str):
        if pd.isna(price_str):
            return np.nan
        
        price_str = str(price_str).strip()
        
        # Remove currency symbols and "Lakh", "Crore" etc.
        price_str = re.sub(r'[₹,\s]', '', price_str)
        
        # Handle "Lakh" and "Crore" conversions
        if 'Crore' in price_str or 'crore' in price_str:
            numbers = re.findall(r'\d+\.?\d*', price_str)
            if numbers:
                return float(numbers[0]) * 100  # Convert crores to lakhs
        elif 'Lakh' in price_str or 'lakh' in price_str:
            numbers = re.findall(r'\d+\.?\d*', price_str)
            if numbers:
                return float(numbers[0])
        else:
            # Assume it's already in lakhs if no unit specified
            numbers = re.findall(r'\d+\.?\d*', price_str)
            if numbers:
                return float(numbers[0])
        
        return np.nan
    
    if 'price' in df_clean.columns:
        df_clean['price'] = df_clean['price'].apply(clean_price)
        df_clean = df_clean.dropna(subset=['price'])
        print(f"Cleaned price column")
    
    # Handle numeric columns
    numeric_columns = ['bath', 'balcony']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            df_clean = df_clean.dropna(subset=[col])
    
    # Remove outliers
    if 'total_sqft' in df_clean.columns:
        df_clean = df_clean[df_clean['total_sqft'] >= 300]  # Minimum 300 sqft
        df_clean = df_clean[df_clean['total_sqft'] <= 10000]  # Maximum 10000 sqft
    
    if 'price' in df_clean.columns:
        df_clean = df_clean[df_clean['price'] >= 10]  # Minimum 10 lakhs
        df_clean = df_clean[df_clean['price'] <= 1000]  # Maximum 1000 lakhs
    
    if 'bhk' in df_clean.columns:
        df_clean = df_clean[df_clean['bhk'] <= 10]  # Maximum 10 BHK
    
    # Create additional useful features
    if 'price' in df_clean.columns and 'total_sqft' in df_clean.columns:
        df_clean['price_per_sqft'] = (df_clean['price'] * 100000) / df_clean['total_sqft']
    
    # Clean location names
    if 'location' in df_clean.columns:
        df_clean['location'] = df_clean['location'].str.strip()
        df_clean['location'] = df_clean['location'].str.title()
    
    print(f"Data cleaning completed. Final shape: {df_clean.shape}")
    
    return df_clean

def generate_data_summary(df):
    """
    Generate a comprehensive summary of the dataset
    """
    summary = []
    summary.append("# Bengaluru House Data Summary\n")
    summary.append(f"**Dataset Shape**: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary.append("## Numeric Features Summary\n")
        summary.append(df[numeric_cols].describe().to_string())
        summary.append("\n\n")
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary.append("## Categorical Features Summary\n")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            summary.append(f"**{col}**: {unique_count} unique values\n")
            if unique_count <= 10:
                value_counts = df[col].value_counts().head(10)
                summary.append(f"Top values: {value_counts.to_dict()}\n")
            summary.append("\n")
    
    # Data quality
    summary.append("## Data Quality\n")
    summary.append(f"- **Missing Values**: {df.isnull().sum().sum()}\n")
    summary.append(f"- **Duplicate Rows**: {df.duplicated().sum()}\n")
    
    # Save summary
    with open('data_summary.md', 'w') as f:
        f.write(''.join(summary))
    
    print("Data summary saved as 'data_summary.md'")

if __name__ == "__main__":
    df = convert_and_process_data()
    if df is not None:
        generate_data_summary(df)
        print("\nData processing completed successfully!")
        print("Files generated:")
        print("- bengaluru_house_data_cleaned.csv")
        print("- data_summary.md")
    else:
        print("\nTo proceed with the project:")
        print("1. Convert your Numbers file to CSV, OR")
        print("2. Run 'python house_price_predictor.py' to use sample data")