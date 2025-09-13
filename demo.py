"""
House Price Prediction Demo
This script demonstrates the complete house price prediction pipeline
"""

from house_price_predictor import HousePricePredictor
import pandas as pd
import numpy as np

def run_demo():
    """Run a complete demonstration of the house price prediction system"""
    print("🏠 Bengaluru House Price Prediction System Demo")
    print("=" * 60)
    
    # Initialize the predictor
    predictor = HousePricePredictor()
    
    # Load data (will use sample data if CSV not available)
    print("\n1. Loading Data...")
    df = predictor.load_data()
    
    if df is not None:
        print(f"   ✓ Data loaded successfully! Shape: {df.shape}")
        
        # Clean data
        print("\n2. Cleaning Data...")
        df_clean = predictor.clean_data(df)
        print(f"   ✓ Data cleaned! Final shape: {df_clean.shape}")
        
        # Feature engineering
        print("\n3. Engineering Features...")
        df_processed = predictor.feature_engineering(df_clean)
        print(f"   ✓ Features engineered! Features: {predictor.feature_columns}")
        
        # Prepare data for training
        X = df_processed[predictor.feature_columns]
        y = df_processed['price']
        
        # Train models
        print("\n4. Training Machine Learning Models...")
        print("   Training multiple models (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)...")
        X_train, X_test, y_train, y_test = predictor.train_models(X, y)
        print("   ✓ All models trained successfully!")
        
        # Generate results
        print("\n5. Generating Results...")
        predictor.plot_results()
        predictor.save_model_report()
        print("   ✓ Visualizations and report generated!")
        
        # Demo predictions
        print("\n6. Example Predictions...")
        demo_predictions(predictor)
        
        print("\n" + "="*60)
        print("✅ Demo completed successfully!")
        print("\nGenerated files:")
        print("   📊 model_performance.png - Model performance visualizations")
        print("   📄 model_report.md - Detailed model performance report")
        print("\nTo make predictions, use predictor.predict_price([features])")
        
    else:
        print("   ❌ Unable to load data. Please convert Numbers file to CSV first.")

def demo_predictions(predictor):
    """Demonstrate different house price predictions"""
    print("   Making sample predictions...")
    
    # Example houses with different characteristics
    examples = [
        {
            'description': 'Budget 2BHK apartment (800 sqft, 1 bath, 1 balcony)',
            'features': [800, 1, 1, 2, 0, 0, 0]  # sqft, bath, balcony, bhk, area_type, location, availability
        },
        {
            'description': 'Premium 3BHK apartment (1500 sqft, 3 bath, 2 balcony)',
            'features': [1500, 3, 2, 3, 0, 1, 0]
        },
        {
            'description': 'Luxury 4BHK house (2500 sqft, 4 bath, 3 balcony)',
            'features': [2500, 4, 3, 4, 1, 2, 0]
        }
    ]
    
    for example in examples:
        try:
            if len(example['features']) == len(predictor.feature_columns):
                predicted_price = predictor.predict_price(example['features'])
                print(f"   • {example['description']}")
                print(f"     Predicted Price: ₹{predicted_price:.2f} lakhs")
            else:
                print(f"   • {example['description']}")
                print(f"     Note: Feature mismatch, skipping prediction")
        except Exception as e:
            print(f"   • Error predicting {example['description']}: {e}")

def interactive_prediction():
    """Allow user to input custom house features for prediction"""
    print("\n" + "="*60)
    print("🔮 Interactive House Price Prediction")
    print("="*60)
    
    predictor = HousePricePredictor()
    
    # Load and train model with sample data
    df = predictor.load_data()
    if df is not None:
        df_clean = predictor.clean_data(df)
        df_processed = predictor.feature_engineering(df_clean)
        X = df_processed[predictor.feature_columns]
        y = df_processed['price']
        predictor.train_models(X, y)
        
        print("Model trained! You can now make predictions.")
        print("Enter house details (press Enter for default values):")
        
        try:
            # Get user input
            sqft = input("Total Square Feet (default: 1200): ").strip()
            sqft = float(sqft) if sqft else 1200
            
            bath = input("Number of Bathrooms (default: 2): ").strip()
            bath = int(bath) if bath else 2
            
            balcony = input("Number of Balconies (default: 1): ").strip()
            balcony = int(balcony) if balcony else 1
            
            bhk = input("Number of BHK (default: 2): ").strip()
            bhk = int(bhk) if bhk else 2
            
            print("\nArea Type:")
            print("0: Super built-up Area, 1: Plot Area, 2: Built-up Area, 3: Carpet Area")
            area_type = input("Select area type (default: 0): ").strip()
            area_type = int(area_type) if area_type else 0
            
            print("\nLocation (encoded):")
            print("0-7: Different locations in Bengaluru")
            location = input("Select location (default: 0): ").strip()
            location = int(location) if location else 0
            
            availability = 0  # Default to "Ready to Move"
            
            # Make prediction
            features = [sqft, bath, balcony, bhk, area_type, location, availability]
            if len(features) == len(predictor.feature_columns):
                predicted_price = predictor.predict_price(features)
                
                print(f"\n🏠 House Details:")
                print(f"   • Size: {sqft} sqft, {bhk} BHK")
                print(f"   • Bathrooms: {bath}, Balconies: {balcony}")
                print(f"   • Area Type: {area_type}, Location: {location}")
                print(f"\n💰 Predicted Price: ₹{predicted_price:.2f} lakhs")
                print(f"   (Approximately ₹{predicted_price*100000:,.0f})")
            else:
                print("Error: Feature count mismatch")
                
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Unable to load data for training.")

if __name__ == "__main__":
    run_demo()
    
    # Optionally run interactive prediction
    response = input("\nWould you like to try interactive prediction? (y/n): ").strip().lower()
    if response == 'y':
        interactive_prediction()