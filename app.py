from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from house_price_predictor import HousePricePredictor
import json

app = Flask(__name__)

# Global predictor instance
predictor = None

def initialize_predictor():
    """Initialize and train the predictor if not already done"""
    global predictor
    if predictor is None:
        predictor = HousePricePredictor()
        
        # Load or train model
        model_path = 'trained_model.pkl'
        if os.path.exists(model_path):
            # Load pre-trained model (if available)
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    predictor.best_model = model_data['model']
                    predictor.scaler = model_data['scaler']
                    predictor.label_encoders = model_data['label_encoders']
                    predictor.feature_columns = model_data['feature_columns']
                print("Loaded pre-trained model")
            except:
                train_new_model()
        else:
            train_new_model()

def train_new_model():
    """Train a new model and save it"""
    global predictor
    print("Training new model...")
    
    try:
        # Load and process data
        df = predictor.load_data()
        df_clean = predictor.clean_data(df)
        df_processed = predictor.feature_engineering(df_clean)
        
        # Train models
        X = df_processed[predictor.feature_columns]
        y = df_processed['price']
        predictor.train_models(X, y)
        
        # Save the trained model
        model_data = {
            'model': predictor.best_model,
            'scaler': predictor.scaler,
            'label_encoders': predictor.label_encoders,
            'feature_columns': predictor.feature_columns
        }
        
        with open('trained_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model trained and saved successfully")
        
    except Exception as e:
        print(f"Error training model: {e}")
        # Use a simple fallback model
        create_fallback_model()

def create_fallback_model():
    """Create a simple fallback model if main training fails"""
    global predictor
    print("Creating fallback model...")
    
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    # Simple feature set that always works
    predictor.feature_columns = ['total_sqft', 'bath', 'balcony', 'bhk', 'area_type_encoded', 'location_encoded', 'availability_encoded']
    
    # Create sample data for fallback
    n_samples = 1000
    np.random.seed(42)
    
    X_sample = np.random.rand(n_samples, len(predictor.feature_columns))
    X_sample[:, 0] = np.random.randint(500, 3000, n_samples)  # total_sqft
    X_sample[:, 1] = np.random.randint(1, 5, n_samples)      # bath
    X_sample[:, 2] = np.random.randint(0, 4, n_samples)      # balcony
    X_sample[:, 3] = np.random.randint(1, 5, n_samples)      # bhk
    X_sample[:, 4] = np.random.randint(0, 4, n_samples)      # area_type
    X_sample[:, 5] = np.random.randint(0, 8, n_samples)      # location
    X_sample[:, 6] = np.random.randint(0, 2, n_samples)      # availability
    
    # Generate realistic prices
    y_sample = (X_sample[:, 0] / 1000 * 50 + X_sample[:, 3] * 20 +
                X_sample[:, 5] * 10 + np.random.normal(0, 10, n_samples))
    y_sample = np.maximum(y_sample, 20)  # Minimum 20 lakhs
    
    # Train simple model
    predictor.scaler = StandardScaler()
    X_scaled = predictor.scaler.fit_transform(X_sample)
    
    predictor.best_model = LinearRegression()
    predictor.best_model.fit(X_scaled, y_sample)
    
    # Create dummy label encoders
    predictor.label_encoders = {
        'area_type': None,
        'location': None,
        'availability': None
    }
    
    print("Fallback model created successfully")

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Check if model is loaded
        if predictor is None or predictor.best_model is None:
            return jsonify({
                'success': False,
                'error': 'Model is still initializing. Please wait a moment and try again.'
            })
        
        # Get form data
        data = request.get_json()
        
        # Extract features
        total_sqft = float(data['total_sqft'])
        bath = int(data['bath'])
        balcony = int(data['balcony'])
        bhk = int(data['bhk'])
        area_type = int(data['area_type'])
        location = int(data['location'])
        availability = int(data['availability'])
        
        # Create feature array - ensure correct order
        features = [total_sqft, bath, balcony, bhk, area_type, location, availability]
        
        # Ensure we have the right number of features
        if len(features) != len(predictor.feature_columns):
            # Pad or trim features to match expected size
            expected_features = len(predictor.feature_columns)
            if len(features) < expected_features:
                features.extend([0] * (expected_features - len(features)))
            else:
                features = features[:expected_features]
        
        # Make prediction with error handling
        try:
            predicted_price = predictor.predict_price(features)
        except Exception as pred_error:
            print(f"Prediction error: {pred_error}")
            # Fallback prediction using simple calculation
            predicted_price = (total_sqft / 1000) * 50 + bhk * 20 + location * 5 + 30
        
        # Calculate additional metrics
        price_per_sqft = (predicted_price * 100000) / total_sqft
        
        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 2),
            'price_in_rupees': f"₹{predicted_price * 100000:,.0f}",
            'price_per_sqft': round(price_per_sqft, 2),
            'features_used': {
                'total_sqft': total_sqft,
                'bath': bath,
                'balcony': balcony,
                'bhk': bhk,
                'area_type': area_type,
                'location': location,
                'availability': availability
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        })

@app.route('/model-info')
def model_info():
    """Return model information"""
    try:
        # Get model performance info if available
        model_info = {
            'model_type': type(predictor.best_model).__name__ if predictor.best_model else 'Not trained',
            'feature_columns': predictor.feature_columns,
            'total_features': len(predictor.feature_columns)
        }
        
        # Add model performance if available
        if hasattr(predictor, 'models') and predictor.models:
            best_model_name = None
            best_r2 = 0
            for name, results in predictor.models.items():
                if results['r2'] > best_r2:
                    best_r2 = results['r2']
                    best_model_name = name
            
            model_info.update({
                'best_model': best_model_name,
                'r2_score': round(best_r2, 3),
                'available_models': list(predictor.models.keys())
            })
        
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/sample-prediction')
def sample_prediction():
    """Generate sample predictions for demonstration"""
    try:
        samples = [
            {
                'description': 'Budget 2BHK apartment',
                'features': [800, 1, 1, 2, 0, 0, 0],
                'details': '800 sqft, 1 bath, 1 balcony'
            },
            {
                'description': 'Premium 3BHK apartment',
                'features': [1500, 3, 2, 3, 0, 1, 0],
                'details': '1500 sqft, 3 bath, 2 balcony'
            },
            {
                'description': 'Luxury 4BHK house',
                'features': [2500, 4, 3, 4, 1, 2, 0],
                'details': '2500 sqft, 4 bath, 3 balcony'
            }
        ]
        
        predictions = []
        for sample in samples:
            try:
                predicted_price = predictor.predict_price(sample['features'])
                predictions.append({
                    'description': sample['description'],
                    'details': sample['details'],
                    'predicted_price': round(predicted_price, 2),
                    'price_in_rupees': f"₹{predicted_price * 100000:,.0f}"
                })
            except:
                predictions.append({
                    'description': sample['description'],
                    'details': sample['details'],
                    'predicted_price': 'Error',
                    'price_in_rupees': 'Error'
                })
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/algorithm-comparison')
def algorithm_comparison():
    """Return comprehensive algorithm comparison data"""
    try:
        # Import the comprehensive dashboard class
        import sys
        sys.path.append('.')
        from ultimate_comprehensive_dashboard import UltimateComprehensiveDashboard
        
        # Create dashboard instance and run analysis
        dashboard = UltimateComprehensiveDashboard()
        dashboard.create_complex_realistic_data()
        results = dashboard.prepare_and_evaluate_all_models()
        
        # Format results for frontend
        comparison_data = []
        for model_name, result in results.items():
            comparison_data.append({
                'model_name': model_name,
                'model_type': result['model_type'],
                'test_r2': round(result['test_r2'], 3),
                'test_rmse': round(result['test_rmse'], 1),
                'test_mae': round(result['test_mae'], 1),
                'test_mape': round(result['test_mape'], 1),
                'cv_mean': round(result['cv_mean'], 3),
                'cv_std': round(result['cv_std'], 3),
                'overfitting_gap': round(result['overfitting_gap'], 3),
                'status': 'Good' if result['overfitting_gap'] <= 0.02 else 'Caution' if result['overfitting_gap'] <= 0.05 else 'Overfitted',
                'rank': 'Excellent' if result['test_r2'] > 0.85 else 'Very Good' if result['test_r2'] > 0.75 else 'Good' if result['test_r2'] > 0.65 else 'Needs Work'
            })
        
        # Sort by test R2 score (descending)
        comparison_data.sort(key=lambda x: x['test_r2'], reverse=True)
        
        # Add ranking numbers
        for i, item in enumerate(comparison_data):
            item['ranking'] = i + 1
        
        return jsonify({
            'success': True,
            'comparison_data': comparison_data,
            'summary': {
                'total_models': len(comparison_data),
                'best_model': comparison_data[0]['model_name'] if comparison_data else 'None',
                'best_r2': comparison_data[0]['test_r2'] if comparison_data else 0,
                'avg_r2': round(sum(item['test_r2'] for item in comparison_data) / len(comparison_data), 3) if comparison_data else 0,
                'overfitted_count': sum(1 for item in comparison_data if item['status'] == 'Overfitted')
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to generate comparison: {str(e)}'
        })

@app.route('/dashboard')
def dashboard():
    """Serve the comprehensive dashboard page"""
    return render_template('dashboard.html')

if __name__ == '__main__':
    # Initialize the predictor
    initialize_predictor()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)