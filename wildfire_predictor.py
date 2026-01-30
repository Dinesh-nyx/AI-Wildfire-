"""
Wildfire Early Detection - Prediction Script
Use this script to make predictions on new data using the trained model
"""

import pandas as pd
import numpy as np
import joblib
import json

def load_model():
    """Load the trained model, scaler, and feature names"""
    try:
        model = joblib.load('wildfire_best_model.pkl')
        scaler = joblib.load('wildfire_scaler.pkl')
        feature_names = joblib.load('wildfire_feature_names.pkl')
        
        with open('wildfire_model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print("✓ Model loaded successfully")
        print(f"✓ Model type: {metadata['best_model']}")
        print(f"✓ Features: {len(feature_names)}")
        
        return model, scaler, feature_names, metadata
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run wildfire_detection_training.py first to train the model.")
        return None, None, None, None

def predict_wildfire(data, model, scaler, feature_names):
    """
    Make wildfire predictions on new data
    
    Parameters:
    -----------
    data : pd.DataFrame or dict
        Input data with features
    model : trained model
        The trained wildfire detection model
    scaler : StandardScaler
        The fitted scaler
    feature_names : list
        List of expected feature names
    
    Returns:
    --------
    predictions : dict
        Dictionary with predictions and probabilities
    """
    # Convert dict to DataFrame if necessary
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Ensure all required features are present
    missing_features = set(feature_names) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Select and order features correctly
    data = data[feature_names]
    
    # Scale features
    data_scaled = scaler.transform(data)
    
    # Make predictions
    predictions = model.predict(data_scaled)
    probabilities = model.predict_proba(data_scaled)
    
    # Prepare results
    results = []
    for i in range(len(data)):
        risk_level = "HIGH RISK" if probabilities[i][1] > 0.7 else "MODERATE RISK" if probabilities[i][1] > 0.4 else "LOW RISK"
        
        results.append({
            'prediction': 'FIRE' if predictions[i] == 1 else 'NO FIRE',
            'fire_probability': probabilities[i][1],
            'no_fire_probability': probabilities[i][0],
            'risk_level': risk_level
        })
    
    return results

def main():
    """Main function for interactive prediction"""
    print("=" * 80)
    print("WILDFIRE EARLY DETECTION - PREDICTION SYSTEM")
    print("=" * 80)
    
    # Load model
    model, scaler, feature_names, metadata = load_model()
    if model is None:
        return
    
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 80)
    for metric, value in metadata['performance_metrics'].items():
        if metric not in ['y_pred', 'y_pred_proba']:
            print(f"{metric}: {value:.4f}")
    
    # Example prediction
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTION")
    print("=" * 80)
    
    # Load a sample from the test data
    try:
        df = pd.read_csv('fire_before_perpixel_2019_cleaned.csv')
        sample_data = df.drop(['fire'], axis=1).iloc[0:3]
        
        print("\nMaking predictions on sample data...")
        results = predict_wildfire(sample_data, model, scaler, feature_names)
        
        for i, result in enumerate(results):
            print(f"\nSample {i+1}:")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Fire Probability: {result['fire_probability']:.2%}")
            print(f"  Risk Level: {result['risk_level']}")
    
    except FileNotFoundError:
        print("No test data found. Please provide data for prediction.")
    
    print("\n" + "=" * 80)
    print("To use this predictor in your code:")
    print("=" * 80)
    print("""
from wildfire_predictor import load_model, predict_wildfire

# Load model
model, scaler, feature_names, metadata = load_model()

# Prepare your data (dictionary or DataFrame)
new_data = {
    'NBR': 0.54,
    'NDMI': 0.24,
    'NDVI': 0.73,
    'aspect': 140.0,
    'elevation': 348,
    'landcover': 10,
    'rain': 40.3,
    'slope': 14.2,
    'temperature': 15.5,
    'latitude': -37.15,
    'longitude': 149.70
}

# Make prediction
results = predict_wildfire(new_data, model, scaler, feature_names)
print(results)
    """)

if __name__ == "__main__":
    main()
