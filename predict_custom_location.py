"""
Interactive Wildfire Prediction Tool
Allows custom location input and real-time predictions
"""

import pandas as pd
import numpy as np
import joblib
import json

class WildfirePredictor:
    def __init__(self):
        print("=" * 80)
        print("🔥 WILDFIRE RISK PREDICTION TOOL")
        print("=" * 80)
        
        # Load models
        print("\nLoading models...")
        self.global_model = joblib.load('wildfire_best_model.pkl')
        self.regional_model = joblib.load('regional_wildfire_model.pkl')
        self.scaler = joblib.load('wildfire_scaler.pkl')
        self.feature_names = joblib.load('wildfire_feature_names.pkl')
        print("✓ Models loaded successfully")
        
        # Risk levels
        self.risk_levels = {
            'LOW': (0.0, 0.20),
            'MODERATE': (0.20, 0.40),
            'HIGH': (0.40, 0.60),
            'CRITICAL': (0.60, 0.80),
            'EXTREME': (0.80, 1.01)
        }
    
    def assess_risk(self, probability):
        """Determine risk level"""
        for level, (low, high) in self.risk_levels.items():
            if low <= probability < high:
                return level
        return 'EXTREME'
    
    def get_response_time(self, risk_level):
        """Get response time for risk level"""
        times = {
            'EXTREME': 'IMMEDIATE (0-5 minutes)',
            'CRITICAL': 'URGENT (5-15 minutes)',
            'HIGH': '15-30 minutes',
            'MODERATE': '30-60 minutes',
            'LOW': 'Routine surveillance'
        }
        return times.get(risk_level, 'N/A')
    
    def predict(self, input_data):
        """Make prediction on custom input"""
        # Convert to DataFrame if dict
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data
        
        # Ensure correct feature order
        df = df[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Get predictions
        prediction = self.global_model.predict(X_scaled)[0]
        probability = self.global_model.predict_proba(X_scaled)[0][1]
        
        # Assess risk
        risk_level = self.assess_risk(probability)
        
        result = {
            'prediction': 'FIRE' if prediction == 1 else 'NO FIRE',
            'fire_probability': probability,
            'risk_level': risk_level,
            'response_time': self.get_response_time(risk_level),
            'confidence': max(self.global_model.predict_proba(X_scaled)[0])
        }
        
        return result
    
    def interactive_mode(self):
        """Interactive prediction mode"""
        print("\n" + "=" * 80)
        print("INTERACTIVE PREDICTION MODE")
        print("=" * 80)
        print("\nEnter location and environmental data for wildfire risk assessment")
        print("(Press Ctrl+C to exit)\n")
        
        while True:
            try:
                print("-" * 80)
                print("\n📍 Enter Location Data:")
                
                # Get user input
                data = {}
                
                print("\n🌍 Geographic Data:")
                data['latitude'] = float(input("  Latitude (e.g., -37.11): "))
                data['longitude'] = float(input("  Longitude (e.g., 149.79): "))
                data['elevation'] = int(input("  Elevation in meters (e.g., 350): "))
                
                print("\n🏞️ Terrain Data:")
                data['aspect'] = float(input("  Aspect/Direction (0-360°, e.g., 140): "))
                data['slope'] = float(input("  Slope in degrees (e.g., 14.5): "))
                data['landcover'] = int(input("  Landcover type (1-20, e.g., 10): "))
                
                print("\n🌡️ Environmental Data:")
                data['temperature'] = float(input("  Temperature in °C (e.g., 25.5): "))
                data['rain'] = float(input("  Rainfall in mm (e.g., 40.3): "))
                
                print("\n🌿 Vegetation Indices:")
                data['NDVI'] = float(input("  NDVI (-1 to 1, e.g., 0.73): "))
                data['NBR'] = float(input("  NBR (-1 to 1, e.g., 0.54): "))
                data['NDMI'] = float(input("  NDMI (-1 to 1, e.g., 0.24): "))
                
                # Make prediction
                print("\n🔄 Analyzing data...")
                result = self.predict(data)
                
                # Display results
                self.display_result(result, data)
                
                # Ask to continue
                cont = input("\n\nMake another prediction? (y/n): ").lower()
                if cont != 'y':
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again with valid values.\n")
    
    def display_result(self, result, input_data):
        """Display prediction result"""
        print("\n" + "=" * 80)
        print("🔥 WILDFIRE RISK ASSESSMENT RESULT")
        print("=" * 80)
        
        # Risk level with color
        risk_colors = {
            'EXTREME': '🔴',
            'CRITICAL': '🟠',
            'HIGH': '🟡',
            'MODERATE': '🟤',
            'LOW': '🟢'
        }
        
        risk_icon = risk_colors.get(result['risk_level'], '⚪')
        
        print(f"\n{risk_icon} RISK LEVEL: {result['risk_level']}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Fire Probability: {result['fire_probability']:.1%}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Response Time: {result['response_time']}")
        
        print(f"\n📍 Location:")
        print(f"   Latitude: {input_data['latitude']:.4f}")
        print(f"   Longitude: {input_data['longitude']:.4f}")
        print(f"   Elevation: {input_data['elevation']}m")
        
        print(f"\n🌡️ Environmental Conditions:")
        print(f"   Temperature: {input_data['temperature']}°C")
        print(f"   Rainfall: {input_data['rain']}mm")
        print(f"   NDVI (Vegetation): {input_data['NDVI']:.2f}")
        
        # Recommendations based on risk
        if result['risk_level'] in ['HIGH', 'CRITICAL', 'EXTREME']:
            print(f"\n⚠️  RECOMMENDED ACTIONS:")
            if result['risk_level'] == 'EXTREME':
                print("   • IMMEDIATE EVACUATION of all residents")
                print("   • DEPLOY all firefighting resources")
                print("   • ACTIVATE emergency response teams")
            elif result['risk_level'] == 'CRITICAL':
                print("   • ALERT firefighting teams to standby")
                print("   • PREPARE evacuation routes")
                print("   • MONITOR weather conditions")
            else:  # HIGH
                print("   • INCREASE surveillance frequency")
                print("   • PREPARE equipment and personnel")
                print("   • NOTIFY local authorities")
        
        print("=" * 80)
    
    def quick_predict(self, location_name, lat, lon, temp, rain, ndvi):
        """Quick prediction with minimal inputs"""
        # Use reasonable defaults
        data = {
            'latitude': lat,
            'longitude': lon,
            'elevation': 350,  # default
            'aspect': 140,     # default
            'slope': 14.5,     # default
            'landcover': 10,   # default
            'temperature': temp,
            'rain': rain,
            'NDVI': ndvi,
            'NBR': 0.54,       # default
            'NDMI': 0.24       # default
        }
        
        result = self.predict(data)
        
        print(f"\n📍 {location_name}")
        print(f"   Risk: {result['risk_level']} ({result['fire_probability']:.1%})")
        
        return result


def main():
    """Main function"""
    predictor = WildfirePredictor()
    
    print("\n" + "=" * 80)
    print("PREDICTION OPTIONS")
    print("=" * 80)
    print("\n1. Interactive Mode - Enter custom location data")
    print("2. Quick Examples - Pre-defined test locations")
    print("3. Batch Mode - Load data from CSV file")
    
    choice = input("\nSelect option (1/2/3): ").strip()
    
    if choice == '1':
        predictor.interactive_mode()
    
    elif choice == '2':
        print("\n" + "=" * 80)
        print("QUICK PREDICTION EXAMPLES")
        print("=" * 80)
        
        # Example locations
        examples = [
            ("High Risk Area", -37.10, 149.80, 35.0, 10.5, 0.25),
            ("Moderate Risk Area", -37.12, 149.75, 22.0, 35.0, 0.55),
            ("Low Risk Area", -37.08, 149.85, 18.0, 50.0, 0.75),
        ]
        
        for name, lat, lon, temp, rain, ndvi in examples:
            predictor.quick_predict(name, lat, lon, temp, rain, ndvi)
    
    elif choice == '3':
        csv_file = input("\nEnter CSV file path: ").strip()
        try:
            df = pd.read_csv(csv_file)
            print(f"\n✓ Loaded {len(df)} locations from {csv_file}")
            
            results = []
            for idx, row in df.iterrows():
                result = predictor.predict(row)
                results.append({
                    'location_id': idx + 1,
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'risk_level': result['risk_level'],
                    'fire_probability': result['fire_probability']
                })
            
            # Save results
            results_df = pd.DataFrame(results)
            output_file = 'batch_predictions.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\n✓ Saved predictions to {output_file}")
            
            # Summary
            print("\n" + "=" * 80)
            print("BATCH PREDICTION SUMMARY")
            print("=" * 80)
            for level in ['EXTREME', 'CRITICAL', 'HIGH', 'MODERATE', 'LOW']:
                count = (results_df['risk_level'] == level).sum()
                if count > 0:
                    print(f"  {level}: {count} locations")
                    
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("Thank you for using the Wildfire Risk Prediction Tool!")
    print("=" * 80)


if __name__ == "__main__":
    main()
