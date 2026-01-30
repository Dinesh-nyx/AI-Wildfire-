"""
Simple API Test Script for Flask Wildfire Application
Run this to test the prediction endpoint
"""

import requests
import json

def test_prediction():
    """Test the prediction API endpoint"""
    
    print("=" * 80)
    print("🔥 TESTING FLASK API - WILDFIRE PREDICTION")
    print("=" * 80)
    
    url = "http://localhost:5000/api/predict"
    
    # High risk example
    data = {
        "latitude": -37.10,
        "longitude": 149.80,
        "elevation": 350,
        "aspect": 140,
        "slope": 14.5,
        "landcover": 10,
        "temperature": 35.0,
        "rain": 10.5,
        "NDVI": 0.25,
        "NBR": 0.20,
        "NDMI": 0.15
    }
    
    print("\n📤 Sending prediction request...")
    print(f"Temperature: {data['temperature']}°C")
    print(f"Rainfall: {data['rain']}mm")
    print(f"NDVI: {data['NDVI']}")
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        result = response.json()
        
        print("\n" + "=" * 80)
        print("📥 PREDICTION RESULT")
        print("=" * 80)
        
        print(f"\n🔥 Risk Level: {result['risk_level']}")
        print(f"📊 Fire Probability: {result['fire_probability']:.1%}")
        print(f"⏱️  Response Time: {result['response_time']}")
        print(f"✅ Confidence: {result['confidence']:.1%}")
        print(f"🕐 Timestamp: {result['timestamp']}")
        
        print(f"\n⚠️  RECOMMENDED ACTIONS:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "=" * 80)
        print("✅ API TEST SUCCESSFUL!")
        print("=" * 80)
        
        return result
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to Flask server")
        print("   Make sure Flask app is running: python app.py")
        return None
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return None

if __name__ == "__main__":
    test_prediction()
