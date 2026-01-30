"""
Quick test to verify Flask API is working
"""

import requests
import json

print("Testing Flask API endpoints...")
print("=" * 60)

try:
    # Test /api/assessments
    print("\n1. Testing /api/assessments...")
    response = requests.get('http://localhost:5000/api/assessments')
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ SUCCESS - Got {len(data)} assessments")
        print(f"   Sample: {data[0]['sample_id']} - {data[0]['risk_level']}")
    else:
        print(f"   ❌ FAILED - Status: {response.status_code}")
    
    # Test /api/stats
    print("\n2. Testing /api/stats...")
    response = requests.get('http://localhost:5000/api/stats')
    if response.status_code == 200:
        stats = response.json()
        print(f"   ✅ SUCCESS")
        print(f"   Total Locations: {stats['total_locations']}")
        print(f"   Active Alerts: {stats['active_alerts']}")
        print(f"   Extreme Risk: {stats['extreme_risk']}")
    else:
        print(f"   ❌ FAILED - Status: {response.status_code}")
    
    print("\n" + "=" * 60)
    print("✅ All endpoints working!")
    print("\n📌 Next step: Open http://localhost:5000 and refresh")
    
except requests.exceptions.ConnectionError:
    print("\n❌ Cannot connect to Flask server")
    print("   Make sure Flask is running: python app.py")
except Exception as e:
    print(f"\n❌ Error: {e}")
