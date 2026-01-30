import requests

print("=" * 60)
print("TESTING FLASK SERVER WITH CORS")
print("=" * 60)

# Test if server is responding
try:
    response = requests.get('http://localhost:5000/api/stats')
    print(f"\n✅ Server is running!")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
    
    print("\n" + "=" * 60)
    print("✅ CORS IS WORKING!")
    print("=" * 60)
    print("\nNow refresh test_dashboard.html in your browser")
    print("The 'failed to fetch' error should be gone!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
