# Testing Flask API on Windows

## Option 1: Using PowerShell (Invoke-WebRequest)

### Test Custom Prediction

```powershell
$body = @{
    latitude = -37.10
    longitude = 149.80
    elevation = 350
    aspect = 140
    slope = 14.5
    landcover = 10
    temperature = 35.0
    rain = 10.5
    NDVI = 0.25
    NBR = 0.20
    NDMI = 0.15
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri "http://localhost:5000/api/predict" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

$response.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### Get Statistics

```powershell
Invoke-WebRequest -Uri "http://localhost:5000/api/stats" | Select-Object -ExpandProperty Content
```

### Get Examples

```powershell
Invoke-WebRequest -Uri "http://localhost:5000/api/examples" | Select-Object -ExpandProperty Content
```

## Option 2: Using Python

```python
import requests
import json

# Make prediction
url = "http://localhost:5000/api/predict"
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

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
```

## Option 3: Using Browser Console

Open http://localhost:5000 and press F12, then run:

```javascript
fetch('/api/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    latitude: -37.10,
    longitude: 149.80,
    elevation: 350,
    aspect: 140,
    slope: 14.5,
    landcover: 10,
    temperature: 35.0,
    rain: 10.5,
    NDVI: 0.25,
    NBR: 0.20,
    NDMI: 0.15
  })
})
.then(r => r.json())
.then(data => console.log(data));
```

## Quick Test Script

Save as `test_api.py`:

```python
import requests

print("Testing Flask API...")

# Test prediction
response = requests.post('http://localhost:5000/api/predict', json={
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
})

print("\n=== PREDICTION RESULT ===")
result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['fire_probability']:.1%}")
print(f"Response Time: {result['response_time']}")
```

Run with: `python test_api.py`
