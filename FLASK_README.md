# Flask Wildfire Dashboard

## Quick Start

```bash
cd "c:\Users\Dinesh.S\Desktop\ml wl"
python app.py
```

Then open: http://localhost:5000

## Features

-Flask backend with real ML model predictions
- API endpoints for custom predictions
- Template rendering
- Static file serving

## API End

points

**GET /** - Dashboard interface  
**GET /api/assessments** - Get existing risk assessments  
**POST /api/predict** - Make custom prediction with ML model  
**GET /api/examples** - Get example input data  
**GET /api/stats** - Get statistics

## Example API Call

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```
