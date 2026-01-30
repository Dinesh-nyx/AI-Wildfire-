"""
Generate Realistic Victoria, Australia Wildfire Risk Data
Creates 50 sample locations with proper risk distribution
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("GENERATING VICTORIA, AUSTRALIA WILDFIRE RISK DATA")
print("=" * 80)

# Known fire-prone regions in Victoria
regions = {
    'East Gippsland': {'lat': (-38.0, -37.0), 'lon': (148.0, 149.5), 'base_risk': 0.65},
    'Alpine National Park': {'lat': (-37.3, -36.5), 'lon': (146.5, 147.5), 'base_risk': 0.55},
    'Snowy River': {'lat': (-37.5, -37.0), 'lon': (148.5, 149.2), 'base_risk': 0.60},
    'High Country': {'lat': (-37.0, -36.0), 'lon': (146.0, 147.0), 'base_risk': 0.50},
    'Central Victoria': {'lat': (-37.5, -37.0), 'lon': (144.0, 145.0), 'base_risk': 0.35},
    'Western Victoria': {'lat': (-38.0, -37.0), 'lon': (142.0, 143.5), 'base_risk': 0.25}
}

# Generate samples
np.random.seed(42)
samples = []
sample_id = 1

for region_name, params in regions.items():
    # Number of samples per region (total ~50)
    n_samples = 8 if 'base_risk' in params and params['base_risk'] > 0.5 else 6
    
    for _ in range(n_samples):
        # Random location within region
        lat = np.random.uniform(params['lat'][0], params['lat'][1])
        lon = np.random.uniform(params['lon'][0], params['lon'][1])
        
        # Environmental parameters
        elevation = int(np.random.uniform(100, 1500))
        
        # Temperature: higher in summer, varies by elevation
        temp = np.random.uniform(18, 42) - (elevation / 200)
        
        # Rainfall: varies by region and season
        rain = np.random.uniform(5, 80)
        
        # NDVI: vegetation health (lower = drier, higher risk)
        ndvi = np.random.uniform(0.1, 0.8)
        
        # Calculate fire probability based on conditions
        # Base risk from region
        prob = params['base_risk']
        
        # Temperature factor (higher temp = higher risk)
        if temp > 35:
            prob += 0.15
        elif temp > 30:
            prob += 0.10
        elif temp > 25:
            prob += 0.05
        
        # Rainfall factor (lower rain = higher risk)
        if rain < 15:
            prob += 0.15
        elif rain < 30:
            prob += 0.10
        elif rain < 45:
            prob += 0.05
        else:
            prob -= 0.10
        
        # Vegetation factor (lower NDVI = higher risk)
        if ndvi < 0.3:
            prob += 0.10
        elif ndvi < 0.5:
            prob += 0.05
        elif ndvi > 0.7:
            prob -= 0.10
        
        # Add some randomness
        prob += np.random.uniform(-0.1, 0.1)
        
        # Clamp probability
        prob = max(0.05, min(0.95, prob))
        
        # Determine risk level
        if prob >= 0.80:
            risk_level = 'EXTREME'
            priority = 1
            response = 'IMMEDIATE (0-5 minutes)'
        elif prob >= 0.60:
            risk_level = 'CRITICAL'
            priority = 2
            response = 'URGENT (5-15 minutes)'
        elif prob >= 0.40:
            risk_level = 'HIGH'
            priority = 3
            response = '15-30 minutes'
        elif prob >= 0.20:
            risk_level = 'MODERATE'
            priority = 4
            response = '30-60 minutes'
        else:
            risk_level = 'LOW'
            priority = 5
            response = 'Routine surveillance'
        
        # Prediction
        prediction = 'FIRE' if prob >= 0.5 else 'NO FIRE'
        
        sample = {
            'sample_id': sample_id,
            'region': region_name,
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'elevation': elevation,
            'temperature': round(temp, 2),
            'rain': round(rain, 2),
            'NDVI': round(ndvi, 4),
            'prediction': prediction,
            'fire_probability': round(prob, 4),
            'risk_level': risk_level,
            'priority': priority,
            'response_time': response
        }
        
        samples.append(sample)
        sample_id += 1

# Create DataFrame
df = pd.DataFrame(samples)

# Display summary
print(f"\nGenerated {len(df)} samples across Victoria, Australia")
print(f"\nGeographic extent:")
print(f"  Latitude: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
print(f"  Longitude: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")

print(f"\nRisk distribution:")
for level in ['EXTREME', 'CRITICAL', 'HIGH', 'MODERATE', 'LOW']:
    count = (df['risk_level'] == level).sum()
    pct = (count / len(df)) * 100
    print(f"  {level}: {count} ({pct:.1f}%)")

print(f"\nFire probability stats:")
print(df['fire_probability'].describe())

# Save to CSV
output_file = 'risk_assessment_results.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ Saved to: {output_file}")

print("\n" + "=" * 80)
print("DATA GENERATION COMPLETE!")
print("=" * 80)
