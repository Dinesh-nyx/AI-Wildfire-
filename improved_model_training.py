"""
Improved Wildfire Model Training for Victoria, Australia
Features:
- Enhanced Random Forest with optimized parameters
- Feature engineering for better predictions
- Better risk stratification
- Cross-validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("IMPROVED WILDFIRE MODEL TRAINING - VICTORIA, AUSTRALIA")
print("=" * 80)

# Load the Victoria-specific data
print("\n[1/7] Loading Victoria wildfire data...")
df = pd.read_csv('risk_assessment_results.csv')
print(f"✓ Loaded {len(df)} samples")
print(f"  Geographic range: Lat {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
print(f"  Geographic range: Lon {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")

# Feature Engineering
print("\n[2/7] Engineering features...")

# Create interaction features
df['temp_rain_ratio'] = df['temperature'] / (df['rain'] + 1)
df['vegetation_stress'] = df['temperature'] / (df['NDVI'] + 0.1)
df['elevation_risk'] = df['elevation'] / 1000
df['drought_index'] = (60 - df['rain']) / 60
df['heat_index'] = df['temperature'] * (1 - df['NDVI'])

# Seasonal approximation (based on latitude - closer to equator = warmer)
df['latitude_factor'] = abs(df['latitude'] + 37.5) / 5  # Normalized around Victoria center

print(f"✓ Created 6 engineered features")
print(f"  Total features: {len(df.columns)}")

# Prepare features and target
print("\n[3/7] Preparing training data...")

# Select features for training
feature_cols = [
    'latitude', 'longitude', 'elevation', 'temperature', 'rain', 'NDVI',
    'temp_rain_ratio', 'vegetation_stress', 'elevation_risk', 
    'drought_index', 'heat_index', 'latitude_factor'
]

X = df[feature_cols]
y = (df['fire_probability'] >= 0.5).astype(int)  # Binary classification

print(f"✓ Features: {len(feature_cols)}")
print(f"✓ Samples: {len(X)}")
print(f"✓ Fire samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"✓ No-fire samples: {(~y.astype(bool)).sum()} ({(~y.astype(bool)).sum()/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# Train improved model
print("\n[4/7] Training improved Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,        # More trees
    max_depth=25,            # Deeper trees
    min_samples_split=3,     # More granular splits
    min_samples_leaf=1,      # Allow finer leaves
    class_weight='balanced', # Handle imbalanced data
    random_state=42,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train_scaled, y_train)
print("✓ Model trained successfully")

# Cross-validation
print("\n[5/7] Performing cross-validation...")
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"✓ Cross-validation scores: {cv_scores}")
print(f"✓ Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Evaluate model
print("\n[6/7] Evaluating model performance...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✓ Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Fire', 'Fire']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 5 Most Important Features:")
print(feature_importance.head())

# Save model and artifacts
print("\n[7/7] Saving model and artifacts...")

# Save model
joblib.dump(model, 'victoria_wildfire_model.pkl')
print("✓ Saved: victoria_wildfire_model.pkl")

# Save scaler
joblib.dump(scaler, 'victoria_wildfire_scaler.pkl')
print("✓ Saved: victoria_wildfire_scaler.pkl")

# Save feature names
joblib.dump(feature_cols, 'victoria_wildfire_features.pkl')
print("✓ Saved: victoria_wildfire_features.pkl")

# Save metadata
metadata = {
    'accuracy': float(accuracy),
    'cv_mean': float(cv_scores.mean()),
    'cv_std': float(cv_scores.std()),
    'n_samples': len(df),
    'n_features': len(feature_cols),
    'feature_names': feature_cols,
    'model_type': 'RandomForestClassifier',
    'region': 'Victoria, Australia'
}

import json
with open('victoria_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Saved: victoria_model_metadata.json")

# Create visualizations
print("\nCreating visualizations...")

# 1. Feature Importance Plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - Victoria Wildfire Model')
plt.tight_layout()
plt.savefig('victoria_feature_importance.png', dpi=150, bbox_inches='tight')
print("✓ Saved: victoria_feature_importance.png")
plt.close()

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Fire', 'Fire'],
            yticklabels=['No Fire', 'Fire'])
plt.title('Confusion Matrix - Victoria Wildfire Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('victoria_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: victoria_confusion_matrix.png")
plt.close()

# 3. Risk Distribution
plt.figure(figsize=(10, 6))
risk_counts = df['risk_level'].value_counts()
colors = {'EXTREME': '#d32f2f', 'CRITICAL': '#f57c00', 'HIGH': '#fbc02d', 
          'MODERATE': '#388e3c', 'LOW': '#1976d2'}
plt.bar(risk_counts.index, risk_counts.values, 
        color=[colors.get(x, '#666') for x in risk_counts.index])
plt.xlabel('Risk Level')
plt.ylabel('Count')
plt.title('Risk Distribution - Victoria Wildfire Data')
plt.tight_layout()
plt.savefig('victoria_risk_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: victoria_risk_distribution.png")
plt.close()

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\n✅ Final Accuracy: {accuracy*100:.2f}%")
print(f"✅ Cross-validation: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*200:.2f}%)")
print(f"✅ All files saved successfully")
print(f"✅ Ready for deployment!")
print("\n" + "=" * 80)
