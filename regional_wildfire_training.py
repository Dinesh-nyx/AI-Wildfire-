"""
Regional Wildfire Analysis and Model Training
Identifies the 4 regions with highest fire occurrences and trains a specialized model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("REGIONAL WILDFIRE ANALYSIS AND MODEL TRAINING")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading datasets...")
df_before = pd.read_csv('fire_before_perpixel_2019_cleaned.csv')
df_after = pd.read_csv('fire_after_perpixel_2020_cleaned.csv')

# Combine datasets
df = pd.concat([df_before, df_after], axis=0, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Total samples: {len(df):,}")
print(f"Fire samples: {df['fire'].sum():,}")
print(f"No-fire samples: {(df['fire']==0).sum():,}")

# ============================================================================
# 2. SPATIAL ANALYSIS - IDENTIFY FIRE REGIONS
# ============================================================================
print("\n[2/6] Performing spatial analysis...")

# Create spatial grid (divide area into cells)
lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
lon_min, lon_max = df['longitude'].min(), df['longitude'].max()

print(f"\nGeographic extent:")
print(f"  Latitude: {lat_min:.4f} to {lat_max:.4f}")
print(f"  Longitude: {lon_min:.4f} to {lon_max:.4f}")

# Create grid cells (10x10 = 100 cells)
n_lat_bins = 10
n_lon_bins = 10

df['lat_bin'] = pd.cut(df['latitude'], bins=n_lat_bins, labels=False)
df['lon_bin'] = pd.cut(df['longitude'], bins=n_lon_bins, labels=False)
df['region_id'] = df['lat_bin'] * n_lon_bins + df['lon_bin']

# Count fires per region
fire_data = df[df['fire'] == 1].copy()
region_fire_counts = fire_data.groupby('region_id').size().sort_values(ascending=False)

print(f"\nTotal regions: {df['region_id'].nunique()}")
print(f"\nTop 10 regions by fire count:")
print(region_fire_counts.head(10))

# Select top 4 regions with highest fire occurrences
top_4_regions = region_fire_counts.head(4).index.tolist()
print(f"\n🔥 Top 4 fire-prone regions: {top_4_regions}")

# Get region boundaries for visualization
region_info = []
for region_id in top_4_regions:
    region_data = df[df['region_id'] == region_id]
    fire_count = region_data['fire'].sum()
    total_count = len(region_data)
    
    info = {
        'region_id': region_id,
        'fire_count': fire_count,
        'total_samples': total_count,
        'fire_percentage': (fire_count / total_count * 100) if total_count > 0 else 0,
        'lat_min': region_data['latitude'].min(),
        'lat_max': region_data['latitude'].max(),
        'lon_min': region_data['longitude'].min(),
        'lon_max': region_data['longitude'].max(),
        'lat_center': region_data['latitude'].mean(),
        'lon_center': region_data['longitude'].mean()
    }
    region_info.append(info)

region_df = pd.DataFrame(region_info)
print("\n" + "=" * 80)
print("TOP 4 FIRE-PRONE REGIONS - DETAILS")
print("=" * 80)
print(region_df.to_string(index=False))

# ============================================================================
# 3. VISUALIZE SPATIAL DISTRIBUTION
# ============================================================================
print("\n[3/6] Creating visualizations...")

# Create spatial visualization
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: All fires
ax1 = axes[0]
fire_samples = df[df['fire'] == 1]
no_fire_samples = df[df['fire'] == 0].sample(n=min(5000, len(df[df['fire'] == 0])), random_state=42)

ax1.scatter(no_fire_samples['longitude'], no_fire_samples['latitude'], 
           c='lightblue', alpha=0.3, s=1, label='No Fire (sample)')
ax1.scatter(fire_samples['longitude'], fire_samples['latitude'], 
           c='red', alpha=0.5, s=2, label='Fire')

# Highlight top 4 regions
colors = ['gold', 'orange', 'darkred', 'purple']
for idx, region_id in enumerate(top_4_regions):
    region_data = df[df['region_id'] == region_id]
    lat_range = [region_data['latitude'].min(), region_data['latitude'].max()]
    lon_range = [region_data['longitude'].min(), region_data['longitude'].max()]
    
    # Draw rectangle for region
    from matplotlib.patches import Rectangle
    rect = Rectangle((lon_range[0], lat_range[0]), 
                     lon_range[1] - lon_range[0], 
                     lat_range[1] - lat_range[0],
                     linewidth=3, edgecolor=colors[idx], 
                     facecolor='none', label=f'Region {region_id}')
    ax1.add_patch(rect)

ax1.set_xlabel('Longitude', fontsize=12)
ax1.set_ylabel('Latitude', fontsize=12)
ax1.set_title('Fire Distribution - Top 4 Fire-Prone Regions Highlighted', 
             fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(alpha=0.3)

# Plot 2: Fire density heatmap
ax2 = axes[1]
pivot_data = df[df['fire'] == 1].groupby(['lat_bin', 'lon_bin']).size().reset_index(name='count')
heatmap_data = pivot_data.pivot(index='lat_bin', columns='lon_bin', values='count').fillna(0)

sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Fire Count'})
ax2.set_xlabel('Longitude Bin', fontsize=12)
ax2.set_ylabel('Latitude Bin', fontsize=12)
ax2.set_title('Fire Density Heatmap', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('regional_fire_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: regional_fire_analysis.png")
plt.close()

# ============================================================================
# 4. FILTER DATA FOR TOP 4 REGIONS
# ============================================================================
print("\n[4/6] Preparing regional dataset...")

# Filter data to include only top 4 regions
df_regional = df[df['region_id'].isin(top_4_regions)].copy()
print(f"\nRegional dataset size: {len(df_regional):,} samples")
print(f"Fire samples: {df_regional['fire'].sum():,}")
print(f"No-fire samples: {(df_regional['fire']==0).sum():,}")

# Remove temporary columns
df_regional = df_regional.drop(['lat_bin', 'lon_bin', 'region_id'], axis=1)

# Prepare features and target
X_regional = df_regional.drop(['fire'], axis=1)
y_regional = df_regional['fire']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_regional, y_regional, test_size=0.2, random_state=42, stratify=y_regional
)

print(f"Training samples: {len(X_train):,}")
print(f"Test samples: {len(X_test):,}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'regional_wildfire_scaler.pkl')
print("✓ Saved: regional_wildfire_scaler.pkl")

# ============================================================================
# 5. TRAIN REGIONAL MODEL
# ============================================================================
print("\n[5/6] Training regional model...")

# Train Random Forest (good balance of performance and interpretability)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest classifier...")
model.fit(X_train_scaled, y_train)
print("✓ Model trained")

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================================================
# 6. EVALUATE MODEL
# ============================================================================
print("\n[6/6] Evaluating model...")

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)
print(classification_report(y_test, y_pred, target_names=['No Fire', 'Fire']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Confusion Matrix
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['No Fire', 'Fire'],
            yticklabels=['No Fire', 'Fire'],
            cbar_kws={'label': 'Count'})
ax1.set_title('Confusion Matrix - Regional Model', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)

# Plot 2: Feature Importance
ax2 = axes[1]
feature_importance = pd.DataFrame({
    'Feature': X_regional.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

ax2.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
ax2.set_xlabel('Importance', fontsize=12)
ax2.set_title('Top 10 Feature Importance - Regional Model', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('regional_model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: regional_model_evaluation.png")
plt.close()

# Save model
joblib.dump(model, 'regional_wildfire_model.pkl')
print("✓ Saved: regional_wildfire_model.pkl")

# Save region information
region_df.to_csv('top_4_fire_regions.csv', index=False)
print("✓ Saved: top_4_fire_regions.csv")

# Save metadata
metadata = {
    'model_type': 'Random Forest',
    'n_estimators': 200,
    'accuracy': float(accuracy),
    'total_samples': len(df_regional),
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'top_regions': top_4_regions,
    'features': X_regional.columns.tolist()
}

pd.Series(metadata).to_json('regional_model_metadata.json', indent=2)
print("✓ Saved: regional_model_metadata.json")

print("\n" + "=" * 80)
print("REGIONAL MODEL TRAINING COMPLETED!")
print("=" * 80)
print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")
print(f"✅ Top 4 Fire-Prone Regions Identified: {top_4_regions}")
print(f"✅ Regional Dataset: {len(df_regional):,} samples")
print(f"✅ Model saved: regional_wildfire_model.pkl")
print(f"✅ Region details: top_4_fire_regions.csv")
print("\nFiles generated:")
print("  - regional_fire_analysis.png (spatial visualization)")
print("  - regional_model_evaluation.png (confusion matrix & feature importance)")
print("  - regional_wildfire_model.pkl (trained model)")
print("  - regional_wildfire_scaler.pkl (feature scaler)")
print("  - top_4_fire_regions.csv (region details)")
print("  - regional_model_metadata.json (model metadata)")
print("\n" + "=" * 80)
