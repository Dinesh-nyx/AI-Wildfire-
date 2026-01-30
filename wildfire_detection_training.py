"""
Wildfire Early Detection Model Training Pipeline
This script trains and evaluates multiple ML models for wildfire detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("WILDFIRE EARLY DETECTION MODEL TRAINING")
print("=" * 80)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("\n[1/7] Loading datasets...")
print("Loading fire_before_perpixel_2019_cleaned.csv (no fire samples)...")
df_before = pd.read_csv('fire_before_perpixel_2019_cleaned.csv')
print(f"Before dataset: {df_before.shape[0]:,} samples")

print("Loading fire_after_perpixel_2020_cleaned.csv (fire samples)...")
df_after = pd.read_csv('fire_after_perpixel_2020_cleaned.csv')
print(f"After dataset: {df_after.shape[0]:,} samples")

# Combine datasets
print("\nCombining datasets...")
df = pd.concat([df_before, df_after], axis=0, ignore_index=True)
print(f"Combined dataset: {df.shape[0]:,} samples")

# Shuffle the combined dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Dataset shuffled")

print(f"\nDataset shape: {df.shape}")
print(f"Total samples: {len(df):,}")
print(f"\nFeatures: {', '.join(df.columns.tolist())}")

# Check data types and missing values
print("\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print(f"\nTotal missing values: {df.isnull().sum().sum()}")

# Basic statistics
print("\n" + "=" * 80)
print("DATASET STATISTICS")
print("=" * 80)
print(df.describe())

# Check target variable distribution
print("\n" + "=" * 80)
print("TARGET VARIABLE DISTRIBUTION")
print("=" * 80)
fire_counts = df['fire'].value_counts().sort_index()
print(fire_counts)
print(fire_counts)
print(f"\nClass balance:")
print(f"No fire (0): {fire_counts[0]:,} ({fire_counts[0]/len(df)*100:.2f}%)")
print(f"Fire (1): {fire_counts[1]:,} ({fire_counts[1]/len(df)*100:.2f}%)")

# Visualize target distribution
plt.figure(figsize=(10, 6))
fire_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Fire vs No-Fire Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Fire Status (0=No Fire, 1=Fire)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('fire_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: fire_distribution.png")
plt.close()

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n[2/7] Performing Exploratory Data Analysis...")

# Correlation analysis
print("\nCalculating feature correlations...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: correlation_heatmap.png")
plt.close()

# Feature correlation with target
print("\nFeature correlation with fire:")
fire_correlation = correlation_matrix['fire'].sort_values(ascending=False)
print(fire_correlation)

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n[3/7] Preprocessing data...")

# Separate features and target
X = df.drop(['fire'], axis=1)
y = df['fire']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# Feature scaling
print("\nApplying StandardScaler to features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for later use
joblib.dump(scaler, 'wildfire_scaler.pkl')
print("✓ Saved: wildfire_scaler.pkl")

# Check if class imbalance exists (if minority class < 40%)
minority_class_ratio = min(fire_counts) / len(df)
print(f"\nMinority class ratio: {minority_class_ratio:.2%}")

if minority_class_ratio < 0.40:
    print("\nClass imbalance detected. Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"Original training set: {X_train_scaled.shape[0]:,} samples")
    print(f"Resampled training set: {X_train_resampled.shape[0]:,} samples")
    
    # Use resampled data for training
    X_train_final = X_train_resampled
    y_train_final = y_train_resampled
else:
    print("\nNo significant class imbalance. Using original data...")
    X_train_final = X_train_scaled
    y_train_final = y_train

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================
print("\n[4/7] Training models...")
print("=" * 80)

models = {}
results = {}

# 4.1 Logistic Regression
print("\n[Model 1/4] Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_final, y_train_final)
models['Logistic Regression'] = lr
print("✓ Logistic Regression trained")

# 4.2 Random Forest
print("\n[Model 2/4] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_final, y_train_final)
models['Random Forest'] = rf
print("✓ Random Forest trained")

# 4.3 XGBoost
print("\n[Model 3/4] Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)
xgb_model.fit(X_train_final, y_train_final)
models['XGBoost'] = xgb_model
print("✓ XGBoost trained")

# 4.4 LightGBM
print("\n[Model 4/4] Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train_final, y_train_final)
models['LightGBM'] = lgb_model
print("✓ LightGBM trained")

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================
print("\n[5/7] Evaluating models...")
print("=" * 80)

for model_name, model in models.items():
    print(f"\n{model_name}")
    print("-" * 40)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    # Print metrics
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================
print("\n[6/7] Comparing model performance...")
print("=" * 80)

# Create comparison DataFrame
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
print("\nModel Performance Comparison:")
print(comparison_df.to_string())

# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    comparison_df[metric].plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

# Hide the last subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_comparison.png")
plt.close()

# Find best model (based on F1-Score)
best_model_name = comparison_df['F1-Score'].idxmax()
best_model = models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1-Score: {comparison_df.loc[best_model_name, 'F1-Score']:.4f}")

# ============================================================================
# 7. SAVE BEST MODEL
# ============================================================================
print("\n[7/7] Saving best model...")

# Save the best model
joblib.dump(best_model, 'wildfire_best_model.pkl')
print(f"✓ Saved: wildfire_best_model.pkl ({best_model_name})")

# Save feature names
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'wildfire_feature_names.pkl')
print("✓ Saved: wildfire_feature_names.pkl")

# Save model metadata
metadata = {
    'best_model': best_model_name,
    'features': feature_names,
    'performance_metrics': comparison_df.loc[best_model_name].to_dict(),
    'training_samples': len(X_train_final),
    'test_samples': len(X_test),
    'class_distribution': fire_counts.to_dict()
}
pd.Series(metadata).to_json('wildfire_model_metadata.json', indent=2)
print("✓ Saved: wildfire_model_metadata.json")

# Generate detailed classification report for best model
print("\n" + "=" * 80)
print(f"DETAILED CLASSIFICATION REPORT - {best_model_name}")
print("=" * 80)
y_pred_best = results[best_model_name]['y_pred']
print(classification_report(y_test, y_pred_best, target_names=['No Fire', 'Fire']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Fire', 'Fire'],
            yticklabels=['No Fire', 'Fire'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: confusion_matrix_best_model.png")
plt.close()

# Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    print(feature_importance.to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['Feature'][:15], feature_importance['Importance'][:15])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top 15 Feature Importance - {best_model_name}', 
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: feature_importance.png")
    plt.close()

# ROC Curve for best model
fpr, tpr, _ = roc_curve(y_test, results[best_model_name]['y_pred_proba'])
roc_auc = results[best_model_name]['ROC-AUC']

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve - {best_model_name}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_best_model.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: roc_curve_best_model.png")
plt.close()

print("\n" + "=" * 80)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n✅ Best Model: {best_model_name}")
print(f"✅ Model File: wildfire_best_model.pkl")
print(f"✅ Scaler File: wildfire_scaler.pkl")
print(f"✅ Feature Names: wildfire_feature_names.pkl")
print(f"✅ Metadata: wildfire_model_metadata.json")
print(f"\n📊 Performance Summary:")
print(f"   Accuracy:  {comparison_df.loc[best_model_name, 'Accuracy']:.2%}")
print(f"   Precision: {comparison_df.loc[best_model_name, 'Precision']:.2%}")
print(f"   Recall:    {comparison_df.loc[best_model_name, 'Recall']:.2%}")
print(f"   F1-Score:  {comparison_df.loc[best_model_name, 'F1-Score']:.2%}")
print(f"   ROC-AUC:   {comparison_df.loc[best_model_name, 'ROC-AUC']:.2%}")
print("\n" + "=" * 80)
