# ============================================================================
# SOUTH AFRICA ENERGY CONSUMPTION - THREE-MODEL ENSEMBLE TRAINING
# ============================================================================
# Trains three interpretable models for different purposes:
# 1. Linear Regression (baseline, explainability)
# 2. Random Forest (behavioral insights, feature importance)
# 3. Gradient Boosting (prediction accuracy)
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print("=" * 70)
print("TRAINING THREE-MODEL ENSEMBLE FOR SA ENERGY PREDICTION")
print("=" * 70)

df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "raw_data", "energy_data.csv"))
print(f"\n📊 Loaded {len(df)} records from raw data")

# Engineer features
X, y, df_with_features = engineer_features(df)
print(f"📊 Engineered {X.shape[1]} features")

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

# ============================================================================
# MODEL 1: LINEAR REGRESSION (Baseline)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 1: LINEAR REGRESSION (Baseline & Explainability)")
print("=" * 70)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)

lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_pred))
lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_test_pred))
lr_test_mae = mean_absolute_error(y_test, lr_test_pred)
lr_test_r2 = r2_score(y_test, lr_test_pred)

print(f"✅ Train RMSE: {lr_train_rmse:.4f} kWh")
print(f"✅ Test RMSE:  {lr_test_rmse:.4f} kWh")
print(f"✅ Test MAE:   {lr_test_mae:.4f} kWh")
print(f"✅ Test R²:    {lr_test_r2:.4f}")

# Save model
model_path_lr = os.path.join(os.path.dirname(__file__), "energy_model_linear.pkl")
joblib.dump(lr_model, model_path_lr)
print(f"💾 Saved to: {model_path_lr}")

# ============================================================================
# MODEL 2: RANDOM FOREST (Behavioral Insights)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 2: RANDOM FOREST (Behavioral Insights & Feature Importance)")
print("=" * 70)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)

rf_train_rmse = np.sqrt(mean_squared_error(y_train, rf_train_pred))
rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_test_pred))
rf_test_mae = mean_absolute_error(y_test, rf_test_pred)
rf_test_r2 = r2_score(y_test, rf_test_pred)

print(f"✅ Train RMSE: {rf_train_rmse:.4f} kWh")
print(f"✅ Test RMSE:  {rf_test_rmse:.4f} kWh")
print(f"✅ Test MAE:   {rf_test_mae:.4f} kWh")
print(f"✅ Test R²:    {rf_test_r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n🔍 Top 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:40} {row['importance']:.4f}")

# Save model
model_path_rf = os.path.join(os.path.dirname(__file__), "energy_model_random_forest.pkl")
joblib.dump(rf_model, model_path_rf)
print(f"\n💾 Saved to: {model_path_rf}")

# ============================================================================
# MODEL 3: GRADIENT BOOSTING (Prediction Accuracy)
# ============================================================================
print("\n" + "=" * 70)
print("MODEL 3: GRADIENT BOOSTING (Prediction Accuracy)")
print("=" * 70)

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb_model.fit(X_train, y_train)

gb_train_pred = gb_model.predict(X_train)
gb_test_pred = gb_model.predict(X_test)

gb_train_rmse = np.sqrt(mean_squared_error(y_train, gb_train_pred))
gb_test_rmse = np.sqrt(mean_squared_error(y_test, gb_test_pred))
gb_test_mae = mean_absolute_error(y_test, gb_test_pred)
gb_test_r2 = r2_score(y_test, gb_test_pred)

print(f"✅ Train RMSE: {gb_train_rmse:.4f} kWh")
print(f"✅ Test RMSE:  {gb_test_rmse:.4f} kWh")
print(f"✅ Test MAE:   {gb_test_mae:.4f} kWh")
print(f"✅ Test R²:    {gb_test_r2:.4f}")

# Save model
model_path_gb = os.path.join(os.path.dirname(__file__), "energy_model_gradient_boosting.pkl")
joblib.dump(gb_model, model_path_gb)
print(f"💾 Saved to: {model_path_gb}")

# ============================================================================
# MODEL COMPARISON SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
    "Test RMSE": [lr_test_rmse, rf_test_rmse, gb_test_rmse],
    "Test MAE": [lr_test_mae, rf_test_mae, gb_test_mae],
    "Test R²": [lr_test_r2, rf_test_r2, gb_test_r2]
})

print("\n" + comparison.to_string(index=False))
print("\n" + "=" * 70)
print("✅ ALL THREE MODELS TRAINED AND SAVED SUCCESSFULLY")
print("=" * 70)
