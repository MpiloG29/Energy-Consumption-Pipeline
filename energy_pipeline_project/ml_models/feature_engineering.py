import pandas as pd
import numpy as np

# ============================================================================
# FEATURE ENGINEERING MODULE FOR SOUTH AFRICA ENERGY PIPELINE
# ============================================================================

def engineer_features(df):
    """
    Create interpretable, purposeful features from raw energy consumption data.
    
    Inputs:
        df: DataFrame with columns [household_id, timestamp, energy_consumption, 
                                     load_shedding_stage, backup_power]
    
    Returns:
        features_df: DataFrame with engineered features
        target: Series of energy_consumption values
    """
    
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # ========================================================================
    # TIME FEATURES (Interpretable)
    # ========================================================================
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    
    # Cyclical encoding for hour (sine/cosine for circular nature)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # Cyclical encoding for day_of_week
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    
    # ========================================================================
    # SEASONAL FEATURES (Interpretable)
    # ========================================================================
    def season_encode(month):
        if month in [6, 7, 8]:  # Winter
            return "Winter"
        elif month in [12, 1, 2]:  # Summer
            return "Summer"
        elif month in [3, 4, 5]:  # Autumn/Fall
            return "Autumn"
        else:  # Spring (9, 10, 11)
            return "Spring"
    
    df["season"] = df["month"].apply(season_encode)
    
    # One-hot encode season
    season_dummies = pd.get_dummies(df["season"], prefix="season", dtype=int)
    df = pd.concat([df, season_dummies], axis=1)
    
    # ========================================================================
    # LOAD SHEDDING FEATURES (Critical for SA context)
    # ========================================================================
    df["is_load_shedding"] = (df["load_shedding_stage"] > 0).astype(int)
    
    # Post-load-shedding flag (1 hour after outage ends)
    df["post_load_shedding"] = 0
    load_shedding_prev = df["load_shedding_stage"].shift(1).fillna(0)
    df.loc[(df["load_shedding_stage"] == 0) & (load_shedding_prev > 0), "post_load_shedding"] = 1
    
    df["load_shedding_stage_normalized"] = df["load_shedding_stage"] / 6  # 0-1 scale
    
    # ========================================================================
    # CONTEXT FEATURES (Interpretable)
    # ========================================================================
    df["has_backup_power"] = df["backup_power"].astype(int)
    
    # Encode household_id as categorical
    household_dummies = pd.get_dummies(df["household_id"], prefix="household", dtype=int)
    df = pd.concat([df, household_dummies], axis=1)
    
    # ========================================================================
    # SELECT FEATURES FOR MODELING
    # ========================================================================
    feature_columns = [
        # Time-based features
        "hour", "day_of_week", "is_weekend", "month",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        
        # Seasonal dummies
        "season_Autumn", "season_Spring", "season_Summer", "season_Winter",
        
        # Load shedding features
        "is_load_shedding", "post_load_shedding", "load_shedding_stage_normalized",
        
        # Context features
        "has_backup_power",
        
        # Household identifiers
        "household_House_1", "household_House_2", "household_House_3", 
        "household_House_4", "household_House_5"
    ]
    
    features_df = df[feature_columns].copy()
    target = df["energy_consumption"].copy()
    
    print(f"✅ Features engineered: {len(feature_columns)} features for {len(features_df)} samples")
    
    return features_df, target, df


def get_feature_descriptions():
    """Return human-readable descriptions of features for explainability."""
    descriptions = {
        "hour": "Hour of the day (0-23)",
        "day_of_week": "Day of week (0=Monday, 6=Sunday)",
        "is_weekend": "1 if weekend, 0 if weekday",
        "month": "Month of the year (1-12)",
        "hour_sin": "Sine component of hour (cyclical encoding)",
        "hour_cos": "Cosine component of hour (cyclical encoding)",
        "dow_sin": "Sine component of day of week",
        "dow_cos": "Cosine component of day of week",
        "season_Winter": "1 if winter (June-August), 0 otherwise",
        "season_Summer": "1 if summer (Dec-Feb), 0 otherwise",
        "season_Autumn": "1 if autumn (Mar-May), 0 otherwise",
        "season_Spring": "1 if spring (Sep-Nov), 0 otherwise",
        "is_load_shedding": "1 if electricity outage is active, 0 otherwise",
        "post_load_shedding": "1 if this hour is immediately after outage ends, 0 otherwise",
        "load_shedding_stage_normalized": "Severity of outage (0=none, 1=stage 6)",
        "has_backup_power": "1 if household has generator/battery, 0 otherwise",
        "household_House_1": "1 if House_1, 0 otherwise",
        "household_House_2": "1 if House_2, 0 otherwise",
        "household_House_3": "1 if House_3, 0 otherwise",
        "household_House_4": "1 if House_4, 0 otherwise",
        "household_House_5": "1 if House_5, 0 otherwise",
    }
    return descriptions
