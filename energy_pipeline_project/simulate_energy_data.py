import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# SOUTH AFRICA ENERGY CONSUMPTION DATA GENERATOR
# 12 months, 5 households, load shedding, seasonality, time-of-day behavior
# ============================================================================

np.random.seed(42)  # For reproducibility

# Generate 12 months of hourly data (2025-01-01 to 2025-12-31)
start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 12, 31, 23, 0)
dates = pd.date_range(start_date, end_date, freq='H')

print(f"Generating {len(dates)} hourly records ({len(dates)//24} days)...")

# ============================================================================
# 1. HOUSEHOLD PROFILES (realistic, parameterized)
# ============================================================================
household_profiles = {
    "House_1": {"base_load": 0.8, "peak_sensitivity": 1.8, "backup_power": False, "resilience": 0.3},    # Low consumption, no backup
    "House_2": {"base_load": 1.2, "peak_sensitivity": 2.5, "backup_power": True, "resilience": 0.7},     # Medium, with backup
    "House_3": {"base_load": 1.0, "peak_sensitivity": 2.2, "backup_power": False, "resilience": 0.4},    # Medium, no backup
    "House_4": {"base_load": 1.5, "peak_sensitivity": 3.2, "backup_power": True, "resilience": 0.8},     # High, with backup
    "House_5": {"base_load": 0.9, "peak_sensitivity": 2.0, "backup_power": False, "resilience": 0.2},    # Low-medium, no backup
}

# ============================================================================
# 2. LOAD SHEDDING SCHEDULE (randomized, realistic)
# ============================================================================
def generate_load_shedding_schedule(dates):
    """
    Simulate randomized load shedding stages (0-6) across the year.
    Stage 0 = no outage, Stage 1-6 = increasing severity.
    """
    stages = np.zeros(len(dates), dtype=int)
    
    # Simulate load shedding: ~60% of days no shedding, ~40% with varying stages
    num_outage_periods = np.random.randint(80, 120)  # ~80-120 outage periods in a year
    
    for _ in range(num_outage_periods):
        start_idx = np.random.randint(0, len(dates) - 48)  # Random start
        duration = np.random.randint(2, 6)  # 2-6 hours
        stage = np.random.randint(1, 7)  # Stage 1-6
        
        end_idx = min(start_idx + duration, len(dates))
        stages[start_idx:end_idx] = stage
    
    return stages

load_shedding_stages = generate_load_shedding_schedule(dates)

# ============================================================================
# 3. SEASONAL VARIATION
# ============================================================================
def get_seasonal_multiplier(month):
    """
    Winter (June, July, August) = higher consumption
    Summer (Dec, Jan, Feb) = moderate
    Shoulder (spring/autumn) = moderate-low
    """
    if month in [6, 7, 8]:  # Winter
        return 1.4
    elif month in [12, 1, 2]:  # Summer
        return 1.0
    else:  # Shoulder
        return 0.9

# ============================================================================
# 4. TIME-OF-DAY BEHAVIOR
# ============================================================================
def get_time_of_day_multiplier(hour):
    """
    Morning peak: 6-9 AM (1.3x)
    Daytime: 9 AM - 5 PM (0.8x)
    Evening peak: 6 PM - 9 PM (1.8x)
    Night: 10 PM - 5 AM (0.5x)
    """
    if 6 <= hour < 9:
        return 1.3
    elif 9 <= hour < 17:
        return 0.8
    elif 18 <= hour < 22:
        return 1.8
    else:
        return 0.5

# ============================================================================
# 5. GENERATE DATA
# ============================================================================
data = []

for household_id, profile in household_profiles.items():
    base_load = profile["base_load"]
    peak_sensitivity = profile["peak_sensitivity"]
    has_backup = profile["backup_power"]
    resilience = profile["resilience"]
    
    for idx, timestamp in enumerate(dates):
        month = timestamp.month
        hour = timestamp.hour
        
        # Calculate base consumption
        seasonal_mult = get_seasonal_multiplier(month)
        time_mult = get_time_of_day_multiplier(hour)
        consumption = base_load * seasonal_mult * time_mult
        
        # Apply load shedding effect
        load_shedding_stage = load_shedding_stages[idx]
        
        if load_shedding_stage > 0:
            # During outage: consumption drops
            if has_backup:
                # Households with backup use reduced power
                consumption *= (0.2 + 0.3 * (1 - load_shedding_stage / 6))
            else:
                # No backup: near-zero consumption
                consumption *= 0.05
        else:
            # After outage: rebound effect (if previous hour was outage)
            if idx > 0 and load_shedding_stages[idx - 1] > 0:
                consumption *= (1.0 + resilience * 0.5)  # Rebound multiplier
        
        # Add small realistic noise last
        noise = np.random.normal(0, consumption * 0.08)  # 8% std noise
        consumption = max(0, consumption + noise)  # No negative consumption
        
        data.append({
            "household_id": household_id,
            "timestamp": timestamp,
            "energy_consumption": round(consumption, 3),
            "load_shedding_stage": load_shedding_stage,
            "backup_power": has_backup
        })

# ============================================================================
# 6. CREATE DATAFRAME AND SAVE
# ============================================================================
df = pd.DataFrame(data)

print(f"✅ Generated {len(df)} records across {len(df['household_id'].unique())} households")
print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"   Load shedding stages active: {(df['load_shedding_stage'] > 0).sum()} hours")
print(f"   Total consumption: {df['energy_consumption'].sum():.1f} kWh")
print(f"   Avg hourly consumption: {df['energy_consumption'].mean():.2f} kWh")

df.to_csv("raw_data/energy_data.csv", index=False)
print("✅ Energy data saved to raw_data/energy_data.csv")



