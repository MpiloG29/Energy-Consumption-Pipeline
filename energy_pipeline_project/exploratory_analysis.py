import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# ============================================================================
# EXPLORATORY DATA ANALYSIS - SOUTH AFRICA ENERGY CONSUMPTION
# ============================================================================
# This script validates data realism and answers key questions about:
# - Load shedding impact
# - Seasonal variation
# - Time-of-day behavior
# - Household differences
# ============================================================================

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load data
data_path = os.path.join(os.path.dirname(__file__), "raw_data", "energy_data.csv")
df = pd.read_csv(data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - SOUTH AFRICA ENERGY CONSUMPTION")
print("=" * 80)

# ============================================================================
# 1. DATA QUALITY CHECKS
# ============================================================================
print("\n📊 DATA OVERVIEW")
print(f"   Records: {len(df):,}")
print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
print(f"   Households: {df['household_id'].nunique()}")
print(f"   Total consumption: {df['energy_consumption'].sum():.1f} kWh")
print(f"   Average hourly: {df['energy_consumption'].mean():.2f} kWh")
print(f"   Min/Max hourly: {df['energy_consumption'].min():.2f} / {df['energy_consumption'].max():.2f} kWh")

# Realism checks
print("\n✅ REALISM CHECKS")
negative_count = (df['energy_consumption'] < 0).sum()
print(f"   Negative consumption: {negative_count} (should be 0)")

outliers = (df['energy_consumption'] > 6).sum()
print(f"   Extremely high usage (>6 kWh): {outliers} ({outliers/len(df)*100:.2f}%)")

print(f"   Load shedding hours: {(df['load_shedding_stage'] > 0).sum():,}")
print(f"   % of time with outages: {(df['load_shedding_stage'] > 0).sum()/len(df)*100:.1f}%")

# ============================================================================
# 2. KEY QUESTION 1: IS ELECTRICITY CONSUMED DURING LOAD SHEDDING?
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 1: Is electricity consumed during load shedding?")
print("=" * 80)

during_outage = df[df['load_shedding_stage'] > 0]
no_outage = df[df['load_shedding_stage'] == 0]

avg_during = during_outage['energy_consumption'].mean()
avg_without = no_outage['energy_consumption'].mean()

print(f"\nAverage consumption WITH outage: {avg_during:.3f} kWh")
print(f"Average consumption WITHOUT outage: {avg_without:.3f} kWh")
print(f"Reduction during outage: {(1 - avg_during/avg_without)*100:.1f}%")

# Check households with backup power
with_backup = df[df['backup_power'] == True]
no_backup = df[df['backup_power'] == False]

during_with_backup = with_backup[with_backup['load_shedding_stage'] > 0]['energy_consumption'].mean()
during_no_backup = no_backup[no_backup['load_shedding_stage'] > 0]['energy_consumption'].mean()

print(f"\n🔌 Backup Power Effect:")
print(f"   Avg consumption during outage WITH backup: {during_with_backup:.3f} kWh")
print(f"   Avg consumption during outage WITHOUT backup: {during_no_backup:.3f} kWh")
print(f"   Backup power increases usage by: {(during_with_backup/during_no_backup - 1)*100:.1f}%")

# ============================================================================
# 3. KEY QUESTION 2: HOW DOES USAGE CHANGE BEFORE AND AFTER OUTAGES?
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 2: How does usage change before and after outages?")
print("=" * 80)

# Create flags for before/during/after
df['is_outage'] = (df['load_shedding_stage'] > 0).astype(int)
df['before_outage'] = 0
df['after_outage'] = 0

for i in range(1, len(df)):
    if df['is_outage'].iloc[i] == 1 and df['is_outage'].iloc[i-1] == 0:
        # Start of outage - mark previous hour as "before"
        df['before_outage'].iloc[i-1] = 1
    elif df['is_outage'].iloc[i] == 0 and df['is_outage'].iloc[i-1] == 1:
        # End of outage - mark this hour as "after"
        df['after_outage'].iloc[i] = 1

before_outage_data = df[df['before_outage'] == 1]
during_outage_data = df[df['is_outage'] == 1]
after_outage_data = df[df['after_outage'] == 1]

print(f"\nBefore outage avg: {before_outage_data['energy_consumption'].mean():.3f} kWh")
print(f"During outage avg: {during_outage_data['energy_consumption'].mean():.3f} kWh")
print(f"After outage avg: {after_outage_data['energy_consumption'].mean():.3f} kWh")

if len(after_outage_data) > 0:
    rebound_effect = (after_outage_data['energy_consumption'].mean() - before_outage_data['energy_consumption'].mean()) / before_outage_data['energy_consumption'].mean() * 100
    print(f"\n⚡ Rebound Effect: Consumption increases {rebound_effect:.1f}% after outage")
else:
    print("\n⚠️ No rebound data (no outages in dataset)")

# ============================================================================
# 4. KEY QUESTION 3: WHICH SEASON SHOWS HIGHEST DEMAND?
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 3: Which season shows the highest demand?")
print("=" * 80)

def get_season(month):
    if month in [6, 7, 8]:
        return "Winter"
    elif month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    else:
        return "Spring"

df['season'] = df['timestamp'].dt.month.apply(get_season)

seasonal_stats = df.groupby('season')['energy_consumption'].agg(['mean', 'sum', 'std']).sort_values('mean', ascending=False)
print("\nSeasonal Consumption (kWh):")
print(seasonal_stats.round(3))

# Analyze seasonal evening peaks
df['hour'] = df['timestamp'].dt.hour
winter_evenings = df[(df['season'] == 'Winter') & (18 <= df['hour']) & (df['hour'] <= 21)]
summer_evenings = df[(df['season'] == 'Summer') & (18 <= df['hour']) & (df['hour'] <= 21)]

print(f"\n🌙 Evening Peaks (6 PM - 10 PM):")
print(f"   Winter evening avg: {winter_evenings['energy_consumption'].mean():.3f} kWh")
print(f"   Summer evening avg: {summer_evenings['energy_consumption'].mean():.3f} kWh")
print(f"   Winter > Summer by: {(winter_evenings['energy_consumption'].mean() / summer_evenings['energy_consumption'].mean() - 1) * 100:.1f}%")

# ============================================================================
# 5. KEY QUESTION 4: DO HOUSEHOLDS BEHAVE DIFFERENTLY?
# ============================================================================
print("\n" + "=" * 80)
print("QUESTION 4: Do households behave differently?")
print("=" * 80)

household_stats = df.groupby('household_id').agg({
    'energy_consumption': ['mean', 'std', 'min', 'max'],
    'backup_power': ['first']
}).round(3)

print("\nHousehold Behavior:")
for hh in df['household_id'].unique():
    hh_data = df[df['household_id'] == hh]
    has_backup = "✓ Backup" if hh_data['backup_power'].iloc[0] else "✗ No backup"
    avg = hh_data['energy_consumption'].mean()
    peak = hh_data['energy_consumption'].max()
    variability = hh_data['energy_consumption'].std()
    print(f"\n   {hh} ({has_backup})")
    print(f"      Avg: {avg:.3f} kWh | Peak: {peak:.3f} kWh | Variability (σ): {variability:.3f}")

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================
print("\n📈 GENERATING VISUALIZATIONS...")

# Create output directory for plots
os.makedirs("dashboard_data/plots", exist_ok=True)

# Plot 1: Hourly consumption pattern by season
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('South Africa Energy Consumption: Seasonal & Time-of-Day Patterns', fontsize=16, fontweight='bold')

for idx, season in enumerate(['Winter', 'Summer', 'Spring', 'Autumn']):
    ax = axes[idx // 2, idx % 2]
    season_data = df[df['season'] == season].groupby('hour')['energy_consumption'].mean()
    ax.plot(season_data.index, season_data.values, linewidth=2.5, color='#4a6fa5')
    ax.fill_between(season_data.index, season_data.values, alpha=0.3, color='#4a6fa5')
    ax.set_title(f'{season}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Avg Consumption (kWh)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 23)

plt.tight_layout()
plt.savefig("dashboard_data/plots/seasonal_patterns.png", dpi=150, bbox_inches='tight')
print("   ✅ seasonal_patterns.png")
plt.close()

# Plot 2: Load shedding impact
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Load Shedding Impact on Energy Consumption', fontsize=14, fontweight='bold')

# Comparison by load shedding presence
conditions = ['No Outage', 'During Outage']
consumptions = [no_outage['energy_consumption'].mean(), during_outage['energy_consumption'].mean()]
colors = ['#28a745', '#dc3545']
ax = axes[0]
bars = ax.bar(conditions, consumptions, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Consumption (kWh)')
ax.set_title('Average Consumption with/without Outage')
for bar, val in zip(bars, consumptions):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}', ha='center', va='bottom', fontsize=11)

# Backup power effect
ax = axes[1]
backup_conditions = ['Without Backup\n(During Outage)', 'With Backup\n(During Outage)']
backup_consumptions = [during_no_backup, during_with_backup]
bars = ax.bar(backup_conditions, backup_consumptions, color=['#ffc107', '#28a745'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Average Consumption (kWh)')
ax.set_title('Backup Power Effect During Outages')
for bar, val in zip(bars, backup_consumptions):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{val:.2f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig("dashboard_data/plots/load_shedding_impact.png", dpi=150, bbox_inches='tight')
print("   ✅ load_shedding_impact.png")
plt.close()

# Plot 3: Household comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Household Behavior Differences', fontsize=14, fontweight='bold')

# Average consumption by household
ax = axes[0]
hh_avgs = df.groupby('household_id')['energy_consumption'].mean().sort_values()
colors_hh = ['#28a745' if df[df['household_id']==hh]['backup_power'].iloc[0] else '#dc3545' for hh in hh_avgs.index]
bars = ax.barh(hh_avgs.index, hh_avgs.values, color=colors_hh, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Average Consumption (kWh)')
ax.set_title('Average Consumption by Household')
for i, (bar, val) in enumerate(zip(bars, hh_avgs.values)):
    ax.text(val, bar.get_y() + bar.get_height()/2., f' {val:.2f}', va='center', fontsize=10)

# Consumption variability
ax = axes[1]
hh_std = df.groupby('household_id')['energy_consumption'].std().sort_values()
bars = ax.barh(hh_std.index, hh_std.values, color='#8e9aaf', alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Standard Deviation (kWh)')
ax.set_title('Consumption Variability by Household')
for i, (bar, val) in enumerate(zip(bars, hh_std.values)):
    ax.text(val, bar.get_y() + bar.get_height()/2., f' {val:.3f}', va='center', fontsize=10)

legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc='#28a745', alpha=0.7, label='With Backup Power'),
    plt.Rectangle((0, 0), 1, 1, fc='#dc3545', alpha=0.7, label='Without Backup Power')
]
fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig("dashboard_data/plots/household_comparison.png", dpi=150, bbox_inches='tight')
print("   ✅ household_comparison.png")
plt.close()

# Plot 4: Daily aggregate time series
fig, ax = plt.subplots(figsize=(16, 6))
daily_consumption = df.groupby(df['timestamp'].dt.date)['energy_consumption'].sum()
ax.plot(daily_consumption.index, daily_consumption.values, linewidth=1.5, color='#4a6fa5', alpha=0.8)
ax.fill_between(daily_consumption.index, daily_consumption.values, alpha=0.2, color='#4a6fa5')
ax.set_title('Daily Total Energy Consumption (12 Months)', fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Daily Consumption (kWh)')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dashboard_data/plots/daily_timeline.png", dpi=150, bbox_inches='tight')
print("   ✅ daily_timeline.png")
plt.close()

print("\n" + "=" * 80)
print("✅ EXPLORATORY ANALYSIS COMPLETE")
print("=" * 80)
