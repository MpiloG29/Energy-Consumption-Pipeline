import pandas as pd
import numpy as np

# Generate 1 month of hourly data
dates = pd.date_range("2025-01-01", periods=24*30, freq='h')
households = [f"House_{i}" for i in range(1, 6)]

data = []
for h in households:
    consumption = np.random.normal(loc=2.5, scale=0.5, size=len(dates))  # kWh
    data.extend(zip([h]*len(dates), dates, consumption))

df = pd.DataFrame(data, columns=["household_id", "timestamp", "energy_consumption"])

# Save into raw_data folder
df.to_csv("raw_data/energy_data.csv", index=False)
print("✅ Energy data generated and saved to raw_data/energy_data.csv")



