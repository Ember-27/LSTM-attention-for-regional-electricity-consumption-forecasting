# ==============================================
# Power Consumption Forecast — Data Cleaning + Visualization
# Columns:
# Datetime, Temperature, Humidity, WindSpeed,
# GeneralDiffuseFlows, DiffuseFlows,
# PowerConsumption_Zone1, PowerConsumption_Zone2, PowerConsumption_Zone3
# Just change the CSV filename!
# ==============================================

# Critical: set backend BEFORE importing matplotlib.pyplot (to avoid PyCharm backend issues)
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import r2_score

# Suppress all warnings
warnings.filterwarnings("ignore")

# Use DejaVu Sans (no CJK dependency)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ===================== [ONLY CHANGE NEEDED: filename] =====================
df = pd.read_csv("powerconsumption.csv")   # <--- Change to your CSV filename
# ======================================================================

# Inspect raw data
print("="*60)
print("Raw data (first 5 rows):")
print(df.head())
print("Data shape:", df.shape)
print("All columns:", df.columns.tolist())
print("="*60)

# ----------------------
# Parse datetime column
# ----------------------
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime")

# ----------------------
# 10-second → hourly aggregation
# ----------------------
df_hour = df.resample("h").agg({
    "Temperature": "mean",
    "Humidity": "mean",
    "WindSpeed": "mean",
    "GeneralDiffuseFlows": "mean",
    "DiffuseFlows": "mean",
    "PowerConsumption_Zone1": "mean",
    "PowerConsumption_Zone2": "mean",
    "PowerConsumption_Zone3": "mean"
})

print("Aggregated to hourly data, shape:", df_hour.shape)

# ----------------------
# Data Cleaning
# ----------------------

# 1. Missing values
print("\nMissing value count:")
print(df_hour.isnull().sum())
df_hour = df_hour.dropna()

# 2. Remove duplicates
df_hour = df_hour[~df_hour.index.duplicated(keep="first")]

# 3. Power consumption must be non-negative
zone_cols = [
    "PowerConsumption_Zone1",
    "PowerConsumption_Zone2",
    "PowerConsumption_Zone3"
]
for col in zone_cols:
    df_hour = df_hour[df_hour[col] >= 0]

# 4. Outlier removal (3-sigma)
def remove_outliers(data, col, n_std=3):
    mean = data[col].mean()
    std = data[col].std()
    return data[(data[col] > mean - n_std*std) & (data[col] < mean + n_std*std)]

for col in zone_cols:
    df_hour = remove_outliers(df_hour, col)

print(f"\nCleaning complete. Final sample count: {len(df_hour)}")

# ----------------------
# Build time features (for modeling)
# ----------------------
df_hour["hour"] = df_hour.index.hour
df_hour["dayofweek"] = df_hour.index.dayofweek
df_hour["month"] = df_hour.index.month

print("Cleaned data (first 5 rows):")
print(df_hour.head())
print("="*60)

# ==============================================
# Visualization
# ==============================================

# ----------------------
# Figure 1: Zone 1 Power Consumption Time Series
# ----------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

plt.figure(figsize=(20,5))
n_points = min(1000, len(df_hour))
plt.plot(df_hour["PowerConsumption_Zone1"].iloc[:n_points], linewidth=0.7)
plt.title("Zone 1 Power Consumption (Hourly)")
plt.xlabel("Time")
plt.ylabel("Power Consumption")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figure_1_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()


# ----------------------
# Figure 2: 24-Hour Power Consumption Pattern
# ----------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

hour_mean = df_hour.groupby("hour")["PowerConsumption_Zone1"].mean()
plt.figure(figsize=(12,5))
plt.plot(hour_mean.index, hour_mean.values, marker='o')
plt.title("24-Hour Average Power Consumption")
plt.xlabel("Hour")
plt.ylabel("Average Power Consumption")
plt.xticks(range(0,24))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figure_2_hourly_pattern.png', dpi=300, bbox_inches='tight')
plt.show()


# ----------------------
# Figure 3: Temperature vs Power Consumption
# ----------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

plt.figure(figsize=(10,6))
n_points = len(df_hour)
s_val = max(1, 1000 / (n_points**0.5))
plt.scatter(df_hour["Temperature"], df_hour["PowerConsumption_Zone1"],
            s=s_val, alpha=0.5, edgecolors='none')
plt.title("Temperature vs Power Consumption")
plt.xlabel("Temperature (C)")
plt.ylabel("Power Consumption")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figure_3_temp_vs_power.png', dpi=300, bbox_inches='tight')
plt.show()


# ----------------------
# Figure 4: Feature Correlation Heatmap
# ----------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

corr = df_hour.corr()
n_features = len(corr)
plt.figure(figsize=(n_features*0.8, n_features*0.7))
mask = np.triu(np.ones_like(corr, dtype=bool))  # only show lower triangle
sns.heatmap(corr, cmap="coolwarm", annot=True, fmt=".2f",
            linewidths=0.5, mask=mask, center=0,
            square=True, cbar_kws={"shrink": 0.8})

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig('figure_4_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()



# ----------------------
# Figure 5: Temperature Binned Mean Power (U-shape relationship)
# ----------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Dynamic temperature binning (2 degC per bin)
temp_min = df_hour['Temperature'].min()
temp_max = df_hour['Temperature'].max()
bins = range(int(temp_min), int(temp_max)+2, 2)

df_hour['temp_bin'] = pd.cut(
    df_hour['Temperature'],
    bins=bins,
    right=False,
    include_lowest=True
)

# 2. Mean power consumption per temperature bin
temp_mean = df_hour.groupby('temp_bin', observed=True)['PowerConsumption_Zone1'].mean()

# 3. Plot temperature binned mean
plt.figure(figsize=(14, 6))

plt.plot(range(len(temp_mean)), temp_mean.values,
         marker='o', color='#1f77b4', alpha=0.8, linewidth=2, label='Mean')

plt.xticks(range(len(temp_mean)), [str(x) for x in temp_mean.index], rotation=45, ha='right')

plt.title('Temperature Bins vs Mean Power Consumption (U-shape)', fontsize=14)
plt.xlabel('Temperature Bins (degC)', fontsize=12)
plt.ylabel('Zone 1 Mean Power Consumption', fontsize=12)

plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('figure_5_temp_bin.png', dpi=300, bbox_inches='tight')
plt.show()



# ----------------------
# Figure 6: 20:00 (peak hour) Temperature vs Power Scatter
# ----------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

df_20h = df_hour[df_hour['hour'] == 20]

plt.figure(figsize=(10, 6))
plt.scatter(
    df_20h['Temperature'],
    df_20h['PowerConsumption_Zone1'],
    s=5, alpha=0.6, color='#ff7f0e'
)
# Add quadratic trend line
z = np.polyfit(df_20h['Temperature'], df_20h['PowerConsumption_Zone1'], 2)
p = np.poly1d(z)

# Calculate R-squared
y_pred = p(df_20h['Temperature'])
r2 = r2_score(df_20h['PowerConsumption_Zone1'], y_pred)

# Display R-squared on plot
plt.text(0.05, 0.95, f'R2 = {r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
plt.plot(
    sorted(df_20h['Temperature']),
    p(sorted(df_20h['Temperature'])),
    color='red', linewidth=2
)
plt.title('20:00 (Peak Hour) Temperature vs Power Consumption', fontsize=14)
plt.xlabel('Temperature (degC)', fontsize=12)
plt.ylabel('Zone 1 Power Consumption', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
temp_range = df_20h['Temperature'].max() - df_20h['Temperature'].min()
plt.xlim(df_20h['Temperature'].min() - 0.1*temp_range,
         df_20h['Temperature'].max() + 0.1*temp_range)

plt.savefig('figure_6_20h_scatter.png', dpi=300, bbox_inches='tight')
plt.show()


# ----------------------
# Quantify U-shape relationship (output temperature-squared correlation)
# ----------------------
# Create temperature-squared feature
df_hour['Temperature_sq'] = df_hour['Temperature'] ** 2

# Compute core feature correlations
core_corr = df_hour[[
    'PowerConsumption_Zone1',
    'Temperature',
    'Temperature_sq',
    'hour',
    'Humidity'
]].corr()

print("\n====== Core Feature Correlations (Quantifying Temperature U-shape) ======")
print(core_corr['PowerConsumption_Zone1'])

# Save cleaned data
df_hour.to_csv("data_cleaned_hour.csv", encoding="utf-8-sig")
print("\nDone! Cleaned data saved to data_cleaned_hour.csv")