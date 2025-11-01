import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ Load dataset
df = pd.read_csv("backend\Metro_Interstate_Traffic_Volume.csv")

# ✅ Convert datetime column
df['date_time'] = pd.to_datetime(df['date_time'])

# ✅ Create folder to save charts
os.makedirs("charts", exist_ok=True)

# ✅ Extract time-based features
df['hour'] = df['date_time'].dt.hour
df['day'] = df['date_time'].dt.day
df['month'] = df['date_time'].dt.month
df['year'] = df['date_time'].dt.year
df['weekday'] = df['date_time'].dt.day_name()

# --- 🧭 BASIC INFO ---
print("\nDataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", list(df.columns))

# --- 🚗 Traffic Volume Over Time ---
plt.figure(figsize=(12, 5))
plt.plot(df['date_time'], df['traffic_volume'], color='orange')
plt.title("Traffic Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Traffic Volume")
plt.grid(True)
plt.tight_layout()
plt.savefig("charts/traffic_over_time.png")
plt.close()

# --- 🕒 Average Traffic by Hour ---
plt.figure(figsize=(10, 5))
hourly = df.groupby('hour')['traffic_volume'].mean()
hourly.plot(kind='bar', color='skyblue')
plt.title("Average Traffic Volume by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Traffic Volume")
plt.tight_layout()
plt.savefig("charts/traffic_by_hour.png")
plt.close()

# --- 📅 Monthly Traffic Trends ---
plt.figure(figsize=(10, 5))
monthly = df.groupby('month')['traffic_volume'].mean()
monthly.plot(kind='line', marker='o', color='green')
plt.title("Average Monthly Traffic Volume")
plt.xlabel("Month")
plt.ylabel("Average Volume")
plt.grid(True)
plt.tight_layout()
plt.savefig("charts/monthly_trends.png")
plt.close()

# --- 🌤️ Weather Impact ---
if 'weather_main' in df.columns:
    plt.figure(figsize=(10, 5))
    weather_avg = df.groupby('weather_main')['traffic_volume'].mean().sort_values()
    weather_avg.plot(kind='bar', color='coral')
    plt.title("Average Traffic by Weather Type")
    plt.xlabel("Weather")
    plt.ylabel("Average Traffic Volume")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("charts/weather_impact.png")
    plt.close()

# --- 📈 Traffic by Weekday ---
plt.figure(figsize=(10, 5))
weekday_avg = df.groupby('weekday')['traffic_volume'].mean()
weekday_avg = weekday_avg.reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
weekday_avg.plot(kind='bar', color='violet')
plt.title("Average Traffic by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Traffic Volume")
plt.tight_layout()
plt.savefig("charts/traffic_by_weekday.png")
plt.close()

# --- 🧮 Pie Chart: Holiday vs Non-Holiday ---
if 'holiday' in df.columns:
    holiday_count = df['holiday'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(
        holiday_count.values,
        labels=holiday_count.index,
        autopct='%1.1f%%',
        colors=['gold', 'lightblue', 'lightgreen']
    )
    plt.title("Holiday Distribution")
    plt.tight_layout()
    plt.savefig("charts/holiday_distribution.png")
    plt.close()

# --- 🔥 Correlation Heatmap ---
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png")
plt.close()

# --- 🌡️ Scatter: Temperature vs Traffic ---
if 'temp' in df.columns:
    plt.figure(figsize=(8, 5))
    plt.scatter(df['temp'], df['traffic_volume'], alpha=0.5, color='red')
    plt.title("Temperature vs Traffic Volume")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Traffic Volume")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("charts/temp_vs_traffic.png")
    plt.close()

print("\n✅ All charts generated and saved inside the 'charts/' folder!")
