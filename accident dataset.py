import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Generate Sample Traffic Accident Data ---
np.random.seed(42)
n = 500  # Number of accident records

data = {
    'Accident_ID': range(1, n+1),
    'Road_Condition': np.random.choice(['Dry', 'Wet', 'Snowy', 'Icy'], n, p=[0.6,0.25,0.1,0.05]),
    'Weather': np.random.choice(['Clear', 'Rain', 'Fog', 'Snow'], n, p=[0.5,0.3,0.1,0.1]),
    'Time_of_Day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n),
    'Latitude': np.random.uniform(12.90, 13.10, n),   # Example city coordinates
    'Longitude': np.random.uniform(77.50, 77.70, n),
    'Severity': np.random.choice(['Minor', 'Major', 'Fatal'], n, p=[0.7,0.25,0.05])
}

df = pd.DataFrame(data)

# --- Step 2: Analyze patterns ---
# Road Condition vs Accident Count
road_accidents = df['Road_Condition'].value_counts()
weather_accidents = df['Weather'].value_counts()
time_accidents = df['Time_of_Day'].value_counts()
severity_accidents = df['Severity'].value_counts()

# --- Step 3: Visualize patterns ---
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.countplot(x='Road_Condition', data=df, palette='Set2')
plt.title("Accidents by Road Condition")

plt.subplot(1,3,2)
sns.countplot(x='Weather', data=df, palette='Set3')
plt.title("Accidents by Weather")

plt.subplot(1,3,3)
sns.countplot(x='Time_of_Day', data=df, palette='Set1')
plt.title("Accidents by Time of Day")

plt.tight_layout()
plt.show()

# --- Step 4: Visualize Accident Hotspots ---
plt.figure(figsize=(8,6))
plt.scatter(df['Longitude'], df['Latitude'], c='red', alpha=0.5)
plt.title("Accident Hotspots")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# --- Step 5: Contribution Spotlight (Severity Distribution) ---
plt.figure(figsize=(6,6))
severity_accidents.plot(kind='pie', autopct='%1.1f%%', colors=['green','orange','red'], startangle=90)
plt.title("Accident Severity Contribution")
plt.ylabel('')
plt.show()

# --- Step 6: Optional: Correlation Heatmap ---
plt.figure(figsize=(6,4))
sns.heatmap(pd.crosstab(df['Road_Condition'], df['Weather']), annot=True, fmt='d', cmap='YlGnBu')
plt.title("Road Condition vs Weather")
plt.show()

print("Sample Accident Data (first 5 rows):")
print(df.head())