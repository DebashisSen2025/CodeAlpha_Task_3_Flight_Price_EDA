# ============================================================
#  ✈️  FLIGHT PRICE ANALYSIS — EDA + ML
#  CodeAlpha Data Analytics Internship
#  Author: Debashis Sen
#
#  DATASET COLUMNS (Data_Train.xlsx):
#  Airline, Date_of_Journey, Source, Destination, Route,
#  Dep_Time, Arrival_Time, Duration, Total_Stops,
#  Additional_Info, Price
#
#  HOW TO RUN:
#  1. pip install pandas numpy matplotlib seaborn openpyxl scikit-learn
#  2. Place Data_Train.xlsx in SAME folder as this script
#  3. Run: python Flight_Price_EDA_FINAL.py
#     OR copy-paste into Jupyter Notebook
# ============================================================


# ─────────────────────────────────────────
# STEP 1: IMPORT LIBRARIES
# ─────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("✅ Libraries loaded!")


# ─────────────────────────────────────────
# STEP 2: LOAD YOUR DATASET
# ─────────────────────────────────────────
df = pd.read_excel("Data_Train.xlsx")

print(f"\n✅ Dataset loaded!")
print(f"📌 Shape     : {df.shape}")
print(f"📌 Columns   : {df.columns.tolist()}")
print(f"\n🔹 First 5 rows:\n{df.head().to_string()}")
print(f"\n🔹 Data Types:\n{df.dtypes}")
print(f"\n🔹 Missing Values:\n{df.isnull().sum()}")


# ─────────────────────────────────────────
# HELPER: save figure
# ─────────────────────────────────────────
def save_fig(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {filename}")


# ─────────────────────────────────────────
# STEP 3: DATA CLEANING
# ─────────────────────────────────────────
print("\n🧹 Cleaning data...")

# Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"  Removed duplicates: {before - len(df)} rows")

# Fix Arrival_Time — remove extra date info like "22 Mar"
df['Arrival_Time'] = df['Arrival_Time'].str.split(' ').str[0]

# Fill missing Total_Stops
df['Total_Stops'].fillna('1 stop', inplace=True)

# Extract numeric stops
df['Total_Stops'] = df['Total_Stops'].replace('non-stop', '0 stop')
df['Stops'] = df['Total_Stops'].str.extract(r'(\d+)').astype(int)

# Extract Date, Month from Date_of_Journey
df['Date']  = pd.to_datetime(df['Date_of_Journey'], dayfirst=True).dt.day
df['Month'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True).dt.month

# Extract Departure Hour & Minute
df['Dep_Hour']   = df['Dep_Time'].str.split(':').str[0].astype(int)
df['Dep_Minute'] = df['Dep_Time'].str.split(':').str[1].astype(int)

# Extract Arrival Hour & Minute
df['Arr_Hour']   = df['Arrival_Time'].str.split(':').str[0].astype(int)
df['Arr_Minute'] = df['Arrival_Time'].str.split(':').str[1].astype(int)

# Extract Duration in total minutes
def parse_duration(d):
    d = str(d)
    hours   = int(d.split('h')[0].strip()) if 'h' in d else 0
    minutes = int(d.split('h')[-1].replace('m','').strip()) if 'm' in d else 0
    return hours * 60 + minutes

df['Duration_mins'] = df['Duration'].apply(parse_duration)

print(f"  ✅ Cleaning complete. Shape: {df.shape}")
print(f"\n🔹 Cleaned Sample:\n{df[['Airline','Source','Destination','Stops','Duration_mins','Price']].head()}")


# ─────────────────────────────────────────
# STEP 4: PRICE DISTRIBUTION
# ─────────────────────────────────────────
print("\n📊 Plot 1: Price Distribution")
plt.figure(figsize=(12, 5))
sns.histplot(df['Price'], bins=60, kde=True, color='steelblue')
plt.title('💰 Distribution of Flight Prices', fontsize=16, fontweight='bold')
plt.xlabel('Price (INR)')
plt.ylabel('Frequency')
save_fig('01_price_distribution.png')

print(f"\n  Price Stats:")
print(f"  Min    : ₹{df['Price'].min():,}")
print(f"  Max    : ₹{df['Price'].max():,}")
print(f"  Mean   : ₹{df['Price'].mean():,.0f}")
print(f"  Median : ₹{df['Price'].median():,.0f}")


# ─────────────────────────────────────────
# STEP 5: AIRLINE-WISE PRICE
# ─────────────────────────────────────────
print("\n📊 Plot 2: Airline vs Price")
plt.figure(figsize=(14, 6))
airline_price = df.groupby('Airline')['Price'].median().sort_values(ascending=False)
sns.barplot(x=airline_price.index, y=airline_price.values, palette='coolwarm')
plt.title('✈️  Median Flight Price by Airline', fontsize=16, fontweight='bold')
plt.xlabel('Airline')
plt.ylabel('Median Price (INR)')
plt.xticks(rotation=35, ha='right')
save_fig('02_airline_price.png')


# ─────────────────────────────────────────
# STEP 6: FLIGHT COUNT PER AIRLINE
# ─────────────────────────────────────────
print("\n📊 Plot 3: Flight Count per Airline")
plt.figure(figsize=(13, 5))
airline_counts = df['Airline'].value_counts()
sns.barplot(x=airline_counts.index, y=airline_counts.values, palette='viridis')
plt.title('🛫 Number of Flights per Airline', fontsize=16, fontweight='bold')
plt.xlabel('Airline')
plt.ylabel('Number of Flights')
plt.xticks(rotation=35, ha='right')
save_fig('03_airline_count.png')


# ─────────────────────────────────────────
# STEP 7: SOURCE & DESTINATION
# ─────────────────────────────────────────
print("\n📊 Plot 4: Source & Destination Cities")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

src = df['Source'].value_counts()
axes[0].pie(src.values, labels=src.index, autopct='%1.1f%%',
            colors=sns.color_palette('pastel'))
axes[0].set_title('🏙️ Source City Distribution', fontweight='bold')

dst = df['Destination'].value_counts()
axes[1].pie(dst.values, labels=dst.index, autopct='%1.1f%%',
            colors=sns.color_palette('Set2'))
axes[1].set_title('🏁 Destination City Distribution', fontweight='bold')

save_fig('04_city_distribution.png')


# ─────────────────────────────────────────
# STEP 8: STOPS VS PRICE
# ─────────────────────────────────────────
print("\n📊 Plot 5: Stops vs Price")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Stops', y='Price',
            palette='Set3', order=sorted(df['Stops'].unique()))
plt.title('🔁 Number of Stops vs Flight Price', fontsize=16, fontweight='bold')
plt.xlabel('Number of Stops')
plt.ylabel('Price (INR)')
save_fig('05_stops_vs_price.png')


# ─────────────────────────────────────────
# STEP 9: DURATION VS PRICE
# ─────────────────────────────────────────
print("\n📊 Plot 6: Duration vs Price")
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Duration_mins', y='Price',
                hue='Airline', alpha=0.6, palette='tab10')
plt.title('⏱️  Flight Duration vs Price', fontsize=16, fontweight='bold')
plt.xlabel('Duration (minutes)')
plt.ylabel('Price (INR)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
save_fig('06_duration_vs_price.png')


# ─────────────────────────────────────────
# STEP 10: DEPARTURE HOUR VS PRICE
# ─────────────────────────────────────────
print("\n📊 Plot 7: Departure Hour vs Price")
plt.figure(figsize=(13, 6))
avg_price_hour = df.groupby('Dep_Hour')['Price'].mean()
sns.lineplot(x=avg_price_hour.index, y=avg_price_hour.values,
             marker='o', color='coral', linewidth=2.5)
plt.title('🕐 Departure Hour vs Avg Flight Price', fontsize=16, fontweight='bold')
plt.xlabel('Departure Hour (0-23)')
plt.ylabel('Average Price (INR)')
plt.xticks(range(0, 24))
save_fig('07_dep_hour_vs_price.png')


# ─────────────────────────────────────────
# STEP 11: MONTH-WISE PRICE TREND
# ─────────────────────────────────────────
print("\n📊 Plot 8: Month-wise Price Trend")
plt.figure(figsize=(12, 6))
month_price = df.groupby('Month')['Price'].mean()
month_names = {3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep'}
sns.lineplot(x=month_price.index, y=month_price.values,
             marker='o', color='steelblue', linewidth=2.5)
plt.xticks(month_price.index, [month_names.get(m, str(m)) for m in month_price.index])
plt.title('📅 Month-wise Average Flight Price', fontsize=16, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Average Price (INR)')
save_fig('08_month_price_trend.png')


# ─────────────────────────────────────────
# STEP 12: AIRLINE VS STOPS HEATMAP
# ─────────────────────────────────────────
print("\n📊 Plot 9: Airline vs Stops Heatmap")
plt.figure(figsize=(12, 6))
pivot = df.pivot_table(values='Price', index='Airline',
                       columns='Stops', aggfunc='median')
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.5)
plt.title('🔥 Median Price — Airline vs Stops', fontsize=16, fontweight='bold')
plt.xlabel('Number of Stops')
plt.ylabel('Airline')
save_fig('09_airline_stops_heatmap.png')


# ─────────────────────────────────────────
# STEP 13: TOP ROUTES BY PRICE
# ─────────────────────────────────────────
print("\n📊 Plot 10: Top Expensive Routes")
df['Route_Simple'] = df['Source'] + ' → ' + df['Destination']
top_routes = (df.groupby('Route_Simple')['Price']
              .median()
              .sort_values(ascending=False)
              .head(10))

plt.figure(figsize=(12, 6))
sns.barplot(x=top_routes.values, y=top_routes.index, palette='flare')
plt.title('💸 Top 10 Most Expensive Routes (Median)', fontsize=16, fontweight='bold')
plt.xlabel('Median Price (INR)')
plt.ylabel('Route')
save_fig('10_top_routes.png')


# ─────────────────────────────────────────
# STEP 14: ADDITIONAL INFO vs PRICE
# ─────────────────────────────────────────
print("\n📊 Plot 11: Additional Info vs Price")
plt.figure(figsize=(14, 6))
info_price = df.groupby('Additional_Info')['Price'].median().sort_values(ascending=False)
sns.barplot(x=info_price.values, y=info_price.index, palette='coolwarm')
plt.title('ℹ️  Additional Info vs Median Price', fontsize=16, fontweight='bold')
plt.xlabel('Median Price (INR)')
plt.ylabel('Additional Info')
save_fig('11_additional_info_price.png')


# ─────────────────────────────────────────
# STEP 15: CORRELATION HEATMAP
# ─────────────────────────────────────────
print("\n📊 Plot 12: Correlation Heatmap")
plt.figure(figsize=(10, 7))
num_cols = ['Stops', 'Duration_mins', 'Dep_Hour', 'Dep_Minute',
            'Arr_Hour', 'Arr_Minute', 'Date', 'Month', 'Price']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, square=True)
plt.title('🔥 Correlation Heatmap (Numeric Features)', fontsize=16, fontweight='bold')
save_fig('12_correlation_heatmap.png')


# ─────────────────────────────────────────
# STEP 16: ML — RANDOM FOREST PRICE PREDICTION
# ─────────────────────────────────────────
print("\n🤖 Running Random Forest Model...")

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

ml_df = df[['Airline','Source','Destination','Stops','Duration_mins',
            'Dep_Hour','Dep_Minute','Arr_Hour','Arr_Minute',
            'Date','Month','Price']].copy()

le = LabelEncoder()
for col in ['Airline', 'Source', 'Destination']:
    ml_df[col] = le.fit_transform(ml_df[col])

X = ml_df.drop('Price', axis=1)
y = ml_df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"  ✅ Model trained!")
print(f"  📌 MAE (Mean Abs Error) : ₹{mae:,.0f}")
print(f"  📌 R² Score             : {r2:.4f} ({r2*100:.1f}% accuracy)")

# Actual vs Predicted plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color='steelblue', edgecolors='white', s=40)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.title('🎯 Actual vs Predicted Flight Price', fontsize=16, fontweight='bold')
plt.xlabel('Actual Price (INR)')
plt.ylabel('Predicted Price (INR)')
plt.legend()
save_fig('13_actual_vs_predicted.png')

# Residual distribution
plt.figure(figsize=(10, 5))
residuals = y_test - y_pred
sns.histplot(residuals, bins=50, kde=True, color='coral')
plt.axvline(0, color='black', linestyle='--', lw=2)
plt.title('📉 Residual Distribution (Actual - Predicted)', fontsize=16, fontweight='bold')
plt.xlabel('Residual (INR)')
plt.ylabel('Frequency')
save_fig('14_residuals.png')

# Feature Importance
plt.figure(figsize=(10, 6))
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
feat_imp.plot(kind='barh', color='teal')
plt.title('🔑 Feature Importance — Random Forest', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score')
save_fig('15_feature_importance.png')


# ─────────────────────────────────────────
# STEP 17: FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "="*58)
print("📌 KEY INSIGHTS FROM FLIGHT PRICE EDA")
print("="*58)
print(f"""
 1. 💰 Average Price        : ₹{df['Price'].mean():,.0f}
 2. 💰 Median Price         : ₹{df['Price'].median():,.0f}
 3. 💰 Price Range          : ₹{df['Price'].min():,} – ₹{df['Price'].max():,}
 4. 🛫 Most flights by      : {df['Airline'].value_counts().idxmax()}
 5. 🏙️  Top Source City     : {df['Source'].value_counts().idxmax()}
 6. 🏁 Top Destination      : {df['Destination'].value_counts().idxmax()}
 7. 🔁 More stops = higher price on average
 8. ⏱️  Longer duration = costlier flight
 9. 🌙 Late night departures are generally cheaper
10. 📅 Prices peak around May–June (summer travel)
11. 🤖 Random Forest R²     : {r2*100:.1f}% accuracy
12. 🤖 Mean Absolute Error  : ₹{mae:,.0f}
""")
print("="*58)
print("✅ ALL DONE! 15 charts + ML model complete.")
print("📂 PNG files saved in your current folder.")
print("="*58)
