# ✈️ Flight Price Analysis — EDA + ML
### CodeAlpha Data Analytics Internship Task

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python) ![Pandas](https://img.shields.io/badge/Pandas-2.x-green?logo=pandas) ![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.8-orange?logo=scikit-learn) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 📌 Project Overview

This project performs a complete **Exploratory Data Analysis (EDA)** and **Machine Learning Price Prediction** on a real-world Indian flight price dataset (`Data_Train.xlsx`).

The goal is to uncover patterns in flight pricing across airlines, routes, stops, duration, and departure times — and build a predictive model using **Random Forest Regressor**.

---

## 📂 Dataset

**File:** `Data_Train.xlsx`

| Column | Description |
|---|---|
| Airline | Name of the airline |
| Date_of_Journey | Date of travel |
| Source | Departure city |
| Destination | Arrival city |
| Route | Flight route path |
| Dep_Time | Departure time |
| Arrival_Time | Arrival time |
| Duration | Total flight duration |
| Total_Stops | Number of stops |
| Additional_Info | Extra flight info |
| Price | Ticket price (INR) — Target variable |

---

## 🛠️ Tools & Libraries

- **Python 3.x**
- **Pandas** — Data cleaning & manipulation
- **NumPy** — Numerical operations
- **Matplotlib & Seaborn** — Data visualization
- **Scikit-learn** — Machine learning (Random Forest)
- **OpenPyXL** — Reading Excel files

---

## 📊 EDA — Charts Generated (15 Visualizations)

| # | Chart | Insight |
|---|---|---|
| 1 | Price Distribution | Right-skewed, most flights ₹3,000–₹12,000 |
| 2 | Airline vs Median Price | Jet Airways Business is the costliest |
| 3 | Flight Count per Airline | IndiGo operates the most flights |
| 4 | Source & Destination Pie | Delhi & Mumbai dominate both ends |
| 5 | Stops vs Price (Boxplot) | More stops = higher price |
| 6 | Duration vs Price (Scatter) | Longer flights cost more |
| 7 | Departure Hour vs Price | Late-night flights are cheaper |
| 8 | Month-wise Price Trend | Prices peak in May–June |
| 9 | Airline × Stops Heatmap | Premium airlines charge more per stop |
| 10 | Top 10 Expensive Routes | Banglore → New Delhi tops the list |
| 11 | Additional Info vs Price | Business class adds huge premium |
| 12 | Correlation Heatmap | Duration & Stops correlate most with Price |
| 13 | Actual vs Predicted | Model follows the trend well |
| 14 | Residual Distribution | Errors are normally distributed |
| 15 | Feature Importance | Duration & Stops are top predictors |

---

## 🤖 Machine Learning — Random Forest Regressor

- **Model:** `RandomForestRegressor (n_estimators=100)`
- **Train/Test Split:** 80% / 20%
- **Encoding:** Label Encoding for categorical columns

### Results:
| Metric | Value |
|---|---|
| R² Score | ~81–85% accuracy |
| Mean Absolute Error | ~₹1,200 |

---

## 🔑 Key Insights

1. 💰 Average flight price is approximately **₹9,000 INR**
2. 🔁 **More stops = higher price** on average
3. ⏱️ **Longer duration** flights cost significantly more
4. 🌙 **Late-night departures** are generally the cheapest
5. 📅 **May–June** sees the highest average prices (summer travel)
6. ✈️ **Jet Airways Business** is the most expensive airline
7. 🛫 **IndiGo** operates the highest number of flights
8. 🏙️ **Delhi** is the top source and destination city
9. 🔑 **Duration & Stops** are the strongest price predictors
10. 🤖 Random Forest achieves strong predictive accuracy

---

## 🚀 How to Run

**Step 1 — Install dependencies:**
```bash
pip install pandas numpy matplotlib seaborn openpyxl scikit-learn
```

**Step 2 — Place both files in the same folder:**
```
📁 Your Folder/
   ├── Flight_Price_EDA_FINAL.py
   └── Data_Train.xlsx
```

**Step 3 — Run the script:**
```bash
python Flight_Price_EDA_FINAL.py
```

All 15 charts will be saved as `.png` files in the same folder.

---

## 📁 Output Files

```
01_price_distribution.png
02_airline_price.png
03_airline_count.png
04_city_distribution.png
05_stops_vs_price.png
06_duration_vs_price.png
07_dep_hour_vs_price.png
08_month_price_trend.png
09_airline_stops_heatmap.png
10_top_routes.png
11_additional_info_price.png
12_correlation_heatmap.png
13_actual_vs_predicted.png
14_residuals.png
15_feature_importance.png
```

---

## 👤 Author

**Debashis Sen**
CodeAlpha Data Analytics Internship
📅 February 2026 – March 2026

---

## 📜 License

This project is for educational and internship submission purposes only.
