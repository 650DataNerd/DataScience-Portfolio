import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from tkinter import Tk, filedialog  
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#hiding the tinker window
Tk().withdraw()

#open file dialog
file_path = askopenfilename(title="Selet CSV File", filetypes=[("CSV files", "*.csv")])

#Load the dataset
if file_path:
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully!")
    print(df.head())
else:
    print("No file selected.")
    
    
#data preprocessing
#check for missing values
print("Misising Values:\n", df.isnull().sum())

#check duplicate
print("Duplicate Rows:", df.duplicated().sum())

#data summary
#data types and basic statistics
print(df.info())
print(df.describe())

#visualizing the data
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Date", y="Close")
plt.title("Bitcoin Closing Price Trend")
plt.xticks(df["Date"][::200], rotation=45, ha='right')
plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.show()

#Ensure date column is in datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="Date", y="Close")
plt.title("Bitcoin Closing Price Trend")

# Format x-axis to show yearly ticks
plt.gca().xaxis.set_major_locator(mdates.YearLocator(1)) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y"))  
plt.xticks(rotation=45, ha="right")  
plt.xlabel("Year")
plt.ylabel("Closing Price")
plt.grid(True)

plt.show()


#supervised learning
#regression
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

df = df.sort_values(by="Date")

#creating new features
df['Previous_Close'] = df['Close'].shift(1)
df['7Day_MA'] = df['Close'].rolling(window=7).mean()
df['30Day_MA'] = df['Close'].rolling(window=30).mean()
df['Volatility'] = df['High'] - df['Low'] 

# Drop rows with NaN values (caused by shifting and rolling)
df = df.dropna()

# Define features and target variable
features = ['Previous_Close', '7Day_MA', '30Day_MA', 'Volatility']
target = 'Close'


# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, shuffle=False
)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Print dataset summary
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# Plot BTC Closing Price over time
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], label="Closing Price", color="blue")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Bitcoin Closing Price Over Time")
plt.legend()
plt.show()

#Initialize and train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Plot actual vs. predicted prices
plt.figure(figsize=(12, 5))
plt.plot(df['Date'].iloc[-len(y_test):], y_test, label="Actual Price", color="blue")
plt.plot(df['Date'].iloc[-len(y_test):], y_pred, label="Predicted Price", linestyle="dashed", color="red")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Actual vs Predicted Bitcoin Closing Prices")
plt.legend()
plt.show()


#LAG FEATURES
# Create lag features
df["Close_Lag_1"] = df["Close"].shift(1)
df["Close_Lag_2"] = df["Close"].shift(2)
df["Close_Lag_3"] = df["Close"].shift(3)

# Drop rows with NaN values (first 3 rows)
df = df.dropna()

# Save the modified dataset
save_path = "B:\\DS\\MACHINELEARNING\\supervised ml\\btc_usd_lagged.csv"
df.to_csv(save_path, index=True)
print(f"Lag features added successfully! Saved as: {save_path}")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")


# Define features and target
features = ["Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Volatility"]
target = "Close"

# Split data
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=False)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf.predict(X_test_scaled)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Random Forest - MAE: {mae:.2f}")
print(f"Random Forest - R² Score: {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(12, 5))
plt.plot(df["Date"].iloc[-len(y_test):], y_test, label="Actual Price", color="blue")
plt.plot(df["Date"].iloc[-len(y_test):], y_pred, label="Predicted Price", linestyle="dashed", color="red")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Actual vs Predicted Bitcoin Closing Prices (Random Forest)")
plt.legend()
plt.show()

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best model
tuned_rf = grid_search.best_estimator_
y_tuned_pred = tuned_rf.predict(X_test_scaled)

# Evaluate tuned model
mae_tuned = mean_absolute_error(y_test, y_tuned_pred)
r2_tuned = r2_score(y_test, y_tuned_pred)
print(f"Tuned Random Forest - MAE: {mae_tuned:.2f}")
print(f"Tuned Random Forest - R² Score: {r2_tuned:.4f}")


#XGBOOST
# Initialize XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

# Train the model
xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluate the model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost - MAE: {mae_xgb:.2f}")
print(f"XGBoost - R² Score: {r2_xgb:.4f}")

# Plot Actual vs Predicted Prices
plt.figure(figsize=(12, 5))
plt.plot(df['Date'].iloc[-len(y_test):], y_test, label="Actual Price", color="blue")
plt.plot(df['Date'].iloc[-len(y_test):], y_pred_xgb, label="Predicted Price", linestyle="dashed", color="red")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Actual vs Predicted Bitcoin Closing Prices (XGBoost)")
plt.legend()
plt.show()


