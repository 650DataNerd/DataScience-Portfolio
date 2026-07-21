import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tkinter.filedialog import askopenfilename
from tkinter import Tk, filedialog  
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

# Ensure Date column is in datetime format
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# Define features and target variable
features = ['Previous_Close', '7Day_MA', '30Day_MA', 'Volatility', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3']
target = 'Close'

# Drop missing values
df = df.dropna()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, shuffle=False)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train LightGBM model
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=10)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"LightGBM - MAE: {mae:.2f}")
print(f"LightGBM - R² Score: {r2:.4f}")

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(df["Date"].iloc[-len(y_test):], y_test, label="Actual Price", color="blue")
plt.plot(df["Date"].iloc[-len(y_test):], y_pred, label="Predicted Price", linestyle="dashed", color="red")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.title("Actual vs Predicted Bitcoin Closing Prices (LightGBM)")
plt.legend()
plt.show()