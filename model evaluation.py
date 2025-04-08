import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tkinter.filedialog import askopenfilename
from tkinter import Tk, filedialog  
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold

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

def evaluate_model(model, X_train, y_train, X_test, y_test):
    preds = model.predict(X_test)
    
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    
    print("\n📊 Model Evaluation Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
  
    print(f"\n🔁 5-Fold CV RMSE: {cv_rmse}")
    print(f"Mean CV RMSE: {cv_rmse.mean():.2f}")

    return {
      "MAE": mae,
      "MSE": mse,
      "RMSE": rmse,
      "CV_RMSE": cv_rmse.mean()
  }

results = {
    'Model': ['Linear Regression', 'Random Forest', 'Tuned RF', 'XGBoost', 'LightGBM'],
    'MAE': [1208.42, 5413.26, 5452.52, 5576.75, 19645.91],
    'R2 Score': [0.9892, 0.5432, 0.5388, 0.5263, -1.0215]
}

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(os.getcwd(), 'btc_model_evaluation_summary.csv'), index=False)
