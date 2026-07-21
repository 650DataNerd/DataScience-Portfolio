import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from tkinter.filedialog import askopenfilename
from tkinter import Tk

# Hide the tkinter root window
Tk().withdraw()

# Open file dialog
file_path = askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])

# Load the dataset
if file_path:
    df = pd.read_csv(file_path)
    print("✅ Dataset Loaded Successfully!")
    print(df.head())
else:
    print("❌ No file selected.")
    exit()

# Define output folder
output_folder = r'B:\DS\MACHINELEARNING\unsupervised ml'
os.makedirs(output_folder, exist_ok=True)

# Select relevant numerical features
features = ['Open', 'High', 'Low', 'Close']
X = df[features]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### --- PCA for Dimensionality Reduction --- ###
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

### --- KMeans Clustering --- ###
kmeans = KMeans(n_clusters=3, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Evaluate KMeans
sil_score = silhouette_score(X_scaled, df['KMeans_Cluster'])
print(f"KMeans Silhouette Score: {sil_score:.4f}")

# Save KMeans plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='KMeans_Cluster', palette='Set1')
plt.title('K-Means Clustering on BTC-USD')
plt.savefig(os.path.join(output_folder, 'kmeans_clusters.png'))
plt.show()

### --- DBSCAN for Anomaly Detection --- ###
dbscan = DBSCAN(eps=1.2, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Save DBSCAN plot
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='DBSCAN_Cluster', palette='tab10')
plt.title('DBSCAN Clustering for BTC-USD Anomalies')
plt.savefig(os.path.join(output_folder, 'dbscan_clusters.png'))
plt.show()

# Save dataset with clusters
df.to_csv(os.path.join(output_folder, 'btc_usd_clusters.csv'), index=False)

print(f"✅ Unsupervised learning results saved in: {output_folder}")
