# -*- coding: utf-8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#loading dataset
file_path = r"B:\\DS\MYSQL\\ECOMMERCE DATASET.csv"
ecommerce_data = pd.read_csv("B:\\DS\\MYSQL\\ECOMMERCE DATASET.csv")

#inspecting data
print(ecommerce_data.head())
print(ecommerce_data.info())
print(ecommerce_data.describe())

#check for missing values
print("\nMissing Values: ")
print(ecommerce_data.isnull().sum())

print("Column Names in Dataset: ")
print(list(ecommerce_data.columns))

ecommerce_data.columns = ecommerce_data.columns.str.strip()

print("Cleaned Column Names in Dataset:")
print(list(ecommerce_data.columns))

# Make a copy of the dataset
ecommerce_data_cleaned = ecommerce_data.copy()

#Clean the 'actual_price' column
ecommerce_data_cleaned['actual_price'] = ecommerce_data_cleaned['actual_price'].replace(
    ',', '', regex=True
) 
# Remove commas
ecommerce_data_cleaned['actual_price'] = pd.to_numeric(
    ecommerce_data_cleaned['actual_price'], errors='coerce'
)  # Convert to numeric, coerce invalid to NaN

#Group by 'category' and calculate the median actual price
category_median_price = ecommerce_data_cleaned.groupby('category')['actual_price'].median()


# Fill missing 'actual_price' values
ecommerce_data_cleaned['actual_price'] = ecommerce_data_cleaned['actual_price'].fillna(
    ecommerce_data_cleaned['category'].map(category_median_price)
)

# 2. Handle 'average_rating': Fill with the overall median rating
median_rating = ecommerce_data_cleaned['average_rating'].median()
ecommerce_data_cleaned['average_rating'] = ecommerce_data_cleaned['average_rating'].fillna(median_rating)

# 3. Handle 'brand': Fill missing brands with 'Unknown'
ecommerce_data_cleaned['brand'] = ecommerce_data_cleaned['brand'].fillna('Unknown')


# 4. Handle 'description': Fill missing descriptions with 'No description available'
ecommerce_data_cleaned['description'] = ecommerce_data_cleaned['description'].fillna('No description available')

# 5. Handle 'discount': Fill with 0 (assuming missing discount means no discount)
ecommerce_data_cleaned['discount'] = ecommerce_data_cleaned['discount'].fillna(0)

# 6. Handle 'seller': Fill missing sellers with 'Unknown Seller'
ecommerce_data_cleaned['seller'] = ecommerce_data_cleaned['seller'].fillna('Unknown Seller')

# Remove commas
ecommerce_data_cleaned['selling_price'] = pd.to_numeric(
    ecommerce_data_cleaned['selling_price'], errors='coerce'
)  # Convert to numeric, coerce invalid to NaN

# 7. Handle 'selling_price': Fill with the median price of the column
median_selling_price = ecommerce_data_cleaned['selling_price'].median()
ecommerce_data_cleaned['selling_price'] = ecommerce_data_cleaned['selling_price'].fillna(median_selling_price)


# Confirm the missing values are handled
print("\nMissing Values: ")
print(ecommerce_data_cleaned.isnull().sum())

print("Current Working Directory:", os.getcwd())

file_path = r"B:\DS\MYSQL\cleaned_ecommerce_dataset.csv"
ecommerce_data_cleaned.to_csv(file_path, index=False)
print("Cleaned dataset saved as '{file_path}'.")

#analysis
# Top 5 categories with the most products
top_categories = ecommerce_data_cleaned['category'].value_counts().head(5)
print("Top 5 Categories with Most Products:\n", top_categories)

# Brands with the highest average ratings
top_brands_ratings = ecommerce_data_cleaned.groupby('brand')['average_rating'].mean().sort_values(ascending=False).head(5)
print("\nTop 5 Brands by Average Ratings:\n", top_brands_ratings)

# Products with the highest discounts
# Ensure 'discount' column is numeric
ecommerce_data_cleaned['discount'] = pd.to_numeric(ecommerce_data_cleaned['discount'], errors='coerce')

# Drop rows where 'discount' is NaN after conversion
ecommerce_data_cleaned = ecommerce_data_cleaned.dropna(subset=['discount'])

# Top 5 products with the highest discounts
top_discounts = ecommerce_data_cleaned.nlargest(5, 'discount')[['title', 'discount']]
print("Top 5 Products with Highest Discounts:\n", top_discounts)

# Total selling price per category
category_sales = ecommerce_data_cleaned.groupby('category')['selling_price'].sum().sort_values(ascending=False)
print("\nTotal Selling Price per Category:\n", category_sales)

# Average selling price per brand
brand_avg_price = ecommerce_data_cleaned.groupby('brand')['selling_price'].mean().sort_values(ascending=False).head(5)
print("\nTop 5 Brands by Average Selling Price:\n", brand_avg_price)

# Temporal analysis on crawled_at
ecommerce_data_cleaned['crawled_at'] = pd.to_datetime(ecommerce_data_cleaned['crawled_at'])
daily_sales = ecommerce_data_cleaned.groupby(ecommerce_data_cleaned['crawled_at'].dt.date)['selling_price'].sum()
print("\nDaily Sales Trends:\n", daily_sales)

#top categories
top_categories = ecommerce_data_cleaned['category'].value_counts().head(5)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis")
plt.title("Top Categories by Product Count", fontsize=18)
plt.xlabel("Product Count", fontsize=13)
plt.ylabel("Category", fontsize=13)
plt.show()

#top brand
top_brands_ratings = ecommerce_data_cleaned.groupby('brand')['average_rating'].mean().nlargest(5)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_brands_ratings.values, y=top_brands_ratings.index, palette="coolwarm")
plt.title("Top Brands by Average Ratings", fontsize=18)
plt.xlabel("Average Rating", fontsize=13)
plt.ylabel("Brand", fontsize=13)
plt.show()

#dicount vs selling prices
plt.figure(figsize=(10, 6))
sns.scatterplot(data=ecommerce_data_cleaned, x='discount', y='selling_price', alpha=0.6, color='blue')
plt.title("Discounts vs Selling Prices", fontsize=17)
plt.xlabel("Discount", fontsize=13)
plt.ylabel("Selling Price", fontsize=13)
plt.show()

#product availability(Out of Stock)
availability_counts = ecommerce_data_cleaned['out_of_stock'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(availability_counts, labels=["In Stock", "Out of Stock"], autopct='%1.1f%%', startangle=140, colors=["green", "red"])
plt.title("Product Availability", fontsize=16)
plt.show()

#2. Trend Analysis
# Convert crawled_at to datetime
ecommerce_data_cleaned['crawled_at'] = pd.to_datetime(ecommerce_data_cleaned['crawled_at'], errors='coerce')

# Extract Year-Month for trend analysis
ecommerce_data_cleaned['year_month'] = ecommerce_data_cleaned['crawled_at'].dt.to_period('M')


ecommerce_data_cleaned['selling_price'] = pd.to_numeric(ecommerce_data_cleaned['selling_price'], errors='coerce').fillna(0)

print(ecommerce_data_cleaned['year_month'].unique())

print(ecommerce_data_cleaned['selling_price'].describe())

category_trend = ecommerce_data_cleaned.groupby(['year_month', 'category'])['selling_price'].sum()
print(category_trend.head(10))  # View the first few rows of grouped data

# Ensure categories have data for all months
category_trend = (
    ecommerce_data_cleaned.groupby(['year_month', 'category'])['selling_price']
    .sum()
    .unstack(fill_value=0)  # Fill missing values with 0
)

# Plot
plt.figure(figsize=(12, 6))
category_trend.plot(kind='line', figsize=(12, 6), marker='o')
plt.title("Category-Wise Revenue Over Time", fontsize=16)
plt.xlabel("Year-Month", fontsize=12)
plt.ylabel("Total Revenue", fontsize=12)
plt.legend(title="Category", loc='upper left')
plt.grid()
plt.show()

#product popularit

# Top 10 Most Popular Products
top_products = ecommerce_data_cleaned['title'].value_counts().head(10)

# Plot
plt.figure(figsize=(10, 6))
top_products.plot(kind='bar', color='skyblue')
plt.title("Top 10 Most Popular Products", fontsize=18)
plt.xlabel("Product Title", fontsize=13)
plt.ylabel("Number of Listings", fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()


#Discount distribution
# Discount Distribution
plt.figure(figsize=(10, 6))
sns.histplot(ecommerce_data_cleaned['discount'], kde=True, bins=30, color='green')
plt.title("Discount Distribution", fontsize=18)
plt.xlabel("Discount (%)", fontsize=13)
plt.ylabel("Frequency", fontsize=13)
plt.grid(axis='y')
plt.show()

#average rating by category

# Average Ratings per Category
avg_ratings_category = ecommerce_data_cleaned.groupby('category')['average_rating'].mean().sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 6))
avg_ratings_category.plot(kind='bar', color='orange')
plt.title("Average Ratings by Category", fontsize=18)
plt.xlabel("Category", fontsize=13)
plt.ylabel("Average Rating", fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()

#stock status analysis
# Stock Status
stock_status = ecommerce_data_cleaned['out_of_stock'].value_counts()

# Plot
plt.figure(figsize=(6, 6))
stock_status.plot(kind='pie', labels=['In Stock', 'Out of Stock'], autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
plt.title("Stock Status Distribution", fontsize=16)
plt.ylabel("")
plt.show()


#revenue contribution by sub-category

# Revenue per Sub-Category
sub_category_revenue = ecommerce_data_cleaned.groupby('sub_category')['selling_price'].sum().sort_values(ascending=False).head(10)

# Plot
plt.figure(figsize=(10, 6))
sub_category_revenue.plot(kind='bar', color='purple')
plt.title("Top 10 Sub-Categories by Revenue", fontsize=18)
plt.xlabel("Sub-Category", fontsize=13)
plt.ylabel("Total Revenue", fontsize=13)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()

#Predictive analysis using linear regression
regression_data = ecommerce_data_cleaned[['actual_price', 'discount', 'average_rating', 'selling_price']].dropna()
X = regression_data[['actual_price', 'discount', 'average_rating']]
y = regression_data['selling_price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

#Correlation
print(ecommerce_data_cleaned.dtypes)

# Select numeric columns
numeric_data = ecommerce_data_cleaned.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Display the correlation matrix
print(correlation_matrix)

# correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#average rating vs selling price
plt.figure(figsize=(8, 6))
sns.scatterplot(data=ecommerce_data_cleaned, x='average_rating', y='selling_price', alpha=0.7)
plt.title("Average Rating vs. Selling Price")
plt.xlabel("Average Rating")
plt.ylabel("Selling Price")
plt.show()

#discount vs selling price
plt.figure(figsize=(8, 6))
sns.scatterplot(data=ecommerce_data_cleaned, x='discount', y='selling_price', alpha=0.7)
plt.title("Discount vs. Selling Price")
plt.xlabel("Discount (%)")
plt.ylabel("Selling Price")
plt.show()

#average rating vs discount
plt.figure(figsize=(8, 6))
sns.scatterplot(data=ecommerce_data_cleaned, x='average_rating', y='discount', alpha=0.7)
plt.title("Average Rating vs. Discount")
plt.xlabel("Average Rating")
plt.ylabel("Discount (%)")
plt.show()












