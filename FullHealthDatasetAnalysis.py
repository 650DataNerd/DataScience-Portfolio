import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

from scipy.stats import pearsonr, ttest_ind
from scipy.stats import f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


#path to files
access_to_medicine_path = "B://DS//SPSS//accesstomedicine.csv"
hospital_beds_path = "B://DS//SPSS//hospitalbeds.csv"
registered_nurses_path = "B://DS//SPSS//Registered nurses and midwives.csv"
registered_medics_path = "B://DS//SPSS//Registeredmedics.csv"

#loading the datasets
access_to_medicine = pd.read_csv(access_to_medicine_path)
hospital_beds = pd.read_csv(hospital_beds_path)
registered_nurses = pd.read_csv(registered_nurses_path)
registered_medics = pd.read_csv(registered_medics_path)

#Display of the loaded data
print(access_to_medicine.head())
print(hospital_beds.head())
print(registered_nurses.head())
print(registered_medics.head())


#Cleaning and standardizing data
access_to_medicine.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)
hospital_beds.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)
registered_nurses.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)
registered_medics.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)

#Convert year to numeric
for df in [access_to_medicine, hospital_beds, registered_nurses, registered_medics]:
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df.dropna(subset=['Year'], inplace=True)
    df['Year'] = df['Year'].astype(int)
    
    
#Checking the data info
print(access_to_medicine.info())
print(hospital_beds.info())
print(registered_nurses.info())
print(registered_medics.info())


print(access_to_medicine.columns)
print(hospital_beds.columns)
print(registered_nurses.columns)
print(registered_medics.columns)


#merging the datasets
print(access_to_medicine.shape)
print(hospital_beds.shape)
print(registered_nurses.shape)
print(registered_medics.shape)

# Merge 'access_to_medicine' and 'hospital_beds' first
merged_data = pd.merge(access_to_medicine, hospital_beds, 
                       on=['Countries,_territories_and_areas', 'Year'], 
                       how='outer')

# Merge the result with 'registered_nurses'
merged_data = pd.merge(merged_data, registered_nurses, 
                       on=['Countries,_territories_and_areas', 'Year'], 
                       how='outer')

# Finally, merge with 'registered_medics'
merged_data = pd.merge(merged_data, registered_medics, 
                       on=['Countries,_territories_and_areas', 'Year'], 
                       how='outer')

# Display the shape and the first few rows of the merged dataset
print("Shape of merged dataset:", merged_data.shape)
print(merged_data.head())

# Optional: Check for missing values and duplicates
print("Missing values per column:\n", merged_data.isnull().sum())
print("Number of duplicate rows:", merged_data.duplicated().sum())


#EDA
#summary statistics
print(merged_data.describe)

#check unique countries and years
print("Unique countries: ", merged_data['Countries,_territories_and_areas'].nunique())
print("Year range:", merged_data['Year'].min(), "to", merged_data['Year'].max())

#drop duplicate
merged_data.drop_duplicates(inplace=True)

#filling missing values
merged_data.fillna(method='ffill', inplace=True)

#verify missing values
print("Missing values after cleaning:\n", merged_data.isnull().sum())

#Data visualizations
#hospital bed per 10,000 population
plt.figure(figsize=(12, 6))
sns.lineplot(data=merged_data, x='Year', y='Hospital_beds_(per_10_000_population)', 
             hue='Countries,_territories_and_areas')
plt.title('Hospital Beds per 10,000 Population Over Time')
plt.ylabel('Hospital Beds (per 10,000)')
plt.xlabel('Year')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#medical doctors by country
top_countries = merged_data.groupby('Countries,_territories_and_areas')['Medical_doctors_(number)'].sum().nlargest(10).index
filtered_data = merged_data[merged_data['Countries,_territories_and_areas'].isin(top_countries)]
plt.figure(figsize=(12, 6))
sns.barplot(data=filtered_data, x='Countries,_territories_and_areas', y='Medical_doctors_(number)', errorbar=None)
plt.title('Top 10 Countries with Most Medical Doctors')
plt.xticks(rotation=45)
plt.ylabel('Number of Medical Doctors')
plt.xlabel('Country')
plt.show()


#correlation btn hospital beds and medical doctors
# Drop rows where either column has missing values
valid_data = merged_data[['Hospital_beds_(per_10_000_population)', 'Medical_doctors_(number)']].dropna()

# Extract the columns as arrays
hospital_beds = valid_data['Hospital_beds_(per_10_000_population)']
medical_doctors = valid_data['Medical_doctors_(number)']


# Calculate Pearson correlation
correlation, p_value = pearsonr(hospital_beds, medical_doctors)
print(f"Correlation: {correlation}, P-value: {p_value}")

#Scatterplot visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=filtered_data,
    x='Hospital_beds_(per_10_000_population)',
    y='Medical_doctors_(number)'
)
plt.title('Correlation between Hospital Beds and Medical Doctors', fontsize=16)
plt.xlabel('Hospital Beds (per 10,000 population)', fontsize=12)
plt.ylabel('Medical Doctors (number)', fontsize=12)
plt.grid(True)
plt.show()


#Check for missing data
missing_data_summary = merged_data.isnull().sum()
print("Missing Data Summary:\n", missing_data_summary)

# Imputation for key columns with minimal missing values

merged_data['Nursing_and_midwifery_personnel_(per_10_000_population)'] = (
    merged_data['Nursing_and_midwifery_personnel_(per_10_000_population)']
    .fillna(merged_data['Nursing_and_midwifery_personnel_(per_10_000_population)'].median())
)

merged_data['Medical_doctors_(number)'] = (
    merged_data['Medical_doctors_(number)']
    .fillna(merged_data['Medical_doctors_(number)'].median())
)

merged_data['Medical_doctors_(per_10_000_population)'] = (
    merged_data['Medical_doctors_(per_10_000_population)']
    .fillna(merged_data['Medical_doctors_(per_10_000_population)'].median())
)

# Confirming no missing values in the key columns
print(merged_data[['Nursing_and_midwifery_personnel_(per_10_000_population)', 
                   'Medical_doctors_(number)', 
                   'Medical_doctors_(per_10_000_population)']].isnull().sum())


# Fill remaining numeric columns with their median values
for col in merged_data.columns:
    if merged_data[col].dtype in ['float64', 'int64']:
        merged_data[col] = merged_data[col].fillna(merged_data[col].median())


print(merged_data.isnull().sum())



# Outlier Detection using Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=merged_data[['Hospital_beds_(per_10_000_population)', 'Medical_doctors_(number)']])
plt.title('Boxplot for Hospital Beds and Medical Doctors', fontsize=16)
plt.ylabel('Value', fontsize=12)
plt.show()

# Distributions of Key Variables
plt.figure(figsize=(12, 6))
sns.histplot(merged_data['Hospital_beds_(per_10_000_population)'].dropna(), kde=True, bins=30, color='blue', label='Hospital Beds')
sns.histplot(merged_data['Medical_doctors_(number)'].dropna(), kde=True, bins=30, color='orange', label='Medical Doctors')
plt.title('Distributions of Hospital Beds and Medical Doctors', fontsize=16)
plt.xlabel('Value', fontsize=12)
plt.legend()
plt.show()


# Correlation Analysis
numeric_cols = merged_data.select_dtypes(include=['float64', 'int64']).columns  # Ensure numeric_cols is defined
plt.figure(figsize=(12, 8))
sns.heatmap(merged_data[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# T-tests (Example: compare countries)
unique_countries = merged_data['Countries,_territories_and_areas'].dropna().unique()
if len(unique_countries) > 1:
    group1 = merged_data[merged_data['Countries,_territories_and_areas'] == unique_countries[0]]['Hospital_beds_(per_10_000_population)']
    group2 = merged_data[merged_data['Countries,_territories_and_areas'] == unique_countries[1]]['Hospital_beds_(per_10_000_population)']
    t_stat, p_val = ttest_ind(group1.dropna(), group2.dropna())
    print(f"T-test between {unique_countries[0]} and {unique_countries[1]}: t-statistic = {t_stat}, p-value = {p_val}")
else:
    print("Not enough unique countries for a t-test.")

# Visualizations

# Visualizations
plt.figure(figsize=(12, 6))
sns.boxplot(data=merged_data[numeric_cols])
plt.title('Boxplot of Critical Variables')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(merged_data['Hospital_beds_(per_10_000_population)'], kde=True, bins=30, color='blue', label='Hospital Beds')
sns.histplot(merged_data['Medical_doctors_(number)'], kde=True, bins=30, color='orange', label='Medical Doctors')
plt.title('Distributions of Critical Variables')
plt.legend()
plt.show()



#T-test comparing hospital bed counts between countries
country1_data = merged_data[merged_data['Countries,_territories_and_areas'] == 'Afghanistan']['Hospital_beds_(per_10_000_population)'].dropna()
country2_data = merged_data[merged_data['Countries,_territories_and_areas'] == 'Albania']['Hospital_beds_(per_10_000_population)'].dropna()

t_stat, p_val = ttest_ind(country1_data, country2_data)
print(f"T-test results: t-statistic = {t_stat}, p-value = {p_val}")



#oneway anova
#Compare hospital beds across three countries
country_a = merged_data[merged_data['Countries,_territories_and_areas'] == 'Afghanistan']['Hospital_beds_(per_10_000_population)'].dropna()
country_b = merged_data[merged_data['Countries,_territories_and_areas'] == 'Albania']['Hospital_beds_(per_10_000_population)'].dropna()
country_c = merged_data[merged_data['Countries,_territories_and_areas'] == 'Algeria']['Hospital_beds_(per_10_000_population)'].dropna()

f_stat, p_val = f_oneway(country_a, country_b, country_c)
print(f"ANOVA results: F-statistic = {f_stat}, p-value = {p_val}")


#regression analysis
X = merged_data[['Hospital_beds_(per_10_000_population)', 'Medical_doctors_(per_10_000_population)']].dropna()
y = merged_data['Nursing_and_midwifery_personnel_(per_10_000_population)'].dropna()

# Ensure matching indices after dropping NaNs
X, y = X.align(y, join='inner', axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Initialize and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))


# Display coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)


#regression modelling using statistically significant p-value
#define independent and dependent variables
X = merged_data[['Hospital_beds_(per_10_000_population)', 'Medical_doctors_(per_10_000_population)']]
y = merged_data['Nursing_and_midwifery_personnel_(per_10_000_population)']

# Add a constant for the intercept
X_with_constant = sm.add_constant(X)

# Fit the model using statsmodels for detailed outputs
model = sm.OLS(y, X_with_constant).fit()

# Display summary of the regression
print(model.summary())

# Calculate Variance Inflation Factor (VIF) for multicollinearity
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVariance Inflation Factor:")
print(vif_data)

# Residuals and diagnostics
residuals = model.resid

# Residual plot
plt.figure(figsize=(8, 6))
sns.residplot(x=model.predict(X_with_constant), y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 2})
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()


# Q-Q plot for normality of residuals
sm.qqplot(residuals, line='45', fit=True)
plt.title('Q-Q Plot of Residuals')
plt.show()



# Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y, model.predict(X_with_constant), alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()


# Histogram of residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30, color='blue')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


#fit an OLS model

def fit_ols_model(data, target, predictors):
    X = data[predictors]
    X = sm.add_constant(X)  # Add intercept
    y = data[target]
    model = sm.OLS(y, X).fit()
    return model


# Function: Diagnostic Plots
def diagnostic_plots(model, data, target, predictors):
    # Residuals
    residuals = model.resid


  # Residuals vs Fitted
    fitted_values = model.fittedvalues
    plt.scatter(fitted_values, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.show()


 # Residual Histogram
    sns.histplot(residuals, kde=True, bins=30)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.show()


# Function: Statistical Diagnostics
def statistical_diagnostics(model, data, predictors):
    # Durbin-Watson Test
    dw_stat = sm.stats.stattools.durbin_watson(model.resid)
    print(f"Durbin-Watson Statistic: {dw_stat}")
    
    # Breusch-Pagan Test
    _, pval, _, _ = het_breuschpagan(model.resid, sm.add_constant(data[predictors]))
    print(f"Breusch-Pagan Test p-value: {pval}")
    
    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = shapiro(model.resid)
    print(f"Shapiro-Wilk Test p-value: {shapiro_p}")


# Function: Calculate VIF
def calculate_vif(data, predictors):
    X = sm.add_constant(data[predictors])
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVariance Inflation Factor (VIF):")
    print(vif_data)


# Function: Automation Workflow
def regression_workflow(data, target, predictors):
    print("Fitting OLS Model...")
    model = fit_ols_model(data, target, predictors)
    print(model.summary())
    
    print("\nPerforming Diagnostic Tests...")
    statistical_diagnostics(model, data, predictors)
    
    print("\nGenerating Diagnostic Plots...")
    diagnostic_plots(model, data, target, predictors)
    
    print("\nCalculating VIF...")
    calculate_vif(data, predictors)


# Variables
target = 'Nursing_and_midwifery_personnel_(per_10_000_population)'
predictors = [
    'Hospital_beds_(per_10_000_population)',
    'Medical_doctors_(per_10_000_population)'
]


# Run Workflow
regression_workflow(merged_data, target, predictors)


#pair plot predictors
sns.pairplot(merged_data[predictors])
plt.title('Pairplot of Predictors')
plt.show()


#autocorrelation and partial
plot_acf(model.resid, lags=20)
plt.title('Autocorrelation of Residuals')
plt.show()

plot_pacf(model.resid, lags=20)
plt.title('Partial Autocorrelation of Residuals')
plt.show()



#multicollinearity using VIF
X = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

#homoscedasticity check(Breusch-Pagan Test)
_, pval, _, _ = het_breuschpagan(y_pred - y_test, X_test)
print(f"Breusch-Pagan Test p-value: {pval}")

#nomality of residuals(Q-Q Plot & Shapiro-Wilk Test)
qqplot(y_pred - y_test, line='s')
plt.show()

_, p_shapiro = shapiro(y_pred - y_test)
print(f"Shapiro-Wilk Test p-value: {p_shapiro}")































