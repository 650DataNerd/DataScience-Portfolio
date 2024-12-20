import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import statsmodels.api as sm


#read file
df = pd.read_csv(r"B:\DS\PYTHON\CurrentPopulationSurvey.csv")
print(df.head())

#dataset inspection
print("first five rows of dataset: ")
print(df.head())

print("\ncolumn Names: ")
print(df.columns)

print("\nDataset Information: ")
print(df.info())

print("\nSummary Statistics: ")
print(df.describe())

print("\nMissing Values: ")
print(df.isnull().sum())


#finding columns relevant to study
keywords = ['sex', 'gender', 'wage', 'income', 'earn']

relevant_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in keywords)]
print("Relevant Columns:", relevant_columns)


#Clean and filter the data
relevant_cols = ['sex', 'incwage', 'realhrwage', 'year']
df_subset = df[relevant_cols]

#drop rows with missing values in the columns
df_clean = df_subset.dropna()

#checking cleaned dataset
print("Cleaned Dataset Info: ")
print(df_clean.info())

print("\nfirst five rows of cleaned data:")
print(df_clean.head())

#check unique values in the sex column
print("Unique values in 'sex':")
print(df_clean['sex'].unique())


#count of each gender
gender_counts = df_clean['sex'].value_counts()
print("\nGender Counts:")
print(gender_counts)


# Group data by gender and calculate average wages
gender_wage_stats = df_clean.groupby('sex')[['incwage', 'realhrwage']].mean()

#renaming gender 
gender_wage_stats.index = ['Male', 'Female']

print("Average Wages by Gender:")
print(gender_wage_stats)

#calculating percentage pay gap
gender_wage_stats['Pay Gap (%)'] = ((gender_wage_stats.loc['Male', 'incwage'] - gender_wage_stats.loc['Female', 'incwage']) 
                                    / gender_wage_stats.loc['Male', 'incwage']) * 100

print("\nGender Pay Gap (%):")
print(gender_wage_stats['Pay Gap (%)'])


#Code visualization
#Bar plot for average wages
gender_wage_stats[['incwage', 'realhrwage']].plot(kind='bar', figsize=(8, 6), color=['skyblue', 'lightcoral'])
plt.title("Average Income and Hourly Wages by Gender")
plt.ylabel("Wage")
plt.xticks(rotation=0)  # Fixed: Changed to xticks
plt.legend(["Average Annual Wage", "Average Hourly Real Wage"])
plt.show()

#Pay Gap for visualization
pay_gap = gender_wage_stats['Pay Gap (%)'].iloc[0]
plt.figure(figsize=(6, 6))
plt.bar(['Gender Pay Gap'], [pay_gap], color='orange')
plt.title("Gender Pay Gap (%)")
plt.ylabel("Percentage Gap")
plt.ylim(0, 50)
plt.show()

#analyzing trends over time

#group data by year , sex annd calculate average
wage_trends = df_clean.groupby(['year', 'sex'])[['incwage']].mean().reset_index()

#pivoting data for easy comparing
wage_pivot = wage_trends.pivot(index='year', columns='sex', values='incwage')
wage_pivot.columns = ['Male', 'Female']

#calcullate annual pay gap
wage_pivot['Pay Gap (%)'] = ((wage_pivot['Male'] - wage_pivot['Female']) / wage_pivot['Male']) * 100

print("Pay Gap Trends Over Time:")
print(wage_pivot.head())

#plot pay gap trends
plt.figure(figsize=(10, 6))
plt.plot(wage_pivot.index, wage_pivot['Male'], label='Male Average Wage', color='blue')
plt.plot(wage_pivot.index, wage_pivot['Female'], label='Female Average Wage', color='red')
plt.title("Gender Wage Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Average Annual Wage")
plt.legend()
plt.grid(True)
plt.show()

# Plot the pay gap percentage over time
plt.figure(figsize=(10, 6))
plt.plot(wage_pivot.index, wage_pivot['Pay Gap (%)'], color='orange', linestyle='--')
plt.title("Gender Pay Gap (%) Over Time")
plt.xlabel("Year")
plt.ylabel("Pay Gap (%)")
plt.grid(True)
plt.show()


# List all column names
print(df_clean.columns)

# Replace 'sex' column values with labels
df_clean['sex'] = df_clean['sex'].replace({1: 'Male', 2: 'Female'})

# Plot histograms of annual income for males and females
plt.figure(figsize=(10, 6))
sns.histplot(df_clean[df_clean['sex'] == 'Male']['incwage'], label='Male', color='blue', kde=True)
sns.histplot(df_clean[df_clean['sex'] == 'Female']['incwage'], label='Female', color='red', kde=True)
plt.title("Income Distribution by Gender")
plt.xlabel("Annual Income")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Boxplot of income by gender
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean, x='sex', y='incwage', palette='coolwarm')
plt.title("Income Distribution by Gender (Box Plot)")
plt.xlabel("Gender")
plt.ylabel("Annual Income")
plt.show()


#wage trends over time
# Group the data by 'year' and 'sex', and calculate average annual income
wage_trends = df_clean.groupby(['year', 'sex'])['incwage'].mean().reset_index()


# Pivot the data for easier plotting
wage_trends_pivot = wage_trends.pivot(index='year', columns='sex', values='incwage')


# Plot wage trends over the years
plt.figure(figsize=(10, 6))
plt.plot(wage_trends_pivot.index, wage_trends_pivot['Male'], label='Male', color='blue', marker='o')
plt.plot(wage_trends_pivot.index, wage_trends_pivot['Female'], label='Female', color='red', marker='o')
plt.title("Wage Trends Over Time by Gender")
plt.xlabel("Year")
plt.ylabel("Average Annual Income")
plt.legend()
plt.grid()
plt.show()


#Statistical test for paygap
male_wages = df_clean[df_clean['sex'] == 'Male']['incwage'].dropna()
female_wages = df_clean[df_clean['sex'] == 'Female']['incwage'].dropna()

# Perform independent t-test
t_stat, p_value = ttest_ind(male_wages, female_wages, equal_var=False)  # Welch's t-test

# Print results
print("T-Test Results:")
print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value:.4e}")



# Interpret results
alpha = 0.05
if p_value < alpha:
    print("The wage difference between males and females is statistically significant (p < 0.05).")
else:
    print("The wage difference between males and females is NOT statistically significant (p >= 0.05).")


#regression analysis for year and gender
regression_columns = ['incwage', 'sex', 'year']
df_reg = df_clean[regression_columns].dropna()


# Encode 'sex' column: 0 = Female, 1 = Male
df_reg['sex_encoded'] = df_reg['sex'].replace({'Male': 1, 'Female': 0})


# Define the regression model: Wages ~ Gender + Year
X = df_reg[['sex_encoded', 'year']]  # Independent variables
y = df_reg['incwage']  # Dependent variable (income)



# Add constant for intercept
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print regression summary
print(model.summary())


print(income_pivot.head())
print(income_pivot.columns)


plt.figure(figsize=(10, 6))

# Plot average income over time for Male and Female
plt.plot(income_pivot.index, income_pivot['Male'], label='Male', color='blue')
plt.plot(income_pivot.index, income_pivot['Female'], label='Female', color='red')

plt.title('Average Income Over Time by Gender')
plt.xlabel('Year')
plt.ylabel('Average Income')
plt.legend(title='Gender')
plt.grid()
plt.show()








