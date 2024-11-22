# Import necessary libraries
import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('diabetes_data.csv')

# Display the first few rows
print("First five rows of the dataset:")
print(data.head())

# Get summary of missing data
print("\nSummary of missing data:")
print(data.isnull().sum())

# Visualize missing data pattern
plt.figure(figsize=(12,6))
msno.matrix(data)
plt.title('Missing Data Matrix')
plt.show()

# Visualize missing data heatmap
plt.figure(figsize=(12,6))
msno.heatmap(data)
plt.title('Missing Data Heatmap')
plt.show()

# Create missingness indicators for variables with missing data
missingness_columns = data.columns[data.isnull().any()]
print("\nVariables with missing data:")
print(missingness_columns)

for col in missingness_columns:
    data[col + '_missing'] = data[col].isnull().astype(int)

# Perform Little's MCAR test approximation
# Note: Little's MCAR test is not readily available in Python.
# We'll perform a chi-squared test as an approximation.

# Function to perform chi-squared test for MCAR
def mcar_test(data):
    # Drop rows with all missing values
    data_non_missing = data.dropna(axis=0, how='all')
    # Create a binary variable indicating if any data is missing in the row
    data_non_missing['any_missing'] = data_non_missing.isnull().any(axis=1).astype(int)
    # Perform chi-squared test between each variable and the missingness indicator
    chi2_results = []
    for col in data_non_missing.columns:
        if col != 'any_missing' and not col.endswith('_missing'):
            if data_non_missing[col].dtype == 'object':
                # Categorical variable
                contingency_table = pd.crosstab(data_non_missing[col], data_non_missing['any_missing'])
                chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
                chi2_results.append({'Variable': col, 'Chi2': chi2, 'p-value': p})
            else:
                # Numerical variable
                groups = data_non_missing.groupby('any_missing')[col]
                try:
                    t_stat, p = stats.ttest_ind(groups.get_group(0).dropna(), groups.get_group(1).dropna())
                    chi2_results.append({'Variable': col, 't-stat': t_stat, 'p-value': p})
                except:
                    continue
    chi2_df = pd.DataFrame(chi2_results)
    return chi2_df

# Run the MCAR approximation test
print("\nMCAR Test Results:")
mcar_results = mcar_test(data)
print(mcar_results)

# Interpret MCAR test results
significant_vars = mcar_results[mcar_results['p-value'] <= 0.05]['Variable'].tolist()
if significant_vars:
    print("\nVariables significantly related to missingness (p <= 0.05):")
    print(significant_vars)
    print("\nData is not MCAR since missingness is related to observed variables.")
else:
    print("\nNo variables significantly related to missingness. Data may be MCAR.")

# Logistic Regression Analysis
print("\nLogistic Regression Analysis:")

# For each variable with missing data, model missingness as a function of observed variables
for col in missingness_columns:
    # Dependent variable: missingness indicator
    y = data[col + '_missing']
    # Independent variables: other observed variables
    X = data.drop(columns=[col, col + '_missing'])
    # Drop columns with missing data
    X = X.dropna(axis=1)
    # Convert categorical variables to dummy variables
    X = pd.get_dummies(X, drop_first=True)
    # Ensure no missing data in X
    X = X.fillna(X.mean())
    # Add constant term for intercept
    X = sm.add_constant(X)
    # Fit logistic regression model
    logit_model = sm.Logit(y, X).fit(disp=0)
    # Display summary
    print(f"\nLogistic Regression for missingness of '{col}':")
    print(logit_model.summary())
    # Identify significant predictors
    significant_predictors = logit_model.pvalues[logit_model.pvalues <= 0.05].index.tolist()
    significant_predictors = [var for var in significant_predictors if var != 'const']
    if significant_predictors:
        print(f"Significant predictors of missingness in '{col}': {significant_predictors}")
    else:
        print(f"No significant predictors of missingness in '{col}'.")

# Interpretation based on logistic regression results
# If missingness can be predicted by observed variables, data is MAR
# If not, and MCAR test was not significant, data may be NMAR

# Final Interpretation
print("\n--- Final Interpretation ---")

if significant_vars:
    print("Based on the MCAR test, missingness is related to observed data.")
    print("Data is likely Missing at Random (MAR).")
else:
    print("Based on the MCAR test, missingness is not related to observed data.")
    print("Data may be Missing Completely at Random (MCAR).")

# Note: Distinguishing between MAR and NMAR requires deeper analysis and domain knowledge.
print("\nNote: Determining if data is Not Missing at Random (NMAR) is challenging and often requires domain expertise.")