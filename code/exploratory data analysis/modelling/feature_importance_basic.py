# ================================
# Feature Importance Analysis Script
# ================================

# -------------------------------
# Section 1: Import Libraries
# -------------------------------

import os  # For directory operations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# -------------------------------
# Section 2: Define Plot Saving Directory
# -------------------------------

# Define the directory name
plot_dir = 'ml_insights_initial'

# Create the directory if it doesn't exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Directory '{plot_dir}' created for saving plots.")
else:
    print(f"Directory '{plot_dir}' already exists.")

# -------------------------------
# Section 3: Load and Explore the Data
# -------------------------------

# Load the dataset
data = pd.read_csv('cleaned_data_final.csv')

# Display the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Check the data types and missing values
print("\nDataset Information:")
print(data.info())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# -------------------------------
# Section 4: Data Preprocessing
# -------------------------------

# a. Exclude Non-Encoded and Non-Scaled Original Columns

# List of all original (non-encoded and non-scaled) columns to exclude
exclude_cols = [
    'file_id', 'row_id', 'gender', 'age', 'hypertension', 
    'diabetes_pedigree_function', 'diet_type', 'star_sign', 
    'BMI', 'weight', 'family_diabetes_history', 'social_media_usage', 
    'physical_activity_level', 'sleep_duration', 'stress_level', 
    'pregnancies', 'alcohol_consumption', 'age_group', 
    'bmi_group', 'height', 'imputed_columns', 
    'predicted_diabetes_flag', 'created_user', 'created_dttm', 
    'modified_user', 'modified_dttm'
]

# Alternatively, select features programmatically based on naming convention
# Assuming all encoded and scaled features contain '_encoded_scaled' or '_scaled'
feature_cols = [col for col in data.columns if ('_encoded_scaled' in col) or ('_scaled' in col)]

# Manually verify if any encoded/scaled features are missing
# If you prefer to specify them manually, uncomment the following lines:
# feature_cols = [
#     'gender_encoded_scaled',
#     'diet_type_encoded_scaled',
#     'physical_activity_level_encoded_scaled',
#     'alcohol_consumption_encoded_scaled',
#     'star_sign_encoded_scaled',
#     'social_media_usage_encoded_scaled',
#     'stress_level_encoded_scaled',
#     'age_scaled',
#     'diabetes_pedigree_function_scaled',
#     'BMI_scaled',
#     'weight_scaled',
#     'sleep_duration_scaled',
#     'pregnancies_scaled'
# ]

# Define feature matrix X and target vector y
X = data[feature_cols]
y = data['diabetes']

# Verify the selected features
print("\nSelected Feature Columns:")
print(X.columns.tolist())

# b. Handle Missing Values (If Any)

# Check for missing values in features
print("\nMissing Values in Features:")
print(X.isnull().sum())

# If missing values exist, handle them (example: imputation)
# Uncomment the following lines if you have missing values
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy='median')
# X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# -------------------------------
# Section 5: Split the Data
# -------------------------------

# Split the dataset into training and testing sets
# Stratify to maintain the proportion of classes in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# -------------------------------
# Section 6: Feature Importance with Random Forest
# -------------------------------

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Get feature importances
rf_importances = rf_model.feature_importances_
rf_features = X.columns

# Create a DataFrame for feature importances
rf_feature_importances = pd.DataFrame({
    'Feature': rf_features,
    'Importance': rf_importances
}).sort_values(by='Importance', ascending=False)

print("\nRandom Forest Feature Importances:")
print(rf_feature_importances)

# -------------------------------
# Section 7: Feature Importance with XGBoost
# -------------------------------

# Initialize the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Get feature importances
xgb_importances = xgb_model.feature_importances_
xgb_features = X.columns

# Create a DataFrame for feature importances
xgb_feature_importances = pd.DataFrame({
    'Feature': xgb_features,
    'Importance': xgb_importances
}).sort_values(by='Importance', ascending=False)

print("\nXGBoost Feature Importances:")
print(xgb_feature_importances)

# -------------------------------
# Section 8: Feature Importance with Logistic Regression
# -------------------------------

# Initialize the Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
lr_model.fit(X_train, y_train)

# Get coefficients and take absolute values for importance
lr_coefficients = lr_model.coef_[0]
lr_features = X.columns

# Create a DataFrame for feature importances
lr_feature_importances = pd.DataFrame({
    'Feature': lr_features,
    'Importance': np.abs(lr_coefficients)
}).sort_values(by='Importance', ascending=False)

print("\nLogistic Regression Feature Importances (Absolute Coefficients):")
print(lr_feature_importances)

# -------------------------------
# Section 9: Visualizing and Saving Feature Importances
# -------------------------------

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# a. Random Forest Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=rf_feature_importances, palette='viridis')
plt.title('Random Forest Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
# Save the plot
rf_plot_path = os.path.join(plot_dir, 'random_forest_feature_importances.png')
plt.savefig(rf_plot_path)
print(f"Random Forest feature importance plot saved to '{rf_plot_path}'.")
plt.show()

# b. XGBoost Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=xgb_feature_importances, palette='magma')
plt.title('XGBoost Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
# Save the plot
xgb_plot_path = os.path.join(plot_dir, 'xgboost_feature_importances.png')
plt.savefig(xgb_plot_path)
print(f"XGBoost feature importance plot saved to '{xgb_plot_path}'.")
plt.show()

# c. Logistic Regression Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=lr_feature_importances, palette='coolwarm')
plt.title('Logistic Regression Feature Importances (Absolute Coefficients)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
# Save the plot
lr_plot_path = os.path.join(plot_dir, 'logistic_regression_feature_importances.png')
plt.savefig(lr_plot_path)
print(f"Logistic Regression feature importance plot saved to '{lr_plot_path}'.")
plt.show()

# -------------------------------
# Section 10: Interpretation and Insights
# -------------------------------

# Function to display top features from each model
def display_top_features(model_importances, model_name, top_n=10):
    print(f"\nTop {top_n} Features in {model_name}:")
    print(model_importances.head(top_n))

# Display top 10 features for each model
display_top_features(rf_feature_importances, "Random Forest")
display_top_features(xgb_feature_importances, "XGBoost")
display_top_features(lr_feature_importances, "Logistic Regression")

# -------------------------------
# Section 11: Conclusion and Next Steps
# -------------------------------

print("\nFeature importance analysis completed successfully.")

print("\nNext Steps:")
print("1. Use the identified important features to build more refined and optimized models.")
print("2. Perform hyperparameter tuning to enhance model performance.")
print("3. Implement cross-validation to ensure model robustness and generalizability.")
print("4. Explore advanced feature importance techniques like SHAP for deeper insights.")
print("5. Consider handling class imbalance if applicable using techniques like SMOTE or class weighting.")
print("6. Deploy the trained model into applications or dashboards for real-time predictions.")
print("7. Review and utilize the saved plots from the 'ml_insights_initial' folder for reporting and further analysis.")

# Optional: Save feature importance plots (Already saved above)
# The plots have been saved in the 'ml_insights_initial' folder as PNG files.