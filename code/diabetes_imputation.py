import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# 1. Load the dataset
df = pd.read_csv('diabetes_data.csv')

# 2. Initialize tracking columns
df['imputed_columns'] = [[] for _ in range(len(df))]
df['predicted_diabetes_flag'] = 'N'

# 3. Define column categories
numerical_cols = ['age', 'diabetes_pedigree_function', 'BMI', 'weight', 'sleep_duration']
binary_cols = ['hypertension', 'family_diabetes_history', 'diabetes']
categorical_predictive_cols = ['gender', 'diet_type', 'physical_activity_level', 'alcohol_consumption']
categorical_random_cols = ['star_sign', 'social_media_usage', 'stress_level']
target_regression_col = 'pregnancies'

# 4. Impute Numerical Columns using KNN Imputer
def impute_numerical_knn(df, cols, imputer):
    """
    Impute numerical columns using KNN Imputer and track imputed columns.
    """
    # Select columns to impute
    df_subset = df[cols]
    
    # Fit and transform
    imputed_data = imputer.fit_transform(df_subset)
    df_imputed = pd.DataFrame(imputed_data, columns=cols)
    
    # Identify rows where imputation occurred
    imputed_mask = df_subset.isnull()
    imputed_rows, imputed_cols = np.where(imputed_mask)
    
    for row, col in zip(imputed_rows, imputed_cols):
        df.at[row, 'imputed_columns'].append(cols[col])
    
    # Replace the original columns with imputed data
    for col in cols:
        df[col] = df_imputed[col]
    
    return df

# Initialize KNN Imputer
knn_imputer = KNNImputer(n_neighbors=5)

# Impute numerical columns
df = impute_numerical_knn(df, numerical_cols, knn_imputer)

# 5. Impute 'pregnancies' using Regression Imputation
def impute_pregnancies(df, target_col, feature_cols):
    """
    Impute 'pregnancies' using RandomForestRegressor based on specified feature columns.
    """
    # Split data into train and predict
    train_df = df[df[target_col].notnull()]
    predict_df = df[df[target_col].isnull()]
    
    if predict_df.empty:
        return df
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col]
    X_predict = predict_df[feature_cols].copy()
    
    # Encode categorical features by replacing NaNs with 'Missing' before fitting
    le_dict = {}
    for col in X_train.select_dtypes(include=['object', 'category']).columns:
        X_train[col] = X_train[col].fillna('Missing').astype(str)
        X_predict[col] = X_predict[col].fillna('Missing').astype(str)
        
        le = LabelEncoder()
        le.fit(X_train[col])  # Fit on X_train[col], which includes 'Missing'
        X_train[col] = le.transform(X_train[col])
        X_predict[col] = le.transform(X_predict[col])
        le_dict[col] = le
    
    # Handle any remaining missing values in features with KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)
    X_predict_imputed = imputer.transform(X_predict)
    
    # Initialize and train the regressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    regressor.fit(X_train_imputed, y_train)
    
    # Predict missing values
    y_pred = regressor.predict(X_predict_imputed)
    
    # Assign predictions to the dataframe
    df.loc[df[target_col].isnull(), target_col] = y_pred.round().astype(int)
    
    # Track imputed columns
    imputed_indices = predict_df.index
    for idx in imputed_indices:
        df.at[idx, 'imputed_columns'].append(target_col)
    
    return df

# Define features for 'pregnancies' imputation
pregnancies_features = ['age', 'BMI', 'weight', 'gender',
                        'diet_type', 'physical_activity_level', 'stress_level', 'alcohol_consumption', 
                        'diabetes_pedigree_function', 'family_diabetes_history', 'diabetes']

df = impute_pregnancies(df, 'pregnancies', pregnancies_features)

# 6. Impute Binary Columns using Predictive Modeling
def impute_binary(df, target_col, feature_cols):
    """
    Impute binary columns using RandomForestClassifier based on specified feature columns.
    """
    # Split data into train and predict
    train_df = df[df[target_col].notnull()]
    predict_df = df[df[target_col].isnull()]
    
    if predict_df.empty:
        return df
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col]
    X_predict = predict_df[feature_cols].copy()
    
    # Encode categorical features by replacing NaNs with 'Missing' before fitting
    le_dict = {}
    for col in X_train.select_dtypes(include=['object', 'category']).columns:
        X_train[col] = X_train[col].fillna('Missing').astype(str)
        X_predict[col] = X_predict[col].fillna('Missing').astype(str)
        
        le = LabelEncoder()
        le.fit(X_train[col])  # Fit on X_train[col], which includes 'Missing'
        X_train[col] = le.transform(X_train[col])
        X_predict[col] = le.transform(X_predict[col])
        le_dict[col] = le
    
    # Handle any remaining missing values in features by KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)
    X_predict_imputed = imputer.transform(X_predict)
    
    # Initialize and train the classifier
    classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
    classifier.fit(X_train_imputed, y_train)
    
    # Predict missing values
    y_pred = classifier.predict(X_predict_imputed)
    
    # Assign predictions to the dataframe
    df.loc[df[target_col].isnull(), target_col] = y_pred
    
    # Track imputed columns
    imputed_indices = predict_df.index
    for idx in imputed_indices:
        df.at[idx, 'imputed_columns'].append(target_col)
    
    return df

# Impute 'hypertension'
hypertension_features = ['BMI', 'weight', 'diabetes_pedigree_function', 'diet_type', 
                         'family_diabetes_history', 'physical_activity_level', 
                         'sleep_duration', 'stress_level', 'alcohol_consumption', 'diabetes']

df = impute_binary(df, 'hypertension', hypertension_features)

# Impute 'family_diabetes_history'
family_history_features = ['BMI', 'weight', 'age', 'diabetes_pedigree_function', 'gender', 
                           'diet_type', 'physical_activity_level', 'sleep_duration', 
                           'stress_level', 'alcohol_consumption', 'hypertension', 'pregnancies', 'diabetes']

df = impute_binary(df, 'family_diabetes_history', family_history_features)

# Impute 'diabetes'
diabetes_features = ['age', 'BMI', 'weight', 'diabetes_pedigree_function', 'gender', 
                     'diet_type', 'physical_activity_level', 'sleep_duration', 
                     'stress_level', 'alcohol_consumption', 'hypertension', 
                     'family_diabetes_history', 'pregnancies']

df = impute_binary(df, 'diabetes', diabetes_features)

# 7. Impute Categorical Columns using Predictive Modeling
def impute_categorical(df, target_col, feature_cols):
    """
    Impute categorical columns using RandomForestClassifier based on specified feature columns.
    """
    # Split data into train and predict
    train_df = df[df[target_col].notnull()]
    predict_df = df[df[target_col].isnull()]
    
    if predict_df.empty:
        return df
    
    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col]
    X_predict = predict_df[feature_cols].copy()
    
    # Encode categorical features by replacing NaNs with 'Missing' before fitting
    le_dict = {}
    for col in X_train.select_dtypes(include=['object', 'category']).columns:
        X_train[col] = X_train[col].fillna('Missing').astype(str)
        X_predict[col] = X_predict[col].fillna('Missing').astype(str)
        
        le = LabelEncoder()
        le.fit(X_train[col])  # Fit on X_train[col], which includes 'Missing'
        X_train[col] = le.transform(X_train[col])
        X_predict[col] = le.transform(X_predict[col])
        le_dict[col] = le
    
    # Handle any remaining missing values in features by KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)
    X_predict_imputed = imputer.transform(X_predict)
    
    # Initialize and train the classifier
    classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
    classifier.fit(X_train_imputed, y_train)
    
    # Predict missing values
    y_pred = classifier.predict(X_predict_imputed)
    
    # Assign predictions to the dataframe
    df.loc[df[target_col].isnull(), target_col] = y_pred
    
    # Track imputed columns
    imputed_indices = predict_df.index
    for idx in imputed_indices:
        df.at[idx, 'imputed_columns'].append(target_col)
    
    return df

# Impute 'gender'
gender_features = ['BMI', 'weight', 'pregnancies', 'diet_type', 'physical_activity_level']

df = impute_categorical(df, 'gender', gender_features)

# Impute 'diet_type'
diet_type_features = ['BMI', 'weight', 'gender', 'physical_activity_level', 
                      'pregnancies', 'family_diabetes_history', 'alcohol_consumption', 'diabetes']

df = impute_categorical(df, 'diet_type', diet_type_features)

# Impute 'physical_activity_level'
physical_activity_features = ['BMI', 'weight', 'pregnancies', 'age', 
                              'diet_type', 'sleep_duration', 'stress_level', 'alcohol_consumption', 
                              'hypertension', 'diabetes']

df = impute_categorical(df, 'physical_activity_level', physical_activity_features)

# Impute 'alcohol_consumption'
alcohol_consumption_features = ['BMI', 'weight', 'pregnancies', 'age', 
                                'diet_type', 'physical_activity_level', 'sleep_duration', 
                                'stress_level', 'hypertension', 'diabetes']

df = impute_categorical(df, 'alcohol_consumption', alcohol_consumption_features)

# 8. Random Imputation for Evenly Distributed Categorical Columns
def random_impute(df, col):
    """
    Randomly impute missing categorical values based on existing distribution.
    """
    # Calculate the distribution excluding NaNs
    distribution = df[col].value_counts(normalize=True)
    categories = distribution.index.tolist()
    probabilities = distribution.values.tolist()
    
    # Identify missing indices
    missing_indices = df[df[col].isnull()].index
    if len(missing_indices) == 0:
        return df
    
    # Randomly assign categories based on distribution
    imputed_values = np.random.choice(categories, size=len(missing_indices), p=probabilities)
    df.loc[missing_indices, col] = imputed_values
    
    # Track imputed columns
    for idx in missing_indices:
        df.at[idx, 'imputed_columns'].append(col)
    
    return df

# Impute 'star_sign'
df = random_impute(df, 'star_sign')

# Impute 'social_media_usage'
df = random_impute(df, 'social_media_usage')

# Impute 'stress_level'
df = random_impute(df, 'stress_level')

# 9. Enforce Data Types
def enforce_data_types(df):
    """
    Enforce specified data types for each column after imputation.
    """
    df['age'] = df['age'].round().astype(int)
    df['diabetes_pedigree_function'] = df['diabetes_pedigree_function'].round(2)
    df['BMI'] = df['BMI'].round(1)
    df['weight'] = df['weight'].round(1)
    df['sleep_duration'] = df['sleep_duration'].round(1)
    df['pregnancies'] = df['pregnancies'].round().astype(int)
    
    # Ensure binary columns are integers (0 or 1)
    binary_columns = ['hypertension', 'family_diabetes_history', 'diabetes']
    for col in binary_columns:
        df[col] = df[col].round().astype(int)
    
    return df

df = enforce_data_types(df)

# 10. Encode Categorical Variables into New Columns
def encode_categorical(df, cols):
    """
    Encode categorical columns and store them in new columns without altering the original columns.
    """
    le_dict = {}
    for col in cols:
        le = LabelEncoder()
        # Fill NaNs with 'Missing' to handle encoding
        df[col] = df[col].fillna('Missing').astype(str)
        encoded_col = f"{col}_encoded"
        df[encoded_col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

# Encode all categorical columns and store in new columns
all_categorical_cols = categorical_predictive_cols + categorical_random_cols
df, le_categorical = encode_categorical(df, all_categorical_cols)

# 11. Create 'imputed_columns' as lists of unique imputed columns per row
df['imputed_columns'] = df['imputed_columns'].apply(lambda x: list(set(x)) if isinstance(x, list) else [])

# 12. Add Scaling (Normalization) to Numerical and Encoded Columns
def add_scaling(df, numerical_cols, encoded_cols):
    """
    Apply normalization to numerical and encoded categorical columns and store them as new columns.
    """
    scaler = StandardScaler()
    
    # Scale numerical columns
    scaled_numerical = scaler.fit_transform(df[numerical_cols])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=[f"{col}_scaled" for col in numerical_cols])
    
    # Scale encoded categorical columns
    scaled_encoded = scaler.fit_transform(df[encoded_cols])
    scaled_encoded_df = pd.DataFrame(scaled_encoded, columns=[f"{col}_scaled" for col in encoded_cols])
    
    # Round scaled values to 2 decimal places
    scaled_numerical_df = scaled_numerical_df.round(2)
    scaled_encoded_df = scaled_encoded_df.round(2)
    
    # Concatenate scaled columns to the dataframe
    df = pd.concat([df, scaled_numerical_df, scaled_encoded_df], axis=1)
    
    return df

# Define numerical and encoded columns to scale
numerical_to_scale = numerical_cols + [target_regression_col]
encoded_categorical_cols = [f"{col}_encoded" for col in all_categorical_cols]
df = add_scaling(df, numerical_to_scale, encoded_categorical_cols)

# 13. Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate classification model performance and print metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("-----------------------------------\n")

# Function to perform evaluation for each imputation step
def perform_evaluation(df, target_col, feature_cols, model_type='classification'):
    """
    Perform model training, prediction, and evaluation.
    """
    # Drop rows where target is still missing
    data = df.dropna(subset=[target_col])
    
    X = data[feature_cols]
    y = data[target_col]
    
    # Encode categorical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    le_dict = {}
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    # Handle any remaining missing values in features with KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    if model_type == 'classification':
        # Initialize and train classifier
        classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
        classifier.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = classifier.predict(X_test)
        
        # Evaluate
        evaluate_model(y_test, y_pred, f"{target_col} Classifier")
        
    elif model_type == 'regression':
        # Initialize and train regressor
        regressor = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        regressor.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = regressor.predict(X_test)
        
        # Evaluate using regression metrics
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"--- {target_col} Regressor Evaluation ---")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R^2 Score: {r2:.4f}")
        print("-----------------------------------\n")
    
    return

# Create a folder for evaluation plots
evaluation_folder = 'hybrid_model_evaluation'
if not os.path.exists(evaluation_folder):
    os.makedirs(evaluation_folder)

# Perform evaluation for 'diabetes' imputation model
perform_evaluation(df, 'diabetes', diabetes_features, model_type='classification')

# 14. Generate Distribution Comparisons
def plot_distributions(original_df, imputed_df, cols, folder):
    """
    Plot and save distribution comparisons for specified columns.
    """
    for col in cols:
        plt.figure(figsize=(10, 5))
        if col in original_df.columns:
            sns.kdeplot(original_df[col].dropna(), label='Original', shade=True)
        sns.kdeplot(imputed_df[col].dropna(), label='Imputed', shade=True)
        plt.title(f'Distribution Comparison for {col}')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{col}_distribution_comparison.png"))
        plt.close()

# Select columns to compare distributions
# For example: numerical columns and encoded categorical columns
columns_to_plot = numerical_cols + encoded_categorical_cols

# To compare, we need to have copies of the original data before imputation
# Reload the original data
df_original = pd.read_csv('diabetes_data.csv')

# Generate distribution plots
plot_distributions(df_original, df, columns_to_plot, evaluation_folder)

# 15. Add 'age_group' and 'bmi_group' Columns
def categorize_age(age):
    if age < 30:
        return 'young_adult'
    elif 30 <= age < 60:
        return 'middle_aged'
    else:
        return 'senior'

def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'underweight'
    elif 18.5 <= bmi < 25:
        return 'normal'
    elif 25 <= bmi < 30:
        return 'overweight'
    else:
        return 'obese'

df['age_group'] = df['age'].apply(categorize_age)
df['bmi_group'] = df['BMI'].apply(categorize_bmi)

# 16. Update 'predicted_diabetes_flag' Correctly
def flag_diabetes_predictions(df):
    """
    Flag rows where 'diabetes' was imputed.
    """
    # Assuming 'imputed_columns' includes 'diabetes' where imputation occurred
    df['predicted_diabetes_flag'] = df['imputed_columns'].apply(lambda x: 'Y' if 'diabetes' in x else 'N')
    return df

df = flag_diabetes_predictions(df)

# 17. Final Missing Value Check
print("Missing values after imputation:")
print(df.isnull().sum())

# 18. Export the cleaned and imputed dataset
df.to_csv('cleaned_data_hybrid_v0.1.csv', index=False)
print("\nImputed dataset saved as 'cleaned_data_hybrid_v0.1.csv'.")

# 19. Save Evaluation Plots (Already saved in the plotting function)
print(f"\nEvaluation plots saved in the '{evaluation_folder}' folder.")