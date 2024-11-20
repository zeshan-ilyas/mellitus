# diabetes_stage_one.py

import pandas as pd
import numpy as np
import random
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def impute_numerical_knn(df, cols, imputer):
    """
    Impute numerical columns using KNN Imputer and track imputed columns.
    """
    # Select columns to impute
    df_subset = df[cols]
    
    # Fit and transform
    imputed_data = imputer.fit_transform(df_subset)
    df_imputed = pd.DataFrame(imputed_data, columns=cols, index=df.index)
    
    # Identify rows where imputation occurred
    imputed_mask = df_subset.isnull()
    imputed_rows, imputed_cols = np.where(imputed_mask)
    
    for row, col in zip(imputed_rows, imputed_cols):
        df.at[row, 'imputed_columns'].append(cols[col])
    
    # Replace the original columns with imputed data
    df[cols] = df_imputed[cols]
    
    return df

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

def cleanse_data(df):
    """
    Performs data cleansing: imputation, handling duplicates, enforcing data types.
    """
    print("Starting Data Cleansing Stage...")
    
    # Remove duplicates
    initial_count = df.shape[0]
    df.drop_duplicates(inplace=True)
    final_count = df.shape[0]
    duplicates_removed = initial_count - final_count
    print(f"Removed {duplicates_removed} duplicate rows.")
    
    # Initialize 'imputed_columns' as empty lists
    df['imputed_columns'] = [[] for _ in range(len(df))]
    
    # Define variables
    numerical_cols = ['age', 'diabetes_pedigree_function', 'BMI', 'weight', 'sleep_duration']
    binary_cols = ['hypertension', 'family_diabetes_history', 'diabetes']
    categorical_predictive_cols = ['gender', 'diet_type', 'physical_activity_level', 'alcohol_consumption']
    categorical_random_cols = ['star_sign', 'social_media_usage', 'stress_level']
    target_regression_col = 'pregnancies'
    
    # Step 1: Impute Numerical Columns using KNN Imputer
    print("\nImputing numerical columns using KNN Imputer...")
    knn_imputer = KNNImputer(n_neighbors=5)
    df = impute_numerical_knn(df, numerical_cols, knn_imputer)
    print("Numerical columns imputed.")
    
    # Step 2: Impute 'pregnancies' using Regression Imputation
    print("\nImputing 'pregnancies' using RandomForestRegressor...")
    pregnancies_features = ['age', 'BMI', 'weight', 'gender',
                            'diet_type', 'physical_activity_level', 'stress_level', 'alcohol_consumption', 
                            'diabetes_pedigree_function', 'family_diabetes_history', 'diabetes']
    
    df = impute_pregnancies(df, 'pregnancies', pregnancies_features)
    print("'pregnancies' imputed.")
    
    # Step 3: Impute Binary Columns using Predictive Modeling
    print("\nImputing binary columns using RandomForestClassifier...")
    # Impute 'hypertension'
    hypertension_features = ['BMI', 'weight', 'diabetes_pedigree_function', 'diet_type', 
                             'family_diabetes_history', 'physical_activity_level', 
                             'sleep_duration', 'stress_level', 'alcohol_consumption', 'diabetes']
    
    df = impute_binary(df, 'hypertension', hypertension_features)
    print("'hypertension' imputed.")
    
    # Impute 'family_diabetes_history'
    family_history_features = ['BMI', 'weight', 'age', 'diabetes_pedigree_function', 'gender', 
                               'diet_type', 'physical_activity_level', 'sleep_duration', 
                               'stress_level', 'alcohol_consumption', 'hypertension', 'pregnancies', 'diabetes']
    
    df = impute_binary(df, 'family_diabetes_history', family_history_features)
    print("'family_diabetes_history' imputed.")
    
    # Impute 'diabetes'
    diabetes_features = ['age', 'BMI', 'weight', 'diabetes_pedigree_function', 'gender', 
                         'diet_type', 'physical_activity_level', 'sleep_duration', 
                         'stress_level', 'alcohol_consumption', 'hypertension', 
                         'family_diabetes_history', 'pregnancies']
    
    df = impute_binary(df, 'diabetes', diabetes_features)
    print("'diabetes' imputed.")
    
    # Step 4: Impute Categorical Columns using Predictive Modeling
    print("\nImputing categorical columns using RandomForestClassifier...")
    # Impute 'gender'
    gender_features = ['BMI', 'weight', 'pregnancies', 'diet_type', 'physical_activity_level']
    
    df = impute_categorical(df, 'gender', gender_features)
    print("'gender' imputed.")
    
    # Impute 'diet_type'
    diet_type_features = ['BMI', 'weight', 'gender', 'physical_activity_level', 
                          'pregnancies', 'family_diabetes_history', 'alcohol_consumption', 'diabetes']
    
    df = impute_categorical(df, 'diet_type', diet_type_features)
    print("'diet_type' imputed.")
    
    # Impute 'physical_activity_level'
    physical_activity_features = ['BMI', 'weight', 'pregnancies', 'age', 
                                  'diet_type', 'sleep_duration', 'stress_level', 'alcohol_consumption', 
                                  'hypertension', 'diabetes']
    
    df = impute_categorical(df, 'physical_activity_level', physical_activity_features)
    print("'physical_activity_level' imputed.")
    
    # Impute 'alcohol_consumption'
    alcohol_consumption_features = ['BMI', 'weight', 'pregnancies', 'age', 
                                    'diet_type', 'physical_activity_level', 'sleep_duration', 
                                    'stress_level', 'hypertension', 'diabetes']
    
    df = impute_categorical(df, 'alcohol_consumption', alcohol_consumption_features)
    print("'alcohol_consumption' imputed.")
    
    # Step 5: Random Imputation for Evenly Distributed Categorical Columns
    print("\nPerforming random imputation for categorical columns with even distribution...")
    # Impute 'star_sign'
    df = random_impute(df, 'star_sign')
    print("'star_sign' randomly imputed.")
    
    # Impute 'social_media_usage'
    df = random_impute(df, 'social_media_usage')
    print("'social_media_usage' randomly imputed.")
    
    # Impute 'stress_level'
    df = random_impute(df, 'stress_level')
    print("'stress_level' randomly imputed.")
    
    # Step 6: Enforce Data Types
    print("\nEnforcing data types...")
    df = enforce_data_types(df)
    print("Data types enforced.")
    
    print("\nData Cleansing Stage Completed.\n")
    return df

# To allow importing from this script without running the main code
if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('diabetes_data.csv')
    df = cleanse_data(df)
    df.to_csv('cleaned_data_stage_one.csv', index=False)
    print("Data Cleansing completed and saved as 'cleaned_data_stage_one.csv'.")