# diabetes_stage_two.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def encode_categorical(df, cols):
    """
    Encode categorical columns using LabelEncoder and store encoders if needed.
    Returns the dataframe and a dictionary of encoders.
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

def add_scaling(df, numerical_cols, encoded_cols):
    """
    Apply StandardScaler to numerical and encoded categorical columns and add them as new scaled columns.
    """
    scaler = StandardScaler()
    
    # Scale numerical columns
    scaled_numerical = scaler.fit_transform(df[numerical_cols])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=[f"{col}_scaled" for col in numerical_cols], index=df.index)
    
    # Scale encoded categorical columns
    scaled_encoded = scaler.fit_transform(df[encoded_cols])
    scaled_encoded_df = pd.DataFrame(scaled_encoded, columns=[f"{col}_scaled" for col in encoded_cols], index=df.index)
    
    # Round scaled values to 2 decimal places
    scaled_numerical_df = scaled_numerical_df.round(2)
    scaled_encoded_df = scaled_encoded_df.round(2)
    
    # Concatenate scaled columns to the dataframe
    df = pd.concat([df, scaled_numerical_df, scaled_encoded_df], axis=1)
    
    return df, scaler

def add_feature_engineering(df):
    """
    Add feature engineering columns such as 'age_group' and 'bmi_group'.
    """
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
    
    return df

def flag_diabetes_predictions(df):
    """
    Flag rows where 'diabetes' was imputed.
    """
    # Assuming 'imputed_columns' includes 'diabetes' where imputation occurred
    df['predicted_diabetes_flag'] = df['imputed_columns'].apply(lambda x: 'Y' if 'diabetes' in x else 'N')
    return df

def add_additional_columns(df):
    """
    Add 'file_id', 'row_id', and 'height' columns.
    Add metadata columns: 'created_user', 'created_dttm', 'modified_user', 'modified_dttm'.
    Rearrange columns as specified.
    """
    # Add 'file_id' column (same for all rows)
    df['file_id'] = 1  # Assuming a single file; modify as needed for multiple files
    
    # Add 'row_id' column (sequential within the file)
    df['row_id'] = np.arange(1, len(df) + 1)
    
    # Calculate 'height' using BMI and weight
    # BMI = weight (kg) / (height in meters)^2 => height = sqrt(weight / BMI)
    df['height'] = np.sqrt(df['weight'] / df['BMI'])
    df['height'] = df['height'].round(2)  # Round to 2 decimal places
    
    # Feature Engineering
    df = add_feature_engineering(df)
    
    # Flag Diabetes Predictions
    df = flag_diabetes_predictions(df)
    
    # Rearrange columns:
    # Move 'file_id' and 'row_id' to the beginning
    # Move 'age_group', 'bmi_group', and 'height' after 'diabetes' column
    # Add metadata columns at the end
    columns = df.columns.tolist()
    
    # Define new order
    new_order = ['file_id', 'row_id']
    
    # Find the index of 'diabetes' column
    try:
        diabetes_index = columns.index('diabetes')
    except ValueError:
        print("Error: 'diabetes' column not found in the DataFrame.")
        return df
    
    # Columns before and including 'diabetes'
    before_diabetes = columns[:diabetes_index + 1]
    # Columns after 'diabetes'
    after_diabetes = columns[diabetes_index + 1:]
    
    # Insert 'age_group', 'bmi_group', 'height' after 'diabetes'
    insert_columns = ['age_group', 'bmi_group', 'height']
    new_order.extend(before_diabetes)
    new_order.extend(insert_columns)
    new_order.extend(after_diabetes)
    
    # Remove duplicates if any
    new_order = list(dict.fromkeys(new_order))
    
    # Assign the new order to the DataFrame
    df = df[new_order]
    
    # Add metadata columns at the end
    df['created_user'] = 'system'  # or any default value
    df['created_dttm'] = pd.Timestamp.now()
    df['modified_user'] = 'system'  # or any default value
    df['modified_dttm'] = pd.Timestamp.now()
    
    return df

def prepare_data(df):
    """
    Orchestrates data preparation steps: encoding, scaling, feature engineering, adding additional columns.
    Returns the prepared dataframe, label encoders, and scaler.
    """
    print("Starting Data Preparation Stage...")
    
    # Define column categories
    categorical_predictive_cols = ['gender', 'diet_type', 'physical_activity_level', 'alcohol_consumption']
    categorical_random_cols = ['star_sign', 'social_media_usage', 'stress_level']
    all_categorical_cols = categorical_predictive_cols + categorical_random_cols
    
    # Encode categorical variables
    df, le_categorical = encode_categorical(df, all_categorical_cols)
    print("Categorical variables encoded.")
    
    # Define numerical and encoded columns to scale
    numerical_cols = ['age', 'diabetes_pedigree_function', 'BMI', 'weight', 'sleep_duration']
    target_regression_col = 'pregnancies'
    numerical_to_scale = numerical_cols + [target_regression_col]
    encoded_categorical_cols = [f"{col}_encoded" for col in all_categorical_cols]
    
    # Apply scaling
    df, scaler = add_scaling(df, numerical_to_scale, encoded_categorical_cols)
    print("Numerical and encoded categorical features scaled.")
    
    # Add additional columns and rearrange
    df = add_additional_columns(df)
    print("Additional columns added and rearranged.")
    
    print("Data Preparation Stage Completed.\n")
    return df, le_categorical, scaler

# To allow importing from this script without running the main code
if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('cleaned_data_stage_one.csv')
    df, le_categorical, scaler = prepare_data(df)
    df.to_csv('cleaned_data_stage_two.csv', index=False)
    print("Data Preparation completed and saved as 'cleaned_data_stage_two.csv'.")