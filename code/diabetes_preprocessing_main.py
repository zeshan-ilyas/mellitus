# diabetes_preprocessing_main.py

import pandas as pd
import os
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import custom modules
from diabetes_stage_one import cleanse_data
from diabetes_stage_two import prepare_data
from diabetes_stage_three import validate_and_evaluate

def main():
    # Define file paths
    raw_data_path = 'diabetes_data.csv'
    cleaned_data_stage_one_path = 'cleaned_data_stage_one.csv'
    cleaned_data_stage_two_path = 'cleaned_data_stage_two.csv'
    final_cleaned_data_path = 'cleaned_data_final.csv'
    model_path = 'random_forest_classifier.joblib'
    scaler_path = 'scaler.joblib'
    evaluation_folder = 'model_evaluation'
    
    # 1. Load the dataset
    print("Loading the dataset...")
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Error: '{raw_data_path}' not found. Please ensure the file is in the correct directory.")
        return
    
    print("First five rows of the dataset:")
    print(df.head())
    
    print("\nSummary of missing data:")
    print(df.isnull().sum())
    
    # 2. Data Cleansing
    df = cleanse_data(df)
    
    # 3. Save Intermediate Cleaned Data
    df.to_csv(cleaned_data_stage_one_path, index=False)
    print(f"\nIntermediate cleaned data saved as '{cleaned_data_stage_one_path}'.")
    
    # 4. Data Preparation
    df, le_categorical, scaler = prepare_data(df)
    
    # 5. Save Cleaned and Prepared Data
    df.to_csv(cleaned_data_stage_two_path, index=False)
    print(f"Cleaned and prepared data saved as '{cleaned_data_stage_two_path}'.")
    
    # 6. Data Validation and Model Evaluation
    classifier = validate_and_evaluate(df, 'diabetes', evaluation_folder)
    
    # 7. Save the Trained Model
    joblib.dump(classifier, model_path)
    print(f"Trained model saved as '{model_path}'.")
    
    # 8. Save the Scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved as '{scaler_path}'.")
    
    # 9. Save the Final Cleaned Data
    df.to_csv(final_cleaned_data_path, index=False)
    print(f"Final cleaned data saved as '{final_cleaned_data_path}'.")
    
    print("\nData Preprocessing Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()