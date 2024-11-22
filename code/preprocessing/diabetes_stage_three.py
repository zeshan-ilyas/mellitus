# diabetes_stage_three.py

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def apply_validation_rules(df, validation_rules):
    """
    Applies validation rules to the DataFrame.
    Adds 'preprocessed_flag' and 'failed_reason' columns.
    """
    print("Applying validation rules...")
    
    # Initialize columns
    df['preprocessed_flag'] = 'Y'
    df['failed_reason'] = ''
    
    for column, rules in validation_rules.items():
        if column not in df.columns:
            # If column is missing and it's mandatory, flag it
            if rules.get('mandatory', False):
                df['preprocessed_flag'] = 'N'
                df['failed_reason'] += f"Missing mandatory column: {column}; "
            continue  # Skip non-existing non-mandatory columns
        
        col_type = rules['type']
        allowed_values = rules.get('allowed_values', [])
        min_val = rules.get('min', None)
        max_val = rules.get('max', None)
        mandatory = rules.get('mandatory', False)
        reason = rules.get('reason', 'Invalid value')
        
        # Check for missing mandatory fields
        if mandatory:
            missing = df[column].isnull()
            if missing.any():
                df.loc[missing, 'preprocessed_flag'] = 'N'
                df.loc[missing, 'failed_reason'] += f"{reason} (missing); "
        
        # Proceed only if not missing
        not_missing = df[column].notnull()
        
        if col_type == 'categorical':
            # Normalize string case and strip whitespace
            df.loc[not_missing, column] = df.loc[not_missing, column].str.lower().str.strip()
            invalid = ~df.loc[not_missing, column].isin([val.lower() for val in allowed_values])
            if invalid.any():
                df.loc[invalid, 'preprocessed_flag'] = 'N'
                df.loc[invalid, 'failed_reason'] += f"{reason}; "
        
        elif col_type == 'numerical':
            if min_val is not None:
                invalid_min = not_missing & (df[column] < min_val)
                if invalid_min.any():
                    df.loc[invalid_min, 'preprocessed_flag'] = 'N'
                    df.loc[invalid_min, 'failed_reason'] += f"{reason} (below min); "
            if max_val is not None:
                invalid_max = not_missing & (df[column] > max_val)
                if invalid_max.any():
                    df.loc[invalid_max, 'preprocessed_flag'] = 'N'
                    df.loc[invalid_max, 'failed_reason'] += f"{reason} (above max); "
        
        elif col_type == 'binary':
            invalid = ~df.loc[not_missing, column].isin(allowed_values)
            if invalid.any():
                df.loc[invalid, 'preprocessed_flag'] = 'N'
                df.loc[invalid, 'failed_reason'] += f"{reason}; "
    
    # Replace empty failed_reason with NaN
    df['failed_reason'] = df['failed_reason'].replace('', np.nan)
    
    print("Validation rules applied.\n")
    return df

def evaluate_model(y_true, y_pred, y_proba, evaluation_folder):
    """
    Evaluate classification model performance and save metrics.
    """
    print("Evaluating model performance...")
    
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Ensure evaluation folder exists
    if not os.path.exists(evaluation_folder):
        os.makedirs(evaluation_folder)
    
    plt.savefig(os.path.join(evaluation_folder, 'roc_curve.png'))
    plt.close()
    print(f"ROC Curve saved as '{os.path.join(evaluation_folder, 'roc_curve.png')}'.\n")

def validate_and_evaluate(df, target_col, evaluation_folder):
    """
    Performs validation and model evaluation.
    """
    # Define validation rules with original categorical columns
    validation_rules = {
        'gender': {  # Changed from 'gender_encoded' to 'gender'
            'type': 'categorical',
            'allowed_values': ['male', 'female'],
            'mandatory': False,
            'reason': 'Invalid gender'
        },
        'age': {
            'type': 'numerical',
            'min': 0,
            'max': 120,  # Increased max age to 120 to accommodate older individuals
            'mandatory': False,
            'reason': 'Invalid age'
        },
        'hypertension': {
            'type': 'binary',
            'allowed_values': [0, 1],
            'mandatory': False,
            'reason': 'Invalid hypertension'
        },
        'diabetes_pedigree_function': {
            'type': 'numerical',
            'min': 0.0,  # Adjusted min to 0.0 as per standard
            'max': 2.42,
            'mandatory': False,
            'reason': 'Invalid diabetes_pedigree_function'
        },
        'diet_type': {  # Changed from 'diet_type_encoded' to 'diet_type'
            'type': 'categorical',
            'allowed_values': [
                'vegetarian', 'vegan', 'low carb', 'mediterranean',
                'standard american diet', 'gluten free', 'pescatarian',
                'carnivore', 'free', 'paleo', 'raw food', 'ketogenic',
                'atkins', 'weight watchers'
            ],
            'mandatory': False,
            'reason': 'Invalid diet_type'
        },
        'star_sign': {  # Changed from 'star_sign_encoded' to 'star_sign'
            'type': 'categorical',
            'allowed_values': [
                'aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo',
                'libra', 'scorpio', 'sagittarius', 'capricorn',
                'aquarius', 'pisces'
            ],
            'mandatory': False,
            'reason': 'Invalid star_sign'
        },
        'BMI': {
            'type': 'numerical',
            'min': 10.0,  # Adjusted min to 10.0
            'max': 70.0,  # Adjusted max to 70.0 for realistic BMI values
            'mandatory': False,
            'reason': 'Invalid BMI'
        },
        'weight': {
            'type': 'numerical',
            'min': 30,  # Assuming minimum weight to be 30 kg
            'max': 300,  # Assuming maximum weight to be 300 kg
            'mandatory': False,
            'reason': 'Invalid weight'
        },
        'family_diabetes_history': {
            'type': 'binary',
            'allowed_values': [0, 1],
            'mandatory': False,
            'reason': 'Invalid family_diabetes_history'
        },
        'social_media_usage': {  # Changed from 'social_media_usage_encoded' to 'social_media_usage'
            'type': 'categorical',
            'allowed_values': ['never', 'rarely', 'occasionally', 'moderate', 'excessive'],
            'mandatory': False,
            'reason': 'Invalid social_media_usage'
        },
        'physical_activity_level': {  # Changed from 'physical_activity_level_encoded' to 'physical_activity_level'
            'type': 'categorical',
            'allowed_values': [
                'sedentary', 'lightly active', 'moderately active',
                'very active', 'extremely active'
            ],
            'mandatory': False,
            'reason': 'Invalid physical_activity_level'
        },
        'sleep_duration': {
            'type': 'numerical',
            'min': 4,
            'max': 12,  # Increased max sleep duration to 12 hours
            'mandatory': False,
            'reason': 'Invalid sleep_duration'
        },
        'stress_level': {  # Changed from 'stress_level_encoded' to 'stress_level'
            'type': 'categorical',
            'allowed_values': ['low', 'moderate', 'elevated', 'high', 'extreme'],
            'mandatory': False,
            'reason': 'Invalid stress_level'
        },
        'pregnancies': {
            'type': 'binary',
            'allowed_values': [0, 1],
            'mandatory': False,
            'reason': 'Invalid pregnancies'
        },
        'alcohol_consumption': {  # Changed from 'alcohol_consumption_encoded' to 'alcohol_consumption'
            'type': 'categorical',
            'allowed_values': ['none', 'light', 'moderate', 'heavy'],
            'mandatory': False,
            'reason': 'Invalid alcohol_consumption'
        },
        'diabetes': {
            'type': 'binary',
            'allowed_values': [0, 1],
            'mandatory': False,
            'reason': 'Invalid diabetes'
        }
    }
    
    def validate_and_evaluate(df, target_col, evaluation_folder):
        """
        Performs validation and model evaluation.
        """
        # Apply validation rules
        df = apply_validation_rules(df, validation_rules)
        
        # Handle invalid rows
        print("Handling invalid rows based on validation rules...")
        initial_count = df.shape[0]
        df_valid = df[df['preprocessed_flag'] == 'Y'].copy()
        final_count = df_valid.shape[0]
        removed_count = initial_count - final_count
        print(f"Removed {removed_count} invalid rows. Remaining rows: {final_count}")
        
        # Drop validation columns
        df_valid.drop(columns=['preprocessed_flag', 'failed_reason'], inplace=True)
        
        # Define features and target variable
        features = df_valid.drop(columns=[target_col, 'imputed_columns', 'predicted_diabetes_flag']).columns.tolist()
        X = df_valid[features]
        y = df_valid[target_col]
        
        # Split into train and test sets (ensure no data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        print(f"\nTraining set size: {X_train.shape[0]} samples")
        print(f"Testing set size: {X_test.shape[0]} samples")
        
        # Initialize and train the classifier
        print("\nTraining the RandomForestClassifier model...")
        classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
        classifier.fit(X_train, y_train)
        print("Model training completed.")
        
        # Predict on test set
        print("\nMaking predictions on the test set...")
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:,1]
        
        # Evaluate the model
        evaluate_model(y_test, y_pred, y_proba, evaluation_folder)
        
        # Return the classifier for saving
        return classifier

# To allow importing from this script without running the main code
if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('cleaned_data_stage_two.csv')
    classifier = validate_and_evaluate(df, 'diabetes', 'model_evaluation')
    # Optionally, save the classifier
    import joblib
    joblib.dump(classifier, 'random_forest_classifier.joblib')
    print("Trained model saved as 'random_forest_classifier.joblib'.")