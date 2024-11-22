# ================================
# Enhanced Feature Importance Analysis Script with Hyperparameter Tuning and Class Imbalance Handling
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_fscore_support, 
                             accuracy_score, f1_score, precision_score, recall_score)
import xgboost as xgb

# Hyperparameter tuning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# SHAP for model interpretability
import shap

# Handling class imbalance
from imblearn.over_sampling import SMOTE

# For saving models and metrics
import joblib

# For handling warnings
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Section 2: Define Plot and Model Saving Directories
# -------------------------------

# Define directories
plot_dir = 'ml_insights_enhanced'
model_dir = 'trained_models'
metrics_dir = 'model_metrics'

# Create directories if they don't exist
for directory in [plot_dir, model_dir, metrics_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

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

# Select features programmatically based on naming convention
# Assuming all encoded and scaled features contain '_encoded_scaled' or '_scaled'
feature_cols = [col for col in data.columns if ('_encoded_scaled' in col) or ('_scaled' in col)]

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
# Section 5: Split the Data and Handle Class Imbalance
# -------------------------------

# Split the dataset into training and testing sets
# Stratify to maintain the proportion of classes in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# a. Analyze Class Distribution
class_counts = y_train.value_counts()
print("\nClass Distribution in Training Set:")
print(class_counts)

# b. Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nClass Distribution after SMOTE:")
print(pd.Series(y_train_res).value_counts())

# -------------------------------
# Section 6: Baseline Model Training and Feature Importance
# -------------------------------

# Function to train models and extract feature importances
def train_and_get_importances(models, X_train, y_train, X_columns):
    feature_importances = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            importances = np.zeros(X_train.shape[1])
        feature_importances[name] = pd.DataFrame({
            'Feature': X_columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
    return feature_importances, trained_models

# Define the models with class_weight='balanced' where applicable
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42, scale_pos_weight=class_counts[0]/class_counts[1]),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
}

# Train models and get feature importances
feature_importances, trained_models = train_and_get_importances(models, X_train_res, y_train_res, X.columns)

# Display feature importances
for model_name, importance_df in feature_importances.items():
    print(f"\n{model_name} Feature Importances:")
    print(importance_df)

# -------------------------------
# Section 7: Hyperparameter Tuning with RandomizedSearchCV
# -------------------------------

print("\nStarting Hyperparameter Tuning...")

# Define parameter distributions for each model
param_distributions = {
    'Random Forest': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [3, 5, 7, 9, 11],
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4]
    },
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'saga']
    }
}

# Number of iterations for RandomizedSearchCV
n_iter_search = 50

# Initialize a dictionary to store best estimators
best_estimators = {}

# Perform RandomizedSearchCV for each model
for model_name in models.keys():
    print(f"\nTuning hyperparameters for {model_name}...")
    model = models[model_name]
    param_dist = param_distributions[model_name]
    
    # Define RandomizedSearchCV
    randomized_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring='roc_auc',
        cv=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit RandomizedSearchCV
    randomized_search.fit(X_train_res, y_train_res)
    
    # Store the best estimator
    best_estimators[model_name] = randomized_search.best_estimator_
    
    print(f"Best parameters for {model_name}: {randomized_search.best_params_}")
    print(f"Best ROC AUC score for {model_name}: {randomized_search.best_score_:.4f}")

# -------------------------------
# Section 8: Model Evaluation
# -------------------------------

print("\nEvaluating Tuned Models on Test Data...")

# Function to evaluate models and save metrics
def evaluate_model(model, X_test, y_test, model_name, plot_dir, metrics_dir):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A'
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n{model_name} Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc if roc_auc != 'N/A' else 'N/A'}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Save classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    report_path = os.path.join(metrics_dir, f"{model_name.lower().replace(' ', '_')}_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to '{report_path}'.")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    cm_plot_path = os.path.join(plot_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix saved to '{cm_plot_path}'.")
    plt.show()
    
    # ROC Curve
    if y_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc='lower right')
        plt.tight_layout()
        roc_plot_path = os.path.join(plot_dir, f'{model_name.lower().replace(" ", "_")}_roc_curve.png')
        plt.savefig(roc_plot_path)
        print(f"ROC curve saved to '{roc_plot_path}'.")
        plt.show()
    
    # Precision-Recall Curve
    if y_proba is not None:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        plt.figure(figsize=(6,4))
        plt.plot(recall_vals, precision_vals, label=f'{model_name} (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{model_name} Precision-Recall Curve')
        plt.legend(loc='upper right')
        plt.tight_layout()
        prc_plot_path = os.path.join(plot_dir, f'{model_name.lower().replace(" ", "_")}_precision_recall_curve.png')
        plt.savefig(prc_plot_path)
        print(f"Precision-Recall curve saved to '{prc_plot_path}'.")
        plt.show()
    
    # Save metrics to a DataFrame
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [accuracy],
        'ROC AUC': [roc_auc],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    })
    
    # Save metrics to CSV
    metrics_csv_path = os.path.join(metrics_dir, 'model_evaluation_metrics.csv')
    if os.path.exists(metrics_csv_path):
        metrics_df.to_csv(metrics_csv_path, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Evaluation metrics saved to '{metrics_csv_path}'.")

# Evaluate each tuned model
for model_name, model in best_estimators.items():
    evaluate_model(model, X_test, y_test, model_name, plot_dir, metrics_dir)

# -------------------------------
# Section 9: Additional Evaluation Metrics with Cross-Validation
# -------------------------------

print("\nPerforming Cross-Validation for Robustness...")

# Function to perform cross-validation and save results
def cross_validate_model(model, X, y, model_name, metrics_dir, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    print(f"{model_name} Cross-Validation ROC AUC Scores: {scores}")
    print(f"Mean ROC AUC: {scores.mean():.4f}, Std: {scores.std():.4f}")
    
    # Save cross-validation scores
    cv_scores_df = pd.DataFrame({
        'Model': [model_name]*cv,
        'Fold': list(range(1, cv+1)),
        'ROC_AUC_Score': scores
    })
    
    cv_scores_csv_path = os.path.join(metrics_dir, 'cross_validation_scores.csv')
    if os.path.exists(cv_scores_csv_path):
        cv_scores_df.to_csv(cv_scores_csv_path, mode='a', header=False, index=False)
    else:
        cv_scores_df.to_csv(cv_scores_csv_path, index=False)
    print(f"Cross-validation scores saved to '{cv_scores_csv_path}'.")

# Perform cross-validation for each tuned model
for model_name, model in best_estimators.items():
    cross_validate_model(model, X, y, model_name, metrics_dir)

# -------------------------------
# Section 10: Exploring Feature Interactions and Correlations
# -------------------------------

print("\nExploring Feature Interactions and Correlations...")

# a. Feature Correlation with Target Variable
# Combine features and target for correlation computation
correlation_df = X.copy()
correlation_df['diabetes'] = y

# Compute Pearson correlation coefficients
feature_correlations = correlation_df.corr()['diabetes'].drop('diabetes').sort_values(ascending=False)

print("\nFeature Correlations with 'diabetes':")
print(feature_correlations)

# Save the correlations to a CSV file
correlations_path = os.path.join(plot_dir, 'feature_correlations_with_diabetes.csv')
feature_correlations.to_csv(correlations_path, header=['Correlation'], index=True)
print(f"\nFeature correlations saved to '{correlations_path}'.")

# b. Visualize Feature Correlations
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_correlations.values, y=feature_correlations.index, palette='coolwarm')
plt.title('Feature Correlations with Diabetes')
plt.xlabel('Pearson Correlation Coefficient')
plt.ylabel('Feature')
plt.tight_layout()
correlation_plot_path = os.path.join(plot_dir, 'feature_correlations_with_diabetes.png')
plt.savefig(correlation_plot_path)
print(f"Feature correlation bar plot saved to '{correlation_plot_path}'.")
plt.show()

# c. Pairwise Correlation Matrix
plt.figure(figsize=(14, 12))
corr_matrix = correlation_df.drop('diabetes', axis=1).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
corr_matrix_plot_path = os.path.join(plot_dir, 'feature_correlation_matrix.png')
plt.savefig(corr_matrix_plot_path)
print(f"Feature correlation matrix saved to '{corr_matrix_plot_path}'.")
plt.show()

# -------------------------------
# Section 11: SHAP Analysis for Model Interpretability
# -------------------------------

print("\nPerforming SHAP Analysis for Model Interpretability...")

# Initialize SHAP explainer
# Select the best model based on ROC AUC from cross-validation
# For demonstration, we'll use XGBoost
shap_model = best_estimators['XGBoost']

# Create a SHAP explainer
explainer = shap.Explainer(shap_model, X_train_res)

# Compute SHAP values
shap_values = explainer(X_test)

# Summary Plot (Bar)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar)')
shap_bar_path = os.path.join(plot_dir, 'shap_feature_importance_bar.png')
plt.savefig(shap_bar_path, bbox_inches='tight')
print(f"SHAP feature importance bar plot saved to '{shap_bar_path}'.")
plt.show()

# Detailed SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot')
shap_summary_path = os.path.join(plot_dir, 'shap_summary_plot.png')
plt.savefig(shap_summary_path, bbox_inches='tight')
print(f"SHAP summary plot saved to '{shap_summary_path}'.")
plt.show()

# Dependence Plot for a specific feature
feature_to_explain = feature_importances['XGBoost'].iloc[0]['Feature']  # Top feature
plt.figure(figsize=(12, 6))
shap.dependence_plot(feature_to_explain, shap_values.values, X_test, show=False)
plt.title(f'SHAP Dependence Plot for {feature_to_explain}')
shap_dependence_path = os.path.join(plot_dir, f'shap_dependence_{feature_to_explain}.png')
plt.savefig(shap_dependence_path, bbox_inches='tight')
print(f"SHAP dependence plot for '{feature_to_explain}' saved to '{shap_dependence_path}'.")
plt.show()

# -------------------------------
# Section 12: Conclusion and Next Steps
# -------------------------------

print("\nEnhanced feature importance and correlation analysis completed successfully.")

print("\nNext Steps:")
print("1. Analyze the SHAP plots to understand the impact of each feature on the model's predictions.")
print("2. Investigate and engineer new features based on identified interactions to potentially improve model performance.")
print("3. Deploy the best-performing model with tuned hyperparameters for real-time predictions or integration into applications.")
print("4. Continuously monitor model performance and retrain with new data to maintain accuracy and relevance.")
print("5. Explore advanced interpretability techniques or ensemble methods to further enhance model insights.")
print("6. Utilize the saved plots from the 'ml_insights_enhanced' folder for reporting, presentations, and further analysis.")

# Optional: Save the best models for future use
for model_name, model in best_estimators.items():
    model_path = os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}_model.pkl")
    joblib.dump(model, model_path)
    print(f"{model_name} saved to '{model_path}'.")
