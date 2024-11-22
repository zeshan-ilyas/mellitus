import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import xlsxwriter
import numpy as np

# Set the style for seaborn
sns.set(style="whitegrid")

# Define paths
data_file = 'diabetes_data.csv'
summary_dir = 'summary'
visualizations_dir = os.path.join(summary_dir, 'visualisations')
excel_file = os.path.join(summary_dir, 'summary.xlsx')

# Create directories if they don't exist
os.makedirs(visualizations_dir, exist_ok=True)

# Load the dataset
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Error: The file '{data_file}' was not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: The file '{data_file}' is empty.")
    exit(1)
except Exception as e:
    print(f"An unexpected error occurred while loading the data: {e}")
    exit(1)

# Display initial data information
print("Initial Data Shape:", data.shape)
print("Initial Missing Values:\n", data.isnull().sum())

# Calculate Height from BMI and Weight
# BMI = weight (kg) / (height (m))^2
# Thus, height (m) = sqrt(weight / BMI)
# Handle cases where BMI or weight <=0 or NaN

# Function to calculate height in cm
def calculate_height(weight, bmi):
    try:
        if weight > 0 and bmi > 0:
            height_m = np.sqrt(weight / bmi)
            height_cm = height_m * 100
            return height_cm
        else:
            return np.nan
    except:
        return np.nan

# Apply the function to calculate height
data['height_cm'] = data.apply(lambda row: calculate_height(row['weight'], row['BMI']), axis=1)

# Display data information after height calculation
print("\nAfter calculating height:")
print("Missing Values:\n", data.isnull().sum())

# Create Age Groups
# Define age bins and labels
age_bins = [0, 18, 35, 50, 65, 80]
age_labels = ['Child', 'Young Adult', 'Adult', 'Middle Age', 'Senior']
data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels, right=False)

# Create BMI Groups
bmi_bins = [0, 18.5, 24.9, 29.9, np.inf]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
data['bmi_group'] = pd.cut(data['BMI'], bins=bmi_bins, labels=bmi_labels, right=False)

# List of variables to analyze (excluding the target variable 'diabetes')
variables = [
    'gender', 'age', 'hypertension', 'diabetes_pedigree_function',
    'diet_type', 'star_sign', 'BMI', 'weight',
    'family_diabetes_history', 'social_media_usage',
    'physical_activity_level', 'sleep_duration',
    'stress_level', 'pregnancies', 'alcohol_consumption',
    'height_cm', 'age_group', 'bmi_group'
]

# Function to create bar plots for categorical variables
def plot_categorical(variable, data_subset, save_dir):
    plt.figure(figsize=(10,6))
    sns.countplot(data=data_subset, x=variable, hue='diabetes', palette='Set1')
    plt.title(f'Distribution of {variable.replace("_", " ").title()} by Diabetes Status')
    plt.xlabel(variable.replace('_', ' ').title())
    plt.ylabel('Count')
    plt.legend(title='Diabetes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_filename = f'{variable}_barplot.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Function to create box plots for numerical variables
def plot_numerical(variable, data_subset, save_dir):
    plt.figure(figsize=(10,6))
    # Removed palette to fix FutureWarning
    sns.boxplot(x='diabetes', y=variable, data=data_subset)
    plt.title(f'{variable.replace("_", " ").title()} by Diabetes Status')
    plt.xlabel('Diabetes')
    plt.ylabel(variable.replace('_', ' ').title())
    plt.tight_layout()
    plot_filename = f'{variable}_boxplot.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Function to create histogram for numerical variables
def plot_histogram(variable, data_subset, save_dir):
    plt.figure(figsize=(10,6))
    sns.histplot(data=data_subset, x=variable, hue='diabetes', kde=True, palette='Set1', element='step')
    plt.title(f'Histogram of {variable.replace("_", " ").title()} by Diabetes Status')
    plt.xlabel(variable.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.legend(title='Diabetes')
    plt.tight_layout()
    plot_filename = f'{variable}_histogram.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Function to create violin plots for numerical variables
def plot_violin(variable, data_subset, save_dir):
    plt.figure(figsize=(10,6))
    # Removed palette and hue to fix FutureWarning
    sns.violinplot(x='diabetes', y=variable, data=data_subset)
    plt.title(f'Violin Plot of {variable.replace("_", " ").title()} by Diabetes Status')
    plt.xlabel('Diabetes')
    plt.ylabel(variable.replace('_', ' ').title())
    plt.tight_layout()
    plot_filename = f'{variable}_violinplot.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Function to create scatter plots for Weight vs BMI
def plot_scatter(weight_var, bmi_var, data_subset, save_dir):
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=data_subset, x=weight_var, y=bmi_var, hue='diabetes', palette='Set1', alpha=0.6)
    plt.title(f'Scatter Plot of {weight_var.replace("_", " ").title()} vs {bmi_var.replace("_", " ").title()} by Diabetes Status')
    plt.xlabel(weight_var.replace('_', ' ').title())
    plt.ylabel(bmi_var.replace('_', ' ').title())
    plt.legend(title='Diabetes')
    plt.tight_layout()
    plot_filename = f'{weight_var}_vs_{bmi_var}_scatter.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# Function to create count plot for Physical Activity Level among non-diabetics
def plot_physical_activity_non_diabetic(data_subset, save_dir):
    # Filter for non-diabetic individuals (assuming 'diabetes' == 0 indicates no diabetes)
    non_diabetic = data_subset[data_subset['diabetes'] == 0]
    if non_diabetic.empty:
        print("No non-diabetic data available for Physical Activity Level analysis. Skipping plot.")
        return None
    
    plt.figure(figsize=(10,6))
    sns.countplot(data=non_diabetic, x='physical_activity_level', palette='Set2')
    plt.title('Physical Activity Levels Among Non-Diabetic Individuals')
    plt.xlabel('Physical Activity Level')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_filename = 'physical_activity_non_diabetic_countplot.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print("Physical Activity Level - Non-Diabetic Count Plot created.")
    return plot_path

# Function to create pie chart for Physical Activity Level distribution
def plot_physical_activity_pie(data_subset, save_dir):
    activity_counts = data_subset['physical_activity_level'].value_counts()
    plt.figure(figsize=(8,8))
    activity_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set2'))
    plt.title('Physical Activity Level Distribution Among All Individuals')
    plt.ylabel('')
    plt.tight_layout()
    plot_filename = 'physical_activity_pie_chart.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print("Physical Activity Level - Pie Chart created.")
    return plot_path

# Function to create box plot for Stress Level
def plot_stress_boxplot(data_subset, save_dir):
    plt.figure(figsize=(10,6))
    # Removed palette to fix FutureWarning
    sns.boxplot(x='diabetes', y='stress_level', data=data_subset)
    plt.title('Stress Level by Diabetes Status')
    plt.xlabel('Diabetes')
    plt.ylabel('Stress Level')
    plt.tight_layout()
    plot_filename = 'stress_level_boxplot.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print("Stress Level Box Plot created.")
    return plot_path

# Function to create pie chart for Stress Level distribution
def plot_stress_pie_chart(data_subset, save_dir):
    stress_counts = data_subset['stress_level'].value_counts()
    plt.figure(figsize=(8,8))
    stress_counts.plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette('Set1'))
    plt.title('Stress Level Distribution Among All Individuals')
    plt.ylabel('')
    plt.tight_layout()
    plot_filename = 'stress_level_pie_chart.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print("Stress Level Pie Chart created.")
    return plot_path

# Function to create count plot with percentage for Stress Level
def plot_stress_count_percentage(data_subset, save_dir):
    plt.figure(figsize=(10,6))
    stress_counts = data_subset['stress_level'].value_counts(normalize=True) * 100
    sns.barplot(x=stress_counts.index, y=stress_counts.values, palette='Set1')
    plt.title('Stress Level Distribution (Percentage)')
    plt.xlabel('Stress Level')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_filename = 'stress_level_percentage.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print("Stress Level Percentage Count Plot created.")
    return plot_path

# Function to create correlation heatmap
def plot_correlation_heatmap(corr_matrix, title, save_dir, filename):
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Correlation Heatmap '{title}' created.")
    return plot_path

# Function to determine and plot based on variable type with non-null filtering
def visualize(variable, data, save_dir):
    # Filter data to include only non-null values for the variable and 'diabetes'
    data_subset = data.dropna(subset=[variable, 'diabetes'])
    if data_subset.empty:
        print(f"No non-null data available for variable: {variable}. Skipping plot.")
        return None
    
    if data_subset[variable].dtype == 'object' or data_subset[variable].nunique() < 15:
        plot_path = plot_categorical(variable, data_subset, save_dir)
        print(f"Bar plot created for categorical variable: {variable}")
    else:
        plot_path = plot_numerical(variable, data_subset, save_dir)
        print(f"Box plot created for numerical variable: {variable}")
    return plot_path

# Store plot paths for embedding into Excel
plot_paths = {}

# Iterate through each variable and create the appropriate plot
for var in variables:
    plot_path = visualize(var, data, visualizations_dir)
    if plot_path:  # Only add if plot was created
        plot_paths[var] = plot_path

# Additional Visualizations

# 1. BMI: Histogram and Violin Plot
bmi_data = data.dropna(subset=['BMI', 'diabetes'])
if not bmi_data.empty:
    bmi_histogram = plot_histogram('BMI', bmi_data, visualizations_dir)
    plot_paths['BMI_histogram'] = bmi_histogram
    
    bmi_violin = plot_violin('BMI', bmi_data, visualizations_dir)
    plot_paths['BMI_violinplot'] = bmi_violin
else:
    print("No non-null data available for BMI additional plots.")

# 2. Weight: Histogram and Scatter Plot with BMI
weight_data = data.dropna(subset=['weight', 'BMI', 'diabetes'])
if not weight_data.empty:
    weight_histogram = plot_histogram('weight', weight_data, visualizations_dir)
    plot_paths['weight_histogram'] = weight_histogram
    
    weight_scatter = plot_scatter('weight', 'BMI', weight_data, visualizations_dir)
    plot_paths['weight_vs_BMI_scatter'] = weight_scatter
else:
    print("No non-null data available for Weight additional plots.")

# 3. Physical Activity Level: Count Plot for Non-Diabetics and Pie Chart
physical_activity_data = data.dropna(subset=['physical_activity_level', 'diabetes'])
if not physical_activity_data.empty:
    physical_activity_non_diabetic_plot = plot_physical_activity_non_diabetic(physical_activity_data, visualizations_dir)
    if physical_activity_non_diabetic_plot:
        plot_paths['physical_activity_non_diabetic_count'] = physical_activity_non_diabetic_plot
    
    # Corrected function name
    physical_activity_pie = plot_physical_activity_pie(physical_activity_data, visualizations_dir)
    plot_paths['physical_activity_pie_chart'] = physical_activity_pie
else:
    print("No non-null data available for Physical Activity Level additional plots.")

# 4. Stress Level: Box Plot and Pie Chart
stress_data = data.dropna(subset=['stress_level', 'diabetes'])
if not stress_data.empty:
    stress_boxplot = plot_stress_boxplot(stress_data, visualizations_dir)
    plot_paths['stress_level_boxplot'] = stress_boxplot
    
    stress_pie_chart = plot_stress_pie_chart(stress_data, visualizations_dir)
    plot_paths['stress_level_pie_chart'] = stress_pie_chart
    
    stress_percentage = plot_stress_count_percentage(stress_data, visualizations_dir)
    plot_paths['stress_level_percentage'] = stress_percentage
else:
    print("No non-null data available for Stress Level additional plots.")

# 5. Age Groups: Distribution by Diabetes Status
def plot_age_group_distribution(data_subset, save_dir):
    plt.figure(figsize=(10,6))
    sns.countplot(data=data_subset, x='age_group', hue='diabetes', palette='Set1')
    plt.title('Distribution of Diabetes Status Across Age Groups')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.legend(title='Diabetes')
    plt.tight_layout()
    plot_filename = 'age_group_distribution.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print("Age Group Distribution Plot created.")
    return plot_path

age_group_data = data.dropna(subset=['age_group', 'diabetes'])
if not age_group_data.empty:
    age_group_plot = plot_age_group_distribution(age_group_data, visualizations_dir)
    plot_paths['age_group_distribution'] = age_group_plot
else:
    print("No non-null data available for Age Group Distribution Plot.")

# 6. BMI Groups: Distribution by Diabetes Status
def plot_bmi_group_distribution(data_subset, save_dir):
    plt.figure(figsize=(10,6))
    sns.countplot(data=data_subset, x='bmi_group', hue='diabetes', palette='Set1')
    plt.title('Distribution of Diabetes Status Across BMI Groups')
    plt.xlabel('BMI Group')
    plt.ylabel('Count')
    plt.legend(title='Diabetes')
    plt.tight_layout()
    plot_filename = 'bmi_group_distribution.png'
    plot_path = os.path.join(save_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    print("BMI Group Distribution Plot created.")
    return plot_path

bmi_group_data = data.dropna(subset=['bmi_group', 'diabetes'])
if not bmi_group_data.empty:
    bmi_group_plot = plot_bmi_group_distribution(bmi_group_data, visualizations_dir)
    plot_paths['bmi_group_distribution'] = bmi_group_plot
else:
    print("No non-null data available for BMI Group Distribution Plot.")

# Interaction Matrices

# 1. Lifestyle: BMI, Weight, Physical Activity Level, Stress Level, Alcohol Consumption
lifestyle_vars = ['BMI', 'weight', 'physical_activity_level', 'stress_level', 'alcohol_consumption']
lifestyle_data = data[lifestyle_vars].dropna()

if not lifestyle_data.empty:
    # Encode categorical variables using one-hot encoding
    lifestyle_encoded = pd.get_dummies(lifestyle_data, columns=['physical_activity_level', 'stress_level', 'alcohol_consumption'])
    corr_matrix_lifestyle = lifestyle_encoded.corr()
    lifestyle_corr_heatmap = plot_correlation_heatmap(
        corr_matrix_lifestyle,
        'Lifestyle Interaction Matrix',
        visualizations_dir,
        'lifestyle_interaction_matrix.png'
    )
    plot_paths['lifestyle_interaction_matrix'] = lifestyle_corr_heatmap
else:
    print("No non-null data available for Lifestyle Interaction Matrix.")

# 2. Genetic: Diabetes Pedigree Function, Family Diabetes History
genetic_vars = ['diabetes_pedigree_function', 'family_diabetes_history']
genetic_data = data[genetic_vars].dropna()

if not genetic_data.empty:
    corr_matrix_genetic = genetic_data.corr()
    genetic_corr_heatmap = plot_correlation_heatmap(
        corr_matrix_genetic,
        'Genetic Interaction Matrix',
        visualizations_dir,
        'genetic_interaction_matrix.png'
    )
    plot_paths['genetic_interaction_matrix'] = genetic_corr_heatmap
else:
    print("No non-null data available for Genetic Interaction Matrix.")

# Create an Excel workbook and embed all visualizations
try:
    workbook = xlsxwriter.Workbook(excel_file)
    worksheet_summary = workbook.add_worksheet('Summary')
    
    # Write a summary table with hyperlinks
    worksheet_summary.write('A1', 'Variable')
    worksheet_summary.write('B1', 'Plot Path')
    
    for row_num, (var, path) in enumerate(plot_paths.items(), start=1):
        worksheet_summary.write(row_num, 0, var)
        # Create an internal hyperlink to the corresponding sheet
        # Sheet names in Excel are limited to 31 characters and cannot contain certain characters
        sheet_name = var[:31].replace('/', '_').replace('\\', '_').replace('*', '_') \
                         .replace('?', '_').replace('[', '_').replace(']', '_')
        # Add the worksheet before linking
        worksheet = workbook.add_worksheet(sheet_name)
        
        # Insert the image
        worksheet.insert_image('B2', path, {'x_scale': 0.8, 'y_scale': 0.8})
        
        # Add title
        if 'interaction_matrix' in var:
            title = var.replace('_', ' ').title()
        elif var == 'age_group_distribution':
            title = 'Age Group Distribution by Diabetes Status'
        elif var == 'bmi_group_distribution':
            title = 'BMI Group Distribution by Diabetes Status'
        elif var.endswith('_histogram'):
            title = f'Histogram for {var.replace("_histogram", "").replace("_", " ").title()}'
        elif var.endswith('_boxplot'):
            title = f'Box Plot for {var.replace("_boxplot", "").replace("_", " ").title()}'
        elif var.endswith('_violinplot'):
            title = f'Violin Plot for {var.replace("_violinplot", "").replace("_", " ").title()}'
        elif var.endswith('_scatter'):
            title = f'Scatter Plot for {var.replace("_scatter", "").replace("_", " ").title()}'
        elif var.endswith('_count'):
            title = f'Count Plot for {var.replace("_count", "").replace("_", " ").title()}'
        elif var.endswith('_pie_chart'):
            title = f'Pie Chart for {var.replace("_pie_chart", "").replace("_", " ").title()}'
        elif var.endswith('_percentage'):
            title = f'Percentage Plot for {var.replace("_percentage", "").replace("_", " ").title()}'
        else:
            title = f'Visualization for {var.replace("_", " ").title()}'
        
        worksheet.write('A1', title)
        
        # Add hyperlink in the summary sheet to the new sheet
        # Excel internal hyperlink format: '#SheetName!A1'
        worksheet_summary.write_url(row_num, 1, f"internal:'{sheet_name}'!A1", string=path)
    
    # Adjust column widths
    worksheet_summary.set_column(0, 0, 40)
    worksheet_summary.set_column(1, 1, 80)
    
    # Close the workbook
    workbook.close()
    
    print(f"\nAll visualizations have been saved in the '{visualizations_dir}' folder and embedded in '{excel_file}'.")
except Exception as e:
    print(f"An error occurred while creating the Excel workbook: {e}")