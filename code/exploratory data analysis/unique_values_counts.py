import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Read the CSV file
df = pd.read_csv('diabetes_data.csv')

# Define the attribute lists based on your specifications
numerical_cols = [
    'age',  # integer
    'diabetes_pedigree_function',  # float, 2dp
    'BMI',  # float, 1dp
    'weight',  # float, 1dp
    'sleep_duration',  # float, 1dp
    'pregnancies'  # integer
]

categorical_cols = [
    'gender',
    'diet_type',
    'star_sign',
    'social_media_usage',
    'physical_activity_level',
    'stress_level',
    'alcohol_consumption'
]

binary_cols = [
    'hypertension',
    'family_diabetes_history',
    'diabetes'
]

# Adjust the lists to include only existing columns
existing_numerical_cols = [col for col in numerical_cols if col in df.columns]
existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
existing_binary_cols = [col for col in binary_cols if col in df.columns]

# Check for missing columns
missing_cols = set(numerical_cols + categorical_cols + binary_cols) - set(df.columns)
if missing_cols:
    print(f"Warning: The following columns are missing in the data: {missing_cols}")

# Create a DataFrame with dataset information
data_info = pd.DataFrame({
    'Attribute': df.columns,
    'Data Type': df.dtypes.astype(str).values,
    'Non-Null Count': df.notnull().sum().values,
    'Null Count': df.isnull().sum().values,
    'Unique Count': df.nunique(dropna=False).values
})

# Reorder the columns in data_info to match the desired output
data_info = data_info[['Attribute', 'Data Type', 'Non-Null Count', 'Null Count', 'Unique Count']]

# Prepare a directory to save plots
plots_dir = 'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# Generate plots for numerical variables
for col in existing_numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plot_path = os.path.join(plots_dir, f'{col}_histogram.png')
    plt.savefig(plot_path)
    plt.close()

# Generate plots for categorical variables
for col in existing_categorical_cols:
    plt.figure(figsize=(8, 6))
    counts = df[col].value_counts(dropna=False)
    counts.plot(kind='bar')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plot_path = os.path.join(plots_dir, f'{col}_bar_chart.png')
    plt.savefig(plot_path)
    plt.close()

# Generate plots for binary variables
for col in existing_binary_cols:
    plt.figure(figsize=(6, 6))
    counts = df[col].value_counts(dropna=False)
    counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title(f'Distribution of {col}')
    plt.ylabel('')  # Hide y-label
    plot_path = os.path.join(plots_dir, f'{col}_pie_chart.png')
    plt.savefig(plot_path)
    plt.close()

# Write to Excel with multiple sheets, using xlsxwriter as the engine
with pd.ExcelWriter('diabetes_data_summary.xlsx', engine='xlsxwriter') as writer:
    # Write dataset information
    data_info.to_excel(writer, sheet_name='Dataset Info', index=False)
    
    # For numerical attributes: compute descriptive statistics including mode
    if existing_numerical_cols:
        numerical_stats = df[existing_numerical_cols].describe().transpose()

        # Compute mode for numerical attributes
        numerical_modes = df[existing_numerical_cols].mode().transpose()
        # In case of multiple modes, combine them into a comma-separated string
        numerical_modes['Mode'] = numerical_modes.apply(lambda row: ', '.join(map(str, row.dropna())), axis=1)
        numerical_modes = numerical_modes[['Mode']]

        numerical_stats['Mode'] = numerical_modes['Mode']

        # Rename columns for clarity
        numerical_stats.rename(columns={
            'count': 'Count',
            'mean': 'Mean',
            'std': 'StdDev',
            'min': 'Min',
            '25%': '25%',
            '50%': 'Median',
            '75%': '75%',
            'max': 'Max'
        }, inplace=True)

        # Write numerical attributes statistics
        numerical_stats.to_excel(writer, sheet_name='Numerical Attributes')
    else:
        print("No numerical attributes to write.")
    
    # For categorical attributes: compute counts and modes
    categorical_counts_list = []
    categorical_modes_list = []

    for col in existing_categorical_cols:
        counts = df[col].value_counts(dropna=False).reset_index()
        counts.columns = ['Value', 'Count']
        counts['Attribute'] = col
        counts = counts[['Attribute', 'Value', 'Count']]
        categorical_counts_list.append(counts)

        # Compute mode for the categorical attribute
        value_counts = df[col].value_counts(dropna=True)
        if not value_counts.empty:
            max_count = value_counts.max()
            modes = value_counts[value_counts == max_count].index.tolist()
            mode_str = ', '.join(map(str, modes))
        else:
            mode_str = None

        categorical_modes_list.append({'Attribute': col, 'Mode': mode_str})

    # Concatenate counts into a single DataFrame
    if categorical_counts_list:
        categorical_counts = pd.concat(categorical_counts_list, ignore_index=True)
    else:
        categorical_counts = pd.DataFrame()

    # Create DataFrame of modes for categorical attributes
    categorical_modes = pd.DataFrame(categorical_modes_list)

    # Write categorical attributes
    if not categorical_modes.empty and not categorical_counts.empty:
        sheet_name = 'Categorical Attributes'
        # Write modes and counts to separate tables in the same sheet
        # Start from row 0
        row = 0
        # Write modes
        categorical_modes.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
        # Calculate the number of rows written
        modes_rows = len(categorical_modes) + 2  # Adding 2 for header and an empty row
        # Write counts
        categorical_counts.to_excel(writer, sheet_name=sheet_name, index=False, startrow=modes_rows)
    else:
        print("No categorical attributes to write.")

    # For binary attributes: compute counts and statistics
    binary_counts_list = []
    for col in existing_binary_cols:
        counts = df[col].value_counts(dropna=False).reset_index()
        counts.columns = ['Value', 'Count']
        counts['Attribute'] = col
        counts = counts[['Attribute', 'Value', 'Count']]
        binary_counts_list.append(counts)

    if binary_counts_list:
        binary_counts = pd.concat(binary_counts_list, ignore_index=True)
    else:
        binary_counts = pd.DataFrame()

    # Compute descriptive statistics for binary attributes
    if existing_binary_cols:
        binary_stats = df[existing_binary_cols].describe().transpose()

        # Compute mode for binary attributes
        binary_modes = df[existing_binary_cols].mode().transpose()
        # Handle multiple modes
        binary_modes['Mode'] = binary_modes.apply(lambda row: ', '.join(map(str, row.dropna())), axis=1)
        binary_modes = binary_modes[['Mode']]

        binary_stats['Mode'] = binary_modes['Mode']

        # Rename columns for clarity
        binary_stats.rename(columns={
            'count': 'Count',
            'mean': 'Mean',
            'std': 'StdDev',
            'min': 'Min',
            '25%': '25%',
            '50%': 'Median',
            '75%': '75%',
            'max': 'Max'
        }, inplace=True)

        # Write binary attributes
        sheet_name = 'Binary Attributes'
        row = 0
        # Write statistics
        binary_stats.to_excel(writer, sheet_name=sheet_name, startrow=row)
        row += len(binary_stats) + 2  # Leave two empty rows
        # Write counts
        binary_counts.to_excel(writer, sheet_name=sheet_name, index=False, startrow=row)
    else:
        print("No binary attributes to write.")
    
    # Compute the distribution of null counts per row
    # Calculate the number of nulls in each row
    null_counts_per_row = df.isnull().sum(axis=1)

    # Count the number of rows with each number of nulls
    null_counts_distribution = null_counts_per_row.value_counts().sort_index()

    # Convert the distribution to a DataFrame
    null_counts_df = null_counts_distribution.reset_index()
    null_counts_df.columns = ['Number of Nulls', 'Number of Rows']

    # Write the null counts distribution to a new sheet
    null_counts_df.to_excel(writer, sheet_name='Row Attribute Nulls', index=False)

    # Insert plots into a new sheet called 'Visualizations'
    workbook  = writer.book
    worksheet = workbook.add_worksheet('Visualizations')
    writer.sheets['Visualizations'] = worksheet

    # Set up formatting for images
    row = 0
    col = 0
    image_row_height = 20  # Approximate row height per image

    # Insert numerical variable plots
    for col_name in existing_numerical_cols:
        plot_path = os.path.join(plots_dir, f'{col_name}_histogram.png')
        if os.path.exists(plot_path):
            worksheet.write(row, 0, f'Histogram of {col_name}')
            worksheet.insert_image(row + 1, 0, plot_path, {'x_scale': 0.7, 'y_scale': 0.7})
            row += image_row_height
        else:
            print(f"Plot for {col_name} not found.")

    # Insert categorical variable plots
    for col_name in existing_categorical_cols:
        plot_path = os.path.join(plots_dir, f'{col_name}_bar_chart.png')
        if os.path.exists(plot_path):
            worksheet.write(row, 0, f'Bar Chart of {col_name}')
            worksheet.insert_image(row + 1, 0, plot_path, {'x_scale': 0.7, 'y_scale': 0.7})
            row += image_row_height
        else:
            print(f"Plot for {col_name} not found.")

    # Insert binary variable plots
    for col_name in existing_binary_cols:
        plot_path = os.path.join(plots_dir, f'{col_name}_pie_chart.png')
        if os.path.exists(plot_path):
            worksheet.write(row, 0, f'Pie Chart of {col_name}')
            worksheet.insert_image(row + 1, 0, plot_path, {'x_scale': 0.7, 'y_scale': 0.7})
            row += image_row_height
        else:
            print(f"Plot for {col_name} not found.")