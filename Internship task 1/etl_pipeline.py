import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Function to read CSV data from a given file path
def read_csv_data(path):
    print(f"Reading data from: {path}")
    try:
        # Attempt to read CSV file with UTF-8 encoding and remove leading spaces
        data = pd.read_csv(path, encoding='utf-8', skipinitialspace=True)
        return data
    except pd.errors.EmptyDataError:
        # Handle case where file is empty or contains no data
        print("Error: CSV file is empty or has no columns to parse.")
        return pd.DataFrame()
    except Exception as e:
        # Handle any unexpected errors
        print(f"An error occurred: {e}")
        return pd.DataFrame()

# Function to handle missing values in numeric columns using mean imputation
def handle_missing_values(data):
    imputer = SimpleImputer(strategy="mean")
    # Select only numeric columns (integers and floats)
    numeric_data = data.select_dtypes(include=["float64", "int64"])
    # Apply mean imputation to fill missing values
    imputed_data = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_data.columns)
    # Replace original numeric columns with imputed values
    data[numeric_data.columns] = imputed_data
    return data

# Function to scale numeric columns using standardization (z-score)
def scale_numeric_columns(data):
    scaler = StandardScaler()
    # Identify numeric columns to scale
    numeric_cols = data.select_dtypes(include=["float64", "int64"]).columns
    # Apply standard scaling (zero mean, unit variance)
    scaled_values = scaler.fit_transform(data[numeric_cols])
    # Update the original data with scaled values
    data[numeric_cols] = scaled_values
    return data

# Function to save the cleaned and processed data to a new CSV file
def save_clean_data(data, output_path):
    data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")

# Main ETL pipeline function: reads, processes, and saves the data
def run_etl(input_file, output_file):
    # Step 1: Read raw input data
    raw_data = read_csv_data(input_file)
    
    # Check if data was successfully read
    if raw_data.empty:
        print("ETL process aborted: No data found in input file.")
        return

    # Step 2: Handle missing values
    clean_data = handle_missing_values(raw_data)

    # Step 3: Scale numeric features
    processed_data = scale_numeric_columns(clean_data)

    # Step 4: Save the cleaned and transformed data
    save_clean_data(processed_data, output_file)

    # Step 5: Final status and preview
    print("ETL process completed successfully.")
    print("\nProcessed Data Preview:")
    print(processed_data.head())

# Define file paths for input and output
input_path = "raw_data.csv"
output_path = "clean_data.csv"

# Run the complete ETL pipeline
run_etl(input_path, output_path)
