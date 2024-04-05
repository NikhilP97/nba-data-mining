import pandas as pd

def keep_columns(input_file, output_file, columns_to_keep):
    """
    Keep only specified columns in the CSV file and save to a new file.
    
    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the modified CSV file.
        columns_to_keep (list): List of column names to keep.
    """
    # Read the CSV file
    data = pd.read_csv(input_file)
    
    # Filter columns
    data = data[columns_to_keep]

    data = data.dropna()
    
    # Save the modified data to a new CSV file
    data.to_csv(output_file, index=False)

# Example usage
if __name__ == "__main__":
    input_file = '../dataset/original-only-players-and-outcome.csv'  # Specify the input CSV file
    output_file = '../dataset/without-na-players-and-outcome.csv'  # Specify the output CSV file
    columns_to_keep = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 'away_0', 'away_1', 'away_2', 'away_3', 'away_4', 'outcome']  # Specify the columns to keep
    
    # Call the function
    keep_columns(input_file, output_file, columns_to_keep)
