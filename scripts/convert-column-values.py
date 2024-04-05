import pandas as pd

def convert_column_values(input_file, output_file, column_name):
    """
    Convert values in a given column from -1 to 1 and save to a new CSV file.
    
    Parameters:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the modified CSV file.
        column_name (str): Name of the column to convert values.
    """
    # Read the CSV file
    data = pd.read_csv(input_file)
    
    # Convert values in the specified column from -1 to 0
    # data[column_name] = data[column_name].replace(-1, 0)
    data[column_name] = data[column_name].map({'-1': 0, '1': 1})
    
    # Save the modified data to a new CSV file
    data.to_csv(output_file, index=False)

# Example usage
if __name__ == "__main__":
    input_file = '../dataset/without-na-players-and-outcome.csv'  # Specify the input CSV file
    output_file = '../dataset/binary-players-and-outcomes.csv'  # Specify the output CSV file
    column_name = 'outcome'  # Specify the column name to convert values
    
    # Call the function
    convert_column_values(input_file, output_file, column_name)
