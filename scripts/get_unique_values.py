import pandas as pd

def print_unique_values_and_types(input_file):
    """
    Read an input CSV file, convert it to a Pandas DataFrame,
    and print unique values and data types for each column.
    
    Parameters:
        input_file (str): Path to the input CSV file.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file, header=0)
    df = df.dropna(how='any',axis=0)
    df = df.astype({'outcome': 'int64'}, errors='ignore').dtypes
    # print(f"outcome: {df['outcome'].unique()}")
    print(f"df.dtypes: {df.dtypes}")
    # df['outcome'] = df['outcome'].astype('Int64')
    
    # Iterate through each column in the DataFrame
    for column in df['outcome']:
        unique_values = df[column].unique()  # Get unique values
        data_type = df[column].dtype  # Get data type
        print(f"Column: {column}")
        print(f"Data Type: {data_type}")
        # print("Unique Values:")
        # for value in unique_values:
        #     print(value)
        print("-------------------")

# Example usage
if __name__ == "__main__":
    input_file = '../dataset/original-only-players-and-outcome.csv'  # Specify the input CSV file
    print_unique_values_and_types(input_file)