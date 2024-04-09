import pandas as pd

def calculate_difference(csv_file):
    # Read CSV file
    df = pd.read_csv(csv_file)
    df = df.dropna()
    
    # Calculate difference between 'home' and 'away' columns
    df["pts_home"] = pd.to_numeric(df["pts_home"], errors='coerce')
    df["pts_visitor"] = pd.to_numeric(df["pts_visitor"], errors='coerce')
    df['score'] = df['pts_home'] - df['pts_visitor']

    df = df.dropna()
    
    return df

def save_with_difference(df, output_csv):
    # Save DataFrame with 'difference' column to a new CSV file
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # Example usage
    input_csv = "../dataset/v1-matchups-original.csv"  # Provide the path to your input CSV file
    output_csv = "../dataset/v6-with-scores.csv"  # Provide the desired path for the output CSV file

    # Calculate difference and save to new CSV file
    df_with_difference = calculate_difference(input_csv)
    save_with_difference(df_with_difference, output_csv)
