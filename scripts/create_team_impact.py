import pandas as pd

def calculate_difference(csv_file):
    # Read CSV file
    df = pd.read_csv(csv_file)
    df = df.dropna()
    
    # Calculate difference between 'home' and 'away' columns
    df["fga_home"] = pd.to_numeric(df["fga_home"], errors='coerce')
    df["fta_home"] = pd.to_numeric(df["fta_home"], errors='coerce')
    df["ast_home"] = pd.to_numeric(df["ast_home"], errors='coerce')
    df["blk_home"] = pd.to_numeric(df["blk_home"], errors='coerce')

    df["fga_visitor"] = pd.to_numeric(df["fga_visitor"], errors='coerce')
    df["fta_visitor"] = pd.to_numeric(df["fta_visitor"], errors='coerce')
    df["ast_visitor"] = pd.to_numeric(df["ast_visitor"], errors='coerce')
    df["blk_visitor"] = pd.to_numeric(df["blk_visitor"], errors='coerce')


    df["home_team_impact"] = df["fga_home"] + df["fta_home"] + df["ast_home"] + df["blk_home"] / 4
    df["away_team_impact"] = df["fga_visitor"] + df["fta_visitor"] + df["ast_visitor"] + df["blk_visitor"] / 4

    df = df.dropna()
    
    return df

def save_with_difference(df, output_csv):
    # Save DataFrame with 'difference' column to a new CSV file
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # Example usage
    input_csv = "../dataset/v1-matchups-original.csv"  # Provide the path to your input CSV file
    output_csv = "../dataset/v8-team-impact.csv"  # Provide the desired path for the output CSV file

    # Calculate difference and save to new CSV file
    df_with_difference = calculate_difference(input_csv)
    columns_to_keep = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 'away_0', 'away_1', 'away_2', 'away_3', 'away_4', 'home_team_impact', 'away_team_impact']  # Specify the columns to keep
    df_with_difference = df_with_difference[columns_to_keep]
    save_with_difference(df_with_difference, output_csv)
