import pandas as pd
import json

def construct_team_map(csv_file):
    team_map = {}

    # Read CSV file
    df = pd.read_csv(csv_file)

    # Iterate over rows in the DataFrame
    for index, row in df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']

        # Extract home team players
        home_players = [row[f'home_{i}'] for i in range(5)]
        
        # Extract away team players
        away_players = [row[f'away_{i}'] for i in range(5)]
        
        # Initialize home team players list if not already present
        if home_team not in team_map:
            team_map[home_team] = []
        
        # Add unique home team players to the team map
        for player in home_players:
            if player not in team_map[home_team]:
                team_map[home_team].append(player)
        
        # Initialize away team players list if not already present
        if away_team not in team_map:
            team_map[away_team] = []
        
        # Add unique away team players to the team map
        for player in away_players:
            if player not in team_map[away_team]:
                team_map[away_team].append(player)

    return team_map

def save_team_map_as_json(team_map, output_json):
    # Save team map as JSON file
    with open(output_json, 'w') as json_file:
        json.dump(team_map, json_file, indent=4)

if __name__ == "__main__":
    # Example usage
    input_csv = "../dataset/v1-matchups-original.csv"  # Provide the path to your input CSV file
    output_json = "team_player_map.json"  # Provide the desired path for the output JSON file

    # Construct team map and save to JSON file
    team_map = construct_team_map(input_csv)
    save_team_map_as_json(team_map, output_json)
