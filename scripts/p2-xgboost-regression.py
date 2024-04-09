import pandas as pd
import json
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error


# Load your dataset
# Assuming your dataset is stored in a pandas DataFrame called 'data'
data = pd.read_csv("../dataset/v7-with-scores-only-players.csv")

# Extract features (player names) and target (score)
X = data.drop(columns=['score'])  # Features: player names
y = data['score']  # Target: score

# One-hot encode player names
# player_encoder = OneHotEncoder(handle_unknown='ignore')
# X_encoded = player_encoder.fit_transform(X)

# define categorical features
categorical_features = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 'away_0', 'away_1', 'away_2', 'away_3', 'away_4']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Debug logs
# print('Before setting type')
# print('X_train dtypes: ', X_train.dtypes)
# print('y_train dtypes: ', y_train.dtypes)
# print('\n')

# Set type as category for input columns
for cur_feature in categorical_features:
    X_train[cur_feature] = X_train[cur_feature].astype("category")

# Train an XGBoost regressor
regressor = xgb.XGBRegressor(enable_categorical=True)

# Debug logs
# print('After setting type')
# print('X_train dtypes: ', X_train.dtypes)
# print('y_train shape: ', y_train.shape)
# print('y_train head: ', y_train.head())
# print('y_train dtypes: ', y_train.dtypes)
print("Training the model")
print('\n')

regressor.fit(X_train, y_train)

# Update type of X_test
for cur_feature in categorical_features:
    X_test[cur_feature] = X_test[cur_feature].astype("category").cat.set_categories(X_train[cur_feature].cat.categories)

# Make predictions on the testing set
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("\n")

# Predict score for a new match
# Use this trained model to predict outcomes for new games
new_game_home_players = ['Andrew Bynum', 'Lamar Odom', 'Luke Walton', 'Sasha Vujacic'] # 'Smush Parker'
new_game_away_players = ['Boris Diaw', 'Kurt Thomas', 'Raja Bell', 'Shawn Marion', 'Steve Nash']
print("4 Home Players: ", new_game_home_players)
print("5 Away Players: ", new_game_away_players)
print("\n")
# new_game_home_players = ['Corey Maggette', 'Cuttino Mobley', 'Elton Brand', 'Shaun Livingston', 'Tim Thomas']
# new_game_away_players = ["Amar'e Stoudemire", 'Boris Diaw', 'James Jones', 'Leandro Barbosa', 'Marcus Banks']
# Other example
# new_game_away_players = ['A', 'B', 'C', 'D', 'E']
# new_game_home_players = ['F', 'G', 'H', 'I', 'J']

def read_team_map_from_json(json_file):
    # Read team map from JSON file into a dictionary
    with open(json_file, 'r') as f:
        team_map = json.load(f)
    return team_map

def get_players_for_team(team_map, team_name):
    # Get array of players for a given team name
    if team_name in team_map:
        return team_map[team_name]
    else:
        return None

# Example usage
input_json = "../dataset/team_player_map.json"  # Provide the path to your JSON file

# Read team map from JSON file
team_map = read_team_map_from_json(input_json)

# Get array of players for a given team name
team_name = "LAL"  # Provide the team name for which you want to get players
players = get_players_for_team(team_map, team_name)

best_player = ''
score_combination_map = {}
best_score_so_far = None

for cur_player in players:
    # Convert player names to categorical variables and create a DataFrame
    new_game_data = pd.DataFrame({
        'home_0': [new_game_home_players[0]],
        'home_1': [new_game_home_players[1]],
        'home_2': [new_game_home_players[2]],
        'home_3': [new_game_home_players[3]],
        'home_4': [cur_player],
        'away_0': [new_game_away_players[0]],
        'away_1': [new_game_away_players[1]],
        'away_2': [new_game_away_players[2]],
        'away_3': [new_game_away_players[3]],
        'away_4': [new_game_away_players[4]]
    })

    # Convert player names to categorical variables
    for cur_feature in categorical_features:
        new_game_data[cur_feature] = new_game_data[cur_feature].astype("category").cat.set_categories(X_train[cur_feature].cat.categories)
    predicted_score = regressor.predict(new_game_data)
    predicted_score = predicted_score[0]
    score_combination_map[cur_player] = predicted_score

    if best_score_so_far is not None and predicted_score > best_score_so_far:
        best_score_so_far = predicted_score
        best_player = cur_player

    if best_score_so_far is None:
        best_score_so_far = predicted_score

    
    # print("Predicted Score:", predicted_score)

# print("Score combination map:", score_combination_map)
sorted_items = sorted(score_combination_map.items(), key=lambda x: x[1], reverse=True)
print("Top 5 players are:")
for i, (key, value) in enumerate(sorted_items[:5], start=1):
        print(f"{i}. {key}: {value}")

print("\n")
print("Best player to choose in the lineup:", best_player)
print("\n\n\n\n\n")
