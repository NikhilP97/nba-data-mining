import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

# Load your dataset
# Assuming your dataset is stored in a pandas DataFrame called 'data'
data = pd.read_csv("../dataset/v7-with-scores-only-players.csv")

# Extract features (player names) and target (score)
X = data.drop(columns=['score'])  # Features: player names
y = data['score']  # Target: score

# define categorical features
categorical_features = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 'away_0', 'away_1', 'away_2', 'away_3', 'away_4']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set type as category for input columns
for cur_feature in categorical_features:
    X_train[cur_feature] = X_train[cur_feature].astype("category")

# Train an XGBoost regressor
print("Training the model")
print('\n')
regressor = xgb.XGBRegressor(enable_categorical=True)

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
print("Root Mean Squared Error: ", rmse)
print("\n")

# Predict score for a new match
# Use this trained model to predict outcomes for new games
# Test set 1
new_game_home_players = ['Andrew Bynum', 'Lamar Odom', 'Luke Walton', 'Sasha Vujacic', 'Smush Parker']
new_game_away_players = ['Boris Diaw', 'Kurt Thomas', 'Raja Bell', 'Shawn Marion', 'Steve Nash']

# Convert player names to categorical variables and create a DataFrame
new_game_data = pd.DataFrame({
    'home_0': [new_game_home_players[0]],
    'home_1': [new_game_home_players[1]],
    'home_2': [new_game_home_players[2]],
    'home_3': [new_game_home_players[3]],
    'home_4': [new_game_home_players[4]],
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
print("Home Players: ", new_game_home_players)
print("Away Players: ", new_game_away_players)
print("Predicted Score: ", predicted_score)
if predicted_score > 0:
    print("Home team wins!!!")
else:
    print("Away team wins!!!")    
print("\n")

# Test set 2
new_game_home_players = ['Corey Maggette', 'Cuttino Mobley', 'Elton Brand', 'Shaun Livingston', 'Tim Thomas']
new_game_away_players= ["Amar'e Stoudemire", 'Boris Diaw', 'James Jones', 'Leandro Barbosa', 'Marcus Banks']

# Convert player names to categorical variables and create a DataFrame
new_game_data = pd.DataFrame({
    'home_0': [new_game_home_players[0]],
    'home_1': [new_game_home_players[1]],
    'home_2': [new_game_home_players[2]],
    'home_3': [new_game_home_players[3]],
    'home_4': [new_game_home_players[4]],
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
print("Home Players: ", new_game_home_players)
print("Away Players: ", new_game_away_players)
print("Predicted Score:", predicted_score)
if predicted_score > 0:
    print("Home team wins!!!")
else:
    print("Away team wins!!!")    
print("\n")
