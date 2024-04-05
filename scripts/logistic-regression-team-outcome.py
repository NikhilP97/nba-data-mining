"""
**********Not working currently***************
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
# Assuming your dataset is in a CSV file named 'nba_data.csv'
data = pd.read_csv('../dataset/binary-players-and-outcomes.csv')
print("Original data frame shape: {0}".format(data.shape))

data = data.dropna()
print("After dropping na: {0}".format(data.shape))

# Split data into features (player names) and target (outcome)
# y = data[['outcome']]  # Target
y = data['outcome']
X = data.drop('outcome', axis=1)  # Features

X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict the outcomes for the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the logistic regression model:", accuracy)

# Now, you can use this trained model to predict outcomes for new games
# For example, let's say you have a new game with the following players:
new_game_away_players = ['Andrew Bynum', 'Lamar Odom', 'Luke Walton', 'Sasha Vujacic', 'Smush Parker']
new_game_home_players = ['Boris Diaw', 'Kurt Thomas', 'Raja Bell', 'Shawn Marion', 'Marcus Banks']

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
new_game_data = pd.get_dummies(new_game_data)

# Make predictions for the new game
outcome_prediction = model.predict(new_game_data)
print("Predicted outcome for the new game:", outcome_prediction[0])
