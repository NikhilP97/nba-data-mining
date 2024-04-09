import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from hyperparameter_tuning import train_model, hps_params, cv_params


# Load the dataset
data = pd.read_csv('../dataset/matchups-2008.csv')
print("Original data frame shape: {0}".format(data.shape))

# drop null value rows
data = data.dropna()
print("After dropping na: {0}".format(data.shape))

# keep only required columns
required_features = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 'away_0', 'away_1', 'away_2', 'away_3', 'away_4', 'outcome']
data = data[required_features]


# Split data into features (player names) and target (outcome)
y = data['outcome']
X = data.drop('outcome', axis=1)  # Features


# define categorical features
categorical_features = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4', 'away_0', 'away_1', 'away_2', 'away_3', 'away_4']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # random_state=42


# Debug logs
print('Before setting type')
print('X_train dtypes: ', X_train.dtypes)
print('y_train dtypes: ', y_train.dtypes)
print('\n')


# Set type as category for input columns
for cur_feature in categorical_features:
   X_train[cur_feature] = X_train[cur_feature].astype("category")


# Update type of X_test
for cur_feature in categorical_features:
   X_test[cur_feature] = X_test[cur_feature].astype(
       "category").cat.set_categories(X_train[cur_feature].cat.categories)


# Debug logs
print('After setting type')
print('X_train dtypes: ', X_train.dtypes)
print('y_train shape: ', y_train.shape)
print('y_train head: ', y_train.head())
print('y_train dtypes: ', y_train.dtypes)
print('\n')


model = train_model(
   X_train,
   y_train,
   (X_test, y_test),
   lgb.LGBMClassifier(random_state=45),
   params=hps_params,
   cv_params=cv_params
)


# Predict the outcomes for the test set
y_pred = model.predict(X_test)


# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the LBGM model: \n", accuracy)


# Uuse this trained model to predict outcomes for new games
new_game_away_players = ['Andrew Bynum', 'Lamar Odom', 'Luke Walton', 'Sasha Vujacic', 'Smush Parker']
new_game_home_players = ['Boris Diaw', 'Kurt Thomas', 'Raja Bell', 'Shawn Marion', 'Marcus Banks']


# Other example
# new_game_away_players = ['A', 'B', 'C', 'D', 'E']
# new_game_home_players = ['F', 'G', 'H', 'I', 'J']


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


# Make predictions for the new game
outcome_prediction = model.predict(new_game_data)
print("Predicted outcome for the new game:", outcome_prediction[0])
