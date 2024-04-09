import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
file_path = "../dataset/matchups-2007.csv"
data = pd.read_csv(file_path)
data = data.dropna()

# Encode categorical variables
data_encoded = pd.get_dummies(data)

# Split data into features and target variable
X = data_encoded.drop('outcome', axis=1)
y = data_encoded['outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier with reduced complexity
rf_classifier_reduced = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf_classifier_reduced.fit(X_train, y_train)

# Test the model
y_pred_reduced = rf_classifier_reduced.predict(X_test)

# Evaluate accuracy
accuracy_reduced = accuracy_score(y_test, y_pred_reduced)
print("Random Forest Classifier Accuracy :", accuracy_reduced)

# Define function to predict match outcome
def predict_match_outcome(home_players, away_players):
    # Encode player names
    home_encoded = [1 if player in home_players else 0 for player in X.columns]
    away_encoded = [1 if player in away_players else 0 for player in X.columns]
    
    # Create DataFrame for home and away team features
    home_team_features = pd.DataFrame([home_encoded], columns=X.columns)
    away_team_features = pd.DataFrame([away_encoded], columns=X.columns)
    
    # Predict probabilities of home and away teams winning
    home_team_score = rf_classifier_reduced.predict_proba(home_team_features)[0][1]
    away_team_score = rf_classifier_reduced.predict_proba(away_team_features)[0][0]
    
    # Determine the outcome based on the scores
    if home_team_score > away_team_score:
        return "Home team wins"
    else:
        return "Away team wins"

# Example usage
home_players = ["Andrew Bynum", "Amar'e Stoudemire", "Kurt Thomas", "Boris Diaw", "James Jones"]
away_players = ["Chris Kaman", "Cuttino Mobley", "Amar'e Stoudemire", "Chucky Atkins", "Corliss Williamson"]
outcome = predict_match_outcome(home_players, away_players)
print("Match Outcome:", outcome)
