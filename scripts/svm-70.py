import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

# Load the dataset
dataset_path = "../dataset/matchups-2008.csv"
df = pd.read_csv(dataset_path)
df = df.dropna()

# Drop unnecessary columns
# columns_to_drop = ['game', 'season', 'home_team', 'away_team', 'starting_min', 'end_min']
# df.drop(columns=columns_to_drop, inplace=True)

# Separate features and target variable
X = df.drop(columns=['outcome'])
y = df['outcome']

# Encode categorical variables
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Predict on the test set
preds = svm_model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, preds)
recall = recall_score(y_test, preds, average='weighted')
f1 = f1_score(y_test, preds, average='weighted')
auc = roc_auc_score(y_test, preds)

print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)
