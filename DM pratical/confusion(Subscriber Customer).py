# Cell 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import graphviz
import os

# Cell 2: Load the Dataset
df = pd.read_csv('C:\\Users\\harsh\\Desktop\\DM\\202004-divvy-tripdata.csv')
print("Dataset Columns:", df.columns)
df.head()

# Cell 3: Data Cleaning
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Convert datetime columns
df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])

# Cell 4: Feature Engineering
df['Trip Duration'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
df['Hour'] = df['started_at'].dt.hour
df['DayOfWeek'] = df['started_at'].dt.dayofweek
df['Month'] = df['started_at'].dt.month

# Map months to seasons
df['Season'] = df['Month'].map({
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall', 11: 'Fall'
})

# Drop unnecessary columns
df.drop(columns=['ride_id', 'started_at', 'ended_at'], inplace=True, errors='ignore')

# Cell 5: Encode Categorical Variables
encoder = LabelEncoder()
categorical_columns = ['start_station_name', 'end_station_name', 'rideable_type', 'Season', 'member_casual']
for col in categorical_columns:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col].astype(str))

# Cell 6: Split Data
X = df.drop(columns=['member_casual'])
y = df['member_casual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cell 7: Train Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Cell 8: Evaluate Base Model
y_pred = decision_tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Cell 9: Visualize the Decision Tree
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # Adjust this path for your system
dot_data = export_graphviz(
    decision_tree,
    out_file=None,
    feature_names=X.columns,
    class_names=['Subscriber', 'Customer'],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("decision_tree")
graph.view()

# Cell 10: Hyperparameter Tuning
decision_tree_tuned = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    criterion='gini',
    random_state=42
)
decision_tree_tuned.fit(X_train, y_train)
y_pred_tuned = decision_tree_tuned.predict(X_test)

print("Tuned Accuracy:", accuracy_score(y_test, y_pred_tuned))
print("Tuned Classification Report:\n", classification_report(y_test, y_pred_tuned))

# Cell 11: Predict on New Sample
sample_input = np.array([X_test.iloc[0].values])
y_new_pred = decision_tree_tuned.predict(sample_input)
print("Predicted User Class:", 'Subscriber' if y_new_pred[0] == 0 else 'Customer')
