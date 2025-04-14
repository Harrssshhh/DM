# ✅ Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Step 2: Load Dataset
df = pd.read_csv('C:\\Users\\harsh\\Downloads\\DM pratical\hour.csv')  # Ensure this file is in your working directory

# ✅ Step 3: Create Binary Target Variable
df['user_class'] = (df['registered'] > df['casual']).astype(int)

# ✅ Step 4: Select Features and Target
features = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
            'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
X = df[features]
y = df['user_class']

# ✅ Step 5: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 6: Initialize and Train Model
model = GaussianNB()
model.fit(X_train, y_train)

# ✅ Step 7: Predict and Evaluate
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("🔹 Accuracy:", acc)
print("🔹 Precision:", prec)
print("🔹 Recall:", rec)
print("🔹 F1 Score:", f1)

# ✅ Step 8: Cross-Validation
cv_f1 = cross_val_score(model, X, y, cv=5, scoring='f1')
print("🔸 Cross-validated F1 scores:", cv_f1)
print("🔸 Mean F1 Score (CV):", cv_f1.mean())

# ✅ Step 9: Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# ✅ Step 10: Plot F1 Score Distribution from Cross-Validation
plt.figure(figsize=(6, 4))
sns.boxplot(data=cv_f1, color='skyblue')
sns.stripplot(data=cv_f1, color='blue', jitter=0.1)
plt.title("Cross-Validation F1 Scores (5-Fold)")
plt.ylabel("F1 Score")
plt.xlabel("Model: GaussianNB")
plt.grid(True)
plt.tight_layout()
plt.show()
