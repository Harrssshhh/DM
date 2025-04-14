import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset from local CSV file
file_path = "C:/Users/harsh/Desktop/DM/boston.csv"  # Update path as needed
df = pd.read_csv(file_path)

# Check if target column is named 'MEDV'; if not, rename it
if "MEDV" not in df.columns:
    df.rename(columns={df.columns[-1]: "MEDV"}, inplace=True)

# EDA
print("Dataset Overview:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Scatter plots
sns.scatterplot(x="RM", y="MEDV", data=df)
plt.title("Number of Rooms vs Median Value")
plt.show()

sns.scatterplot(x="LSTAT", y="MEDV", data=df)
plt.title("Lower Status Population vs Median Value")
plt.show()

# Prepare features and target
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
