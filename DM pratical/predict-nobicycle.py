import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Reading csv
df = pd.read_csv("C:\\Users\\harsh\\Desktop\\DM\\fremont.csv")

# Check for null values
print("Null values in each column:")
print(df.isnull().sum())

# Handle missing values (example: fill with mean or drop rows)
# df.fillna(df.mean(), inplace=True)  # or df.dropna(inplace=True)

# Dataset Description
print("\nDataset Description:")
print(df.describe())

# Check for duplicated rows
print("\nNumber of duplicated rows:", df.duplicated().sum())

# Check for non-numeric columns
print("\nColumn data types:")
print(df.dtypes)

# Convert non-numeric columns to numeric
if 'Direction' in df.columns:
    print("\nUnique values in 'Direction' column:", df['Direction'].unique())
    # Encoding the 'Direction' column
    df['Direction'] = df['Direction'].astype(str).str.strip()  # Remove leading/trailing spaces
    direction_mapping = {'NB': 0, 'SB': 1, 'EB': 2, 'WB': 3}
    df['Direction'] = df['Direction'].map(direction_mapping)

# Check for any remaining non-numeric data
print("\nColumn data types after processing:")
print(df.dtypes)

# Splitting the data
x = df.drop(columns=['CyclistCount', 'Date','Time'])
y = df['CyclistCount']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling (optional)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model fitting
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# Making predictions
y_pred = model.predict(x_test_scaled)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R-Squared: {r2}")

# Regression coefficients
coefficients = pd.DataFrame(model.coef_, x.columns, columns=['Coefficients'])
print("\nFeature Coefficients:")
print(coefficients)

# Plotting True vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted')
plt.show()

# Plotting residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
plt.xlabel('Predictions')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.show()
