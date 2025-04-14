import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# ✅ Load the dataset
df = pd.read_csv('C:\\Users\\harsh\\Downloads\\DM pratical\\house_price_regression_dataset.csv')  # ensure it's in your directory

# ✅ Features and Target
X = df.drop('House_Price', axis=1)
y = df['House_Price']

# ✅ Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Polynomial Features
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# ✅ Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=10000)
}

# ✅ Evaluation + Visualization
plt.figure(figsize=(15, 5))
results = []

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results.append((name, r2, rmse, mae))

    # Plot Actual vs Predicted
    plt.subplot(1, 3, i + 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'{name}\nR²={r2:.2f}, RMSE={rmse:.0f}')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")

plt.tight_layout()
plt.show()

# ✅ Residual Plot for Best Model (based on RMSE)
best_model = min(results, key=lambda x: x[2])[0]  # model with lowest RMSE
model = models[best_model]
y_pred_best = model.predict(X_test_poly)
residuals = y_test - y_pred_best

plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=30, kde=True, color='teal')
plt.title(f'Residual Distribution ({best_model})')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ Print Table of Results
results_df = pd.DataFrame(results, columns=["Model", "R²", "RMSE", "MAE"])
print("\n🔍 Model Performance Comparison:\n")
print(results_df)
