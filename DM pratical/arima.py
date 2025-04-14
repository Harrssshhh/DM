# Install necessary versions (only needed in some notebook environments)
# !pip uninstall -y numpy pmdarima
# !pip install numpy==1.23.5
# !pip install pmdarima

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt
import pmdarima as pm

# Load COVID-19 dataset from GitHub
url = 'C:\\Users\\harsh\\Desktop\\DM\\countries-aggregated.csv'

df = pd.read_csv(url)

# Filter for India
df = df[df['Country'] == 'India']

# Convert 'Date' to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Use only the 'Confirmed' cases
df = df[['Confirmed']]

# Plot the original time series
plt.figure(figsize=(12, 6))
plt.plot(df, label='Confirmed Cases')
plt.title('India COVID-19 Confirmed Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.legend()
plt.show()

# ADF test function
def adf_test(series):
    result = adfuller(series)
    print("ADF Test Statistic:", result[0])
    print("p-value:", result[1])
    print("Critical Values:", result[4])
    if result[1] <= 0.05:
        print("✅ Data is stationary")
    else:
        print("❌ Data is NOT stationary")

# Perform ADF test on original data
print("ADF Test on Original Data:")
adf_test(df['Confirmed'])

# Differencing to make the data stationary
df_diff = df.diff().dropna()

print("\nADF Test on Differenced Data:")
adf_test(df_diff['Confirmed'])

# Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_acf(df_diff, ax=ax[0], lags=40)
plot_pacf(df_diff, ax=ax[1], lags=40)
plt.show()

# Use auto_arima to find optimal p, d, q
auto_arima_model = pm.auto_arima(df['Confirmed'], seasonal=False, trace=True, stepwise=True)
p, d, q = auto_arima_model.order
print(f"\n✅ Optimal ARIMA Order: p={p}, d={d}, q={q}")

# Train ARIMA model
model = ARIMA(df['Confirmed'], order=(p, d, q))
fitted_model = model.fit()

# Forecast next 30 days
forecast_steps = 30
forecast = fitted_model.forecast(steps=forecast_steps)

# Plot actual and forecasted values
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Confirmed'], label='Actual Confirmed Cases')
future_dates = pd.date_range(df.index[-1], periods=forecast_steps+1, freq='D')[1:]
plt.plot(future_dates, forecast, label='Forecast', linestyle='dashed', color='red')
plt.title('India COVID-19 Forecast using ARIMA')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.legend()
plt.show()

# Calculate RMSE (only if you have actual future data for comparison)
# For now, we'll just compare with the last 30 days of the dataset
actual = df['Confirmed'][-forecast_steps:].dropna()
predicted = forecast[:len(actual)]
rmse = sqrt(mean_squared_error(actual, predicted))
print(f"\n📊 Root Mean Squared Error (RMSE): {rmse:.2f}")
