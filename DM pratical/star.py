import pandas as pd
import sqlite3

# ================= STAR SCHEMA =================

# Dimension: Customer
df_customer = pd.DataFrame({
    'customer_id': [1, 2],
    'name': ['Alice', 'Bob'],
    'gender': ['F', 'M'],
    'phone': ['9876543210', '8765432190'],
    'email': ['alice@ola.com', 'bob@ola.com'],
    'city': ['Mumbai', 'Delhi']
})

# Dimension: Driver
df_driver = pd.DataFrame({
    'driver_id': [10, 11],
    'name': ['John', 'Mike'],
    'gender': ['M', 'M'],
    'phone': ['9988776655', '9123456780'],
    'license_no': ['MH01DR1234', 'DL05DR5678'],
    'experience_yrs': [5, 3]
})

# Dimension: Vehicle
df_vehicle = pd.DataFrame({
    'vehicle_id': [101, 102],
    'type': ['Sedan', 'Mini'],
    'model': ['Swift Dzire', 'WagonR'],
    'number_plate': ['MH01AB1234', 'DL05CD5678']
})

# Dimension: Date
df_date = pd.DataFrame({
    'date_id': [1, 2],
    'day': [12, 13],
    'month': ['April', 'April'],
    'year': [2025, 2025],
    'weekday': ['Saturday', 'Sunday']
})

# Dimension: Payment
df_payment = pd.DataFrame({
    'payment_id': [1, 2],
    'method': ['UPI', 'Credit Card'],
    'status': ['Success', 'Failed'],
    'promo_code': ['OLA50', None]
})

# Dimension: Support
df_support = pd.DataFrame({
    'support_id': [1, 2],
    'issue_type': ['App Crash', 'Fare Dispute'],
    'resolution_time_hrs': [2.5, 1.0],
    'resolved': ['Yes', 'Yes']
})

# Fact Table: Ride Transactions
df_fact_rides = pd.DataFrame({
    'ride_id': [1001, 1002],
    'customer_id': [1, 2],
    'driver_id': [10, 11],
    'vehicle_id': [101, 102],
    'date_id': [1, 2],
    'payment_id': [1, 2],
    'support_id': [1, 2],
    'distance_km': [12.5, 8.0],
    'fare': [250.0, 180.0],
    'rating': [5, 4]
})

# Connect to SQLite and load all tables (Star Schema)
conn = sqlite3.connect("ola_star_schema.db")

tables = {
    "Dim_Customer": df_customer,
    "Dim_Driver": df_driver,
    "Dim_Vehicle": df_vehicle,
    "Dim_Date": df_date,
    "Dim_Payment": df_payment,
    "Dim_Support": df_support,
    "Fact_Rides": df_fact_rides
}

for table_name, df in tables.items():
    df.to_sql(table_name, conn, if_exists="replace", index=False)

print("✅ Star schema loaded into SQLite: ola_star_schema.db")

conn.close()

# ================ SNOWFLAKE SCHEMA (with normalization) ================

# Normalizing city and separating from Customer
df_city = pd.DataFrame({
    'city_id': [1, 2],
    'city_name': ['Mumbai', 'Delhi'],
    'state': ['Maharashtra', 'Delhi'],
    'country': ['India', 'India']
})

df_customer_snow = df_customer.copy()
df_customer_snow['city_id'] = [1, 2]
df_customer_snow = df_customer_snow.drop(columns=['city'])

# Connect and load Snowflake schema
conn2 = sqlite3.connect("ola_snowflake_schema.db")

# Upload dimension tables (normalized)
df_city.to_sql("Dim_City", conn2, if_exists="replace", index=False)
df_customer_snow.to_sql("Dim_Customer", conn2, if_exists="replace", index=False)
df_driver.to_sql("Dim_Driver", conn2, if_exists="replace", index=False)
df_vehicle.to_sql("Dim_Vehicle", conn2, if_exists="replace", index=False)
df_date.to_sql("Dim_Date", conn2, if_exists="replace", index=False)
df_payment.to_sql("Dim_Payment", conn2, if_exists="replace", index=False)
df_support.to_sql("Dim_Support", conn2, if_exists="replace", index=False)
df_fact_rides.to_sql("Fact_Rides", conn2, if_exists="replace", index=False)

print("✅ Snowflake schema loaded into SQLite: ola_snowflake_schema.db")

conn2.close()

import matplotlib.pyplot as plt

# Visualize Top Rides by Fare
df_plot = df_fact_rides.sort_values(by="fare", ascending=False)

plt.figure(figsize=(8, 5))
plt.bar(df_plot['ride_id'].astype(str), df_plot['fare'], color='skyblue')
plt.title("Top Rides by Fare")
plt.xlabel("Ride ID")
plt.ylabel("Fare (INR)")
plt.grid(True, axis='y')
plt.show()
