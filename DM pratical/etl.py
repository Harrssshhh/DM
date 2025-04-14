# ✅ Step 1: Create Sample CSV Files
trip_data = """trip_id,customer_id,driver_id,date,distance_km,fare
T001,C001,D001,2024-03-01,12.5,250
T002,C002,D002,2024-03-02,8.2,180
T003,C003,D001,2024-03-03,5.0,120
T004,C004,D003,2024-03-04,15.3,"""

customer_data = """customer_id,name,phone,email
C001,Alice,9876543210,alice@example.com
C002,Bob,9988776655,BOB@MAIL.COM
C003,Charlie,9876123456,charlie@domain.com
C004,David,8765432190,david@email.net"""

transactions_data = """transaction_id,trip_id,payment_mode,status
TX001,T001,UPI,Success
TX002,T002,Credit Card,Failed
TX003,T003,Cash,Success
TX004,T004,Wallet,Success"""

with open("trip_data.csv", "w") as f:
    f.write(trip_data)

with open("customer_data.csv", "w") as f:
    f.write(customer_data)

with open("transactions.csv", "w") as f:
    f.write(transactions_data)

print("✅ Sample CSV files created.")

# ✅ Step 2: Import Libraries
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")

# ✅ Step 3: Extract — Load CSV files into DataFrames
trip_df = pd.read_csv("trip_data.csv")
customer_df = pd.read_csv("customer_data.csv")
transaction_df = pd.read_csv("transactions.csv")

print("✅ CSV files loaded.")

# ✅ Step 4: Transform — Clean and Format Data
trip_df['fare'] = trip_df['fare'].fillna(0)
trip_df['date'] = pd.to_datetime(trip_df['date'], errors='coerce')
customer_df['email'] = customer_df['email'].str.lower()
transaction_df = transaction_df[transaction_df['status'] == 'Success']

print("✅ Data cleaned and transformed.")

# ✅ Step 5: Load — Save to SQLite database
conn = sqlite3.connect("ride_hailing.db")
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS Trips (
    trip_id TEXT PRIMARY KEY,
    customer_id TEXT,
    driver_id TEXT,
    date TEXT,
    distance_km REAL,
    fare REAL
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS Customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT,
    phone TEXT,
    email TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS Transactions (
    transaction_id TEXT PRIMARY KEY,
    trip_id TEXT,
    payment_mode TEXT,
    status TEXT
)''')

trip_df.to_sql("Trips", conn, if_exists="replace", index=False)
customer_df.to_sql("Customers", conn, if_exists="replace", index=False)
transaction_df.to_sql("Transactions", conn, if_exists="replace", index=False)

conn.commit()
conn.close()

print("✅ ETL completed. Data loaded into 'ride_hailing.db'.")

# ✅ Step 6: Visualize — Generate Graphs

# 🔹 1. Total Fare Collected per Driver
plt.figure(figsize=(6, 4))
fare_per_driver = trip_df.groupby('driver_id')['fare'].sum().reset_index()
sns.barplot(x='driver_id', y='fare', data=fare_per_driver, palette='Blues_d')
plt.title("Total Fare Collected per Driver")
plt.xlabel("Driver ID")
plt.ylabel("Total Fare")
plt.tight_layout()
plt.show()

# 🔹 2. Successful Transactions by Payment Mode
plt.figure(figsize=(6, 4))
success_payment_modes = transaction_df['payment_mode'].value_counts().reset_index()
success_payment_modes.columns = ['payment_mode', 'count']
sns.barplot(x='payment_mode', y='count', data=success_payment_modes, palette='Greens_d')
plt.title("Successful Transactions by Payment Mode")
plt.xlabel("Payment Mode")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.show()

# 🔹 3. Number of Trips per Day
plt.figure(figsize=(6, 4))
trip_counts = trip_df['date'].value_counts().sort_index()
trip_counts.plot(kind='line', marker='o', color='purple')
plt.title("Number of Trips per Date")
plt.xlabel("Date")
plt.ylabel("Trip Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
