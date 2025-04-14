import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. **Load the bike-sharing dataset**
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Columns in the dataset: {df.columns}")
    return df

# 2. **Handle missing values**
def handle_missing_values(df):
    print("Missing values before handling:")
    print(df.isnull().sum())

    # Handle missing values
    df['Duration'].fillna(df['Duration'].mean(), inplace=True)  # Use 'Duration' column
    df['Start station'].fillna('Unknown', inplace=True)
    df['Member type'].fillna('Unknown', inplace=True)
    
    # Fill missing values in other columns if any (you can customize this)
    df.fillna({'End station': 'Unknown'}, inplace=True)

    print("\nMissing values after handling:")
    print(df.isnull().sum())
    return df

# 3. **Parse Date-Time Columns**
def parse_date_time(df):
    # Assuming 'Start date' and 'End date' columns exist
    df['Start time'] = pd.to_datetime(df['Start date'])
    df['End time'] = pd.to_datetime(df['End date'])

    # Calculate trip duration (in minutes) if the column doesn't exist
    if 'Duration' not in df.columns:
        df['Duration'] = (df['End time'] - df['Start time']).dt.total_seconds() / 60.0

    # Extract additional date-related features
    df['Year'] = df['Start time'].dt.year
    df['Month'] = df['Start time'].dt.month
    df['Day'] = df['Start time'].dt.day
    df['Weekday'] = df['Start time'].dt.weekday  # Monday=0, Sunday=6

    return df

# 4. **Feature Engineering**
def feature_engineering(df):
    # Create trip duration bins (e.g., 0-15 min, 15-30 min, 30+ min)
    bins = [0, 15, 30, np.inf]
    labels = ['0-15 min', '15-30 min', '30+ min']
    df['Duration_bin'] = pd.cut(df['Duration'], bins=bins, labels=labels, right=False)

    # Create user age groups (assuming 'Birth Year' is available)
    # We'll need to assume a 'Birth Year' column or use another logic for age.
    # For example, we could extract the age groups from the 'Member type' column if applicable.
    df['Age'] = df['Start time'].dt.year - 1980  # Example, you should replace it with actual data
    bins_age = [0, 18, 25, 35, 45, 60, np.inf]
    labels_age = ['0-18', '19-25', '26-35', '36-45', '46-60', '60+']
    df['Age_group'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, right=False)

    # Optional: Create other features, e.g., 'Time_of_day' based on start time
    df['Time_of_day'] = pd.cut(df['Start time'].dt.hour, bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'], right=False)

    return df

# 5. **Plotting functions**
def plot_data(df):
    # Plot 1: Distribution of Trip Duration (Histogram)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Duration'], kde=True, bins=30, color='skyblue')
    plt.title('Distribution of Trip Duration')
    plt.xlabel('Trip Duration (Minutes)')
    plt.ylabel('Frequency')
    plt.show()

    # Plot 2: Distribution of Users by Age Group (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Age_group', data=df, palette='viridis')
    plt.title('Distribution of Users by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Users')
    plt.show()

    # Plot 3: Distribution of Trips by Time of Day (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Time_of_day', data=df, palette='coolwarm')
    plt.title('Distribution of Trips by Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Trips')
    plt.show()

    # Plot 4: Trip Duration Bins (Bar Plot)
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Duration_bin', data=df, palette='muted')
    plt.title('Distribution of Trip Duration Bins')
    plt.xlabel('Duration Bin')
    plt.ylabel('Number of Trips')
    plt.show()

# 6. **Main function to combine all preprocessing and plotting**
def preprocess_and_plot_bike_sharing_data(file_path):
    # Load data
    df = load_data(file_path)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Parse date-time columns
    df = parse_date_time(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Plot the data
    plot_data(df)

# Run the preprocessing and plotting
file_path = r'C:\Users\harsh\Desktop\DM\bike_share.csv'  # Updated file path
preprocess_and_plot_bike_sharing_data(file_path)
