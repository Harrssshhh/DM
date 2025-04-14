# Practical 3 - Apriori Algorithm with Charts

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load dataset
file_path = 'C:\\Users\\harsh\\Desktop\\DM\\Groceries_dataset.csv'
transactions = []
with open(file_path, 'r') as file:
    for line in file:
        transactions.append(line.strip().split(','))

# Convert transactions into a one-hot encoded DataFrame
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
one_hot_df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply Apriori Algorithm
min_support = 0.02
frequent_itemsets = apriori(one_hot_df, min_support=min_support, use_colnames=True)

# Calculate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display top 10 frequent itemsets
top_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).head(10)
print("Top 10 Frequent Itemsets:")
print(top_itemsets)

# Display top 10 rules sorted by lift
top_rules = rules.sort_values(by='lift', ascending=False).head(10)
print("\nTop 10 Association Rules (sorted by Lift):")
print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# ===================
# Visualization
# ===================

# Bar chart: Top 10 frequent itemsets
plt.figure(figsize=(10, 6))
item_labels = top_itemsets['itemsets'].apply(lambda x: ', '.join(list(x)))
plt.barh(item_labels, top_itemsets['support'], color='skyblue')
plt.xlabel('Support')
plt.title('Top 10 Frequent Itemsets')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Scatter plot: Top 10 association rules
plt.figure(figsize=(8, 6))
plt.scatter(top_rules['confidence'], top_rules['lift'], color='orange', edgecolors='black')
plt.xlabel('Confidence')
plt.ylabel('Lift')
plt.title('Top 10 Association Rules: Confidence vs Lift')
plt.grid(True)
plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the dataset
df = pd.read_csv('C:\\Users\\harsh\\Desktop\\DM\\BicycleCounter.csv')

# Drop rows and columns with missing values
df_cleaned = df.dropna()
df_cleaned_columns = df_cleaned.dropna(axis=1)

# Use actual bicycle count data for histogram
data = df_cleaned_columns['Fremont Bridge Sidewalks, south of N 34th St']

# Plotting histogram
plt.hist(data, bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Bicycle Count')
plt.ylabel('Frequency')
plt.title('Bicycle Crossing Histogram')
plt.show()
