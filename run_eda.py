import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- File Paths ---
TRAIN_CLEANED_02 = "train_cleaned_02.csv"
OUTPUT_PLOT = 'spend_category_distribution.png'

# Load the fully cleaned training data
train_df = pd.read_csv(TRAIN_CLEANED_02)

# Set up the visualization style
sns.set_style("whitegrid")
plt.figure(figsize=(8, 5))

# Calculate the distribution of the target variable
target_counts = train_df['spend_category'].value_counts(normalize=True).sort_index() * 100

# Create the bar plot
ax = sns.barplot(x=target_counts.index, y=target_counts.values, hue=target_counts.index, palette="viridis", legend=False)

# Add labels and title
plt.title('Distribution of Target Variable (Spend Category)', fontsize=14)
plt.xlabel('Spend Category (0: Low, 1: Medium, 2: High)', fontsize=12)
plt.ylabel('Percentage of Trips (%)', fontsize=12)

# Add value labels on top of the bars
for i in ax.containers:
    ax.bar_label(i, fmt='%.2f%%')

plt.ylim(0, 60)
plt.tight_layout()

# Save the plot
plt.savefig(OUTPUT_PLOT)
plt.close()

print("eda_01_target_imbalance.py executed successfully.")
print("Target Distribution:")
print(target_counts)