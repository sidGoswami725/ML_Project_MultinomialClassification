import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def perform_friend_eda(train_file="train.csv"):
    print(f"--- Starting Comprehensive EDA on {train_file} ---")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(train_file)
        print(f"Loaded {train_file}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {train_file} not found.")
        return

    # --- 2. Quick Clean for Visualization ---
    # We drop rows where target is missing just for plotting
    if 'spend_category' in df.columns:
        df = df.dropna(subset=['spend_category'])
        df['spend_category'] = df['spend_category'].astype(int)
    else:
        print("Error: Target 'spend_category' not found.")
        return

    # Set visualization style
    sns.set_style("whitegrid")
    
    # ==========================================================================
    # FIGURE 1: Target Variable Distribution
    # ==========================================================================
    print("Generating Target Distribution...")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='spend_category', data=df, palette='viridis')
    plt.title('Distribution of Spend Category (0=Low, 1=Med, 2=High)')
    plt.xlabel('Spend Category')
    plt.ylabel('Count')
    
    # Add percentage labels
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig('eda_1_target_distribution.png')
    print("Saved 'eda_1_target_distribution.png'")

    # ==========================================================================
    # FIGURE 2: Numerical Feature Histograms
    # ==========================================================================
    # Analyzing group sizes and stay duration
    num_cols = ['num_females', 'num_males', 'mainland_stay_nights', 'island_stay_nights']
    # Filter to exist columns
    num_cols = [c for c in num_cols if c in df.columns]

    print("Generating Numerical Histograms...")
    if num_cols:
        df[num_cols].hist(bins=20, figsize=(12, 8), layout=(2, 2), color='#3498db', edgecolor='black')
        plt.suptitle('Histograms of Numerical Features')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('eda_2_numerical_histograms.png')
        print("Saved 'eda_2_numerical_histograms.png'")

    # ==========================================================================
    # FIGURE 3: Categorical Distributions (Top Categories)
    # ==========================================================================
    print("Generating Categorical Bar Charts...")
    
    cat_cols = ['country', 'tour_type', 'main_activity', 'visit_purpose', 'info_source']
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        if col in df.columns:
            # Get top 10 categories to avoid clutter (especially for Country)
            top_cats = df[col].value_counts().nlargest(10).index
            data_filtered = df[df[col].isin(top_cats)]
            
            sns.countplot(y=col, data=data_filtered, order=top_cats, ax=axes[i], palette='mako')
            axes[i].set_title(f'Top 10 {col}')
            axes[i].set_xlabel('Count')
        else:
            axes[i].axis('off')

    # Hide unused subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig('eda_3_categorical_distributions.png')
    print("Saved 'eda_3_categorical_distributions.png'")

    # ==========================================================================
    # FIGURE 4: Correlation Heatmap
    # ==========================================================================
    print("Generating Correlation Heatmap...")
    
    # We need to temporarily encode "Yes"/"No" columns to see their correlation
    corr_df = df.copy()
    
    # 1. Map Yes/No
    binary_map = {'Yes': 1, 'No': 0}
    cols_to_map = [c for c in corr_df.columns if corr_df[c].dtype == 'object']
    for col in cols_to_map:
        # Check if column primarily contains Yes/No
        unique_vals = set(corr_df[col].dropna().unique())
        if unique_vals.issubset({'Yes', 'No', 'yes', 'no'}):
            corr_df[col] = corr_df[col].map(binary_map)

    # 2. Select only numeric columns for heatmap
    numeric_corr_df = corr_df.select_dtypes(include=['int64', 'float64'])
    
    if not numeric_corr_df.empty:
        plt.figure(figsize=(12, 10))
        corr = numeric_corr_df.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('eda_4_correlation_heatmap.png')
        print("Saved 'eda_4_correlation_heatmap.png'")

    # ==========================================================================
    # FIGURE 5: Spend Category vs. Key Features (Boxplots & Stacked Bars)
    # ==========================================================================
    print("Generating Spend Analysis Plots...")
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Plot A: Total People vs Spend (Do bigger groups spend more?)
    if 'num_females' in df.columns and 'num_males' in df.columns:
        df['total_people'] = df['num_females'] + df['num_males']
        # Filter outliers for cleaner plot
        plot_data = df[df['total_people'] < 10] 
        sns.boxplot(x='spend_category', y='total_people', data=plot_data, ax=ax[0], palette='Set2')
        ax[0].set_title('Group Size vs. Spend Category')
        ax[0].set_ylabel('Total People')

    # Plot B: Tour Type vs Spend (Stacked Bar)
    if 'tour_type' in df.columns:
        # Calculate percentages
        ct = pd.crosstab(df['tour_type'], df['spend_category'], normalize='index')
        ct.plot(kind='bar', stacked=True, ax=ax[1], colormap='viridis')
        ax[1].set_title('Tour Type vs. Spend Category Proportion')
        ax[1].set_ylabel('Proportion')
        ax[1].legend(title='Spend Cat')

    plt.tight_layout()
    plt.savefig('eda_5_spend_analysis.png')
    print("Saved 'eda_5_spend_analysis.png'")

    print("\n--- EDA Complete! Check your files. ---")

if __name__ == "__main__":
    perform_friend_eda()