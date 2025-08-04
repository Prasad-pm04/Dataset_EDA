"""
Airlines Flight Data EDA (Exploratory Data Analysis)
Author: Data Analyst
Description: Comprehensive EDA script for airlines flight data including 
data cleaning, feature engineering, and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """Main function to execute the EDA pipeline"""
    
    print("Starting Airlines Flight Data EDA...")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('airlines_flights_data.csv')
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # Drop redundant column
    print("Cleaning data...")
    df.drop(columns=['index'], inplace=True)
    
    # Convert 'stops' to numeric
    df['stops'] = df['stops'].map({'zero': 0, 'one': 1, 'two_or_more': 2})
    
    # Drop duplicates
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Feature engineering
    print("Creating new features...")
    df['price_per_hour'] = df['price'] / df['duration']
    df['route'] = df['source_city'] + " to " + df['destination_city']
    
    # Display basic info
    print("\nDataset Info:")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Univariate analysis - numerical
    print("\nGenerating numerical features histograms...")
    df[['duration', 'days_left', 'price', 'price_per_hour']].hist(
        bins=30, figsize=(12, 8), color='skyblue', edgecolor='black')
    plt.suptitle("Histograms of Numeric Features")
    plt.tight_layout()
    plt.show()
    
    # Univariate analysis - categorical
    print("Generating categorical features distribution plots...")
    categorical_cols = ['airline', 'source_city', 'destination_city',
                        'departure_time', 'arrival_time', 'class']
    
    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        # Fixed: Assign y variable to hue and set legend=False
        sns.countplot(y=df[col], order=df[col].value_counts().index, 
                      hue=df[col], palette='viridis', legend=False)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()
    
    # Correlation heatmap
    print("Generating correlation matrix...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[['duration', 'days_left', 'price', 'price_per_hour']].corr(),
                annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
    # Boxplots for price analysis
    print("Generating price analysis boxplots...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='airline', y='price')
    plt.xticks(rotation=45, ha='right')
    plt.title("Flight Price by Airline")
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='class', y='price')
    plt.title("Flight Price by Class")
    plt.tight_layout()
    plt.show()
    
    # Outlier detection
    print("Generating outlier detection plots...")
    numeric_cols = ['duration', 'days_left', 'price', 'price_per_hour']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], color='orange', ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}")
    
    plt.tight_layout()
    plt.show()
    
    # Save cleaned dataset
    print("Saving cleaned dataset...")
    df.to_csv('cleaned_airlines_data.csv', index=False)
    print("Cleaned dataset saved as 'cleaned_airlines_data.csv'")
    
    print("\n" + "=" * 50)
    print("EDA completed successfully!")
    print(f"Final dataset shape: {df.shape}")
    print("All visualizations have been generated.")

if __name__ == "__main__":
    main()
