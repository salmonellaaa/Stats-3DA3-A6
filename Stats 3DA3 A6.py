pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
chronic_kidney_disease = fetch_ucirepo(id=336) 

# data (as pandas dataframes) 
X = chronic_kidney_disease.data.features 
y = chronic_kidney_disease.data.targets 

# metadata 
print(chronic_kidney_disease.metadata) 

# variable information 
print(chronic_kidney_disease.variables) 
# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "https://archive.ics.uci.edu/static/public/336/data.csv"
df = pd.read_csv(url, na_values='NaN')

# 1. Classification Problem Identification
# Define target variable and features
target_variable = 'class'
features = df.columns[df.columns != target_variable]
print(features)

# 3. Dataset Overview
dataset_summary = df.describe(include='all')
print(dataset_summary)

# 4. Association Between Variables
# Analyze correlation between numerical features
corr_matrix = df.corr(numeric_only=True)  # Explicitly set numeric_only to True

# Plot correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 5. Missing Value Analysis and Handling
# Count missing values
missing_values_count = df.isnull().sum()

# Handling missing values
# For now, let's drop rows with missing values
df_cleaned = df.dropna()

# Display results
print("Missing Values Count:\n", missing_values_count)
print("\nShape of dataset after removing missing values:", df_cleaned.shape)