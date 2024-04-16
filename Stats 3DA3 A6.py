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

# 6. Outlier Analysis
#select the numeric features using 'select_dtypes' from pandas
numeric_features = df_cleaned.select_dtypes(include=[np.number]).columns
print("Numeric Features:", numeric_features)

#Plot the numeric features in a boxplot to identify outliers
#Iterates over each column, generate a multi-row, three_column grid of boxplots for each numeric column
plt.figure(figsize=(15, 15))
for i, col in enumerate(numeric_features):
    plt.subplot((len(numeric_features) + 2) // 3, 5, i + 1)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7. Sub-group analysis
#Create a copy for clustering
df_clustering = df_cleaned.copy()

# Scale the numeric features
scaler = StandardScaler()
df_clustering[numeric_features] = scaler.fit_transform(df_clustering[numeric_features])

# Convert the categorical variables to numerical
categorical_features = df_clustering.select_dtypes(include=['object']).columns
for col in categorical_features:
    df_clustering[col] = df_clustering[col].astype('category').cat.codes

# Fit into a new dataframe X, drop 'class'
X = df_clustering.drop('class', axis=1)

#PCA
from sklearn.decomposition import PCA
pca = PCA()
pca_scores = pca.fit_transform(X)
pc_scores_df = pd.DataFrame(pc_scores, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=X.index)

# Plot variance explained by each principal component (Scree plot)
plt.figure(figsize=(10, 7))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, '-o', label='Individual component')
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), '-s', label='Cumulative')
plt.ylabel('Proportion of Variance Explained')
plt.xlabel('Principal Component')
plt.xlim(0.75, len(pca.explained_variance_ratio_) + 0.25)
plt.ylim(0, 1.05)
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.legend(loc='best')
plt.show()

#Compute silhoutte scores
from sklearn.metrics import silhouette_score
k_values = range(2, 15)
silhouette_scores = []

# Calculate silhouette scores
from sklearn.cluster import KMeans
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pc_scores)  
    score = silhouette_score(pc_scores, labels)
    silhouette_scores.append(score)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.xticks(k_values)
plt.grid(True)
plt.show()

#silhouette scores
print(silhouette_scores)

#Calculate ARI after choosing K=3 for K-means clustering
from sklearn.metrics import adjusted_rand_score
y = df_cleaned['class'].values  # True labels as a numpy array

#k-means
kmeans = KMeans(n_clusters=3, n_init=20, random_state=1)
kmeans.fit(pc_scores_df) 

# Calculate ari
rand_index = adjusted_rand_score(y, kmeans.labels_)
print(f"ARI: {rand_index}")

#Visualization of the clusters
clusters = kmeans.fit_predict(pc_scores_df)
plt.figure(figsize=(8, 6))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], c=clusters, cmap='plasma', marker='o')
plt.title('Sub-group Analysis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label').set_label('Cluster Label', rotation=270, labelpad=15)
plt.grid(True)
plt.show()

#8. Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   X, df_cleaned['class'], test_size=0.3, random_state=1, stratify=df_cleaned['class'])
