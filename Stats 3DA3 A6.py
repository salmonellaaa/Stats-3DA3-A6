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

# 2. data transformation
df.dtypes
# Splitting the df into features and target
X = df[features]
y = df[target_variable]
categorical_cols = X.select_dtypes(include=['object']).columns  # 'object' types are categorical
numerical_cols = X.select_dtypes(include=['float64']).columns  # 'float64' types are numerical

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Creating transformers for numerical and categorical colomns
numerical_scaler = StandardScaler() 
categorical_transformer = OneHotEncoder(handle_unknown='ignore') 

# Combining transformers into a columntransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_scaler, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Applying transformations
X_transformed= preprocessor.fit_transform(X)

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
#data imputation
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean(numerical)
    ('scaler', StandardScaler())  # Scale numerical data
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),   #Impute missing values with most freqent(aka mode)(categorical)
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # OneHot encode categorical data
])
# Apply ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, X.columns[X.dtypes != 'object']),
        ('cat', categorical_transformer, categorical_cols)])

# Apply preprocessing to the features
X_imputed = preprocessor.fit_transform(X)
new_categorical_features = preprocessor.named_transformers_['cat'].named_steps['onehot']\
.get_feature_names_out(categorical_cols) #keep categorical column names
all_features = list(numerical_cols) + list(new_categorical_features)
# Back to DataFrame
X_imputed_df = pd.DataFrame(X_imputed, columns=all_features)

# Display results
print("Missing Values Count:\n", missing_values_count)
print("Shape of dataset before handling missing values:", df.shape)
print("\nShape of dataset after handling missing values:", X_imputed_df.shape)

#check data types before further analysis
print(X_imputed_df.dtypes)

# 6. Outlier Analysis
#select the numeric features using 'select_dtypes' from pandas
numeric_features = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc'  ]
print("Numeric Features:", numeric_features)

#Plot the numeric features in a boxplot to identify outliers
#Iterates over each column, generate a multi-row, three_column grid of boxplots for each numeric column
plt.figure(figsize=(15, 15))
for i, col in enumerate(numeric_features):
    plt.subplot((len(numeric_features) + 2) // 3, 5, i + 1)
    sns.boxplot(x=X_imputed_df[col])
    plt.title(f'Boxplot of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#7. Sub-group analysis, we are doing K-means clustering
#Compute silhoutte scores
from sklearn.metrics import silhouette_score
k_values = range(2, 15)
silhouette_scores = []

# Calculate silhouette scores
from sklearn.cluster import KMeans
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_imputed_df)  # Using the principal component scores for clustering
    score = silhouette_score(X_imputed_df, labels)
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

#Calculate ARI after choosing K=6 for K-means clustering
from sklearn.metrics import adjusted_rand_score
y = df['class'].values  # True labels as a numpy array

#k-means
kmeans = KMeans(n_clusters=6, n_init=20, random_state=1)
kmeans.fit(X) 

# Calculate ari
rand_index = adjusted_rand_score(y, kmeans.labels_)
print(f"ARI: {rand_index}")

#Visualization of the clusters in PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(X_imputed_df)
clusters = kmeans.fit_predict(X_imputed_df)
plt.figure(figsize=(8, 6))
plt.scatter(pca_scores[:, 0], pca_scores[:, 1], c=clusters, cmap='plasma', marker='o')
plt.title('Sub-group Analysis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label').set_label('Cluster Label', rotation=270, labelpad=15)
plt.grid(True)
plt.show()

#8. 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed_df, y, test_size=0.3, random_state=1, stratify=y)

#9.we choose logistic regression and random forest as classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Initialization
log_reg = LogisticRegression(random_state=1)
random_forest = RandomForestClassifier(n_estimators=100, random_state=1)

#10.
from sklearn.metrics import accuracy_score, f1_score

# Training the models
log_reg.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

# Predictions
y_pred_log_reg = log_reg.predict(X_test)
y_pred_rf = random_forest.predict(X_test)

# Accuracy and F1
log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
log_reg_f1 = f1_score(y_test, y_pred_log_reg, average='weighted')
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')
print(f"Logistic regression accuracy: {log_reg_accuracy:.4f}, F1 Score: {log_reg_f1:.4f}")
print(f"Random forest accuracy: {rf_accuracy:.4f}, F1 Score: {rf_f1:.4f}")
