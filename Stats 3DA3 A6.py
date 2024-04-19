#Step 0
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency

from sklearn.preprocessing import scale
from patsy import dmatrices, dmatrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import statsmodels.api as sm
from fancyimpute import SoftImpute

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
#Step 1
url = "https://archive.ics.uci.edu/static/public/336/data.csv"
df = pd.read_csv(url)
df
df.dtypes
#Step 2
# Transforming selected variables into categorical data type:

columns_to_convert = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'class']
for col in columns_to_convert:
   df[col] = pd.Categorical(df[col])

print(df.dtypes)
df

#Step 3
df.describe()

sns.set(style="whitegrid")
sns.boxplot(
    x='class', 
    y='age', 
    data=df, 
    )
plt.xlabel('CKD diagnosed')
plt.ylabel('age')
plt.title('Boxplot of age by diagnosis') 
plt.show()

#Step 4
# Heatmap for float variables:
corr_matrix = df.corr(numeric_only=True) 
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

cat_cols = df.select_dtypes(include='category').columns
print(cat_cols)

# Chi-Square test for categorical variables:

def chisqutest(df, catvar, tarvar):
    contingency_table = pd.crosstab(df[catvar], df[tarvar])
    chi2, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2, p_value

chisqures = []

for column in cat_cols[:-1]:  
    chi2, p_value = chisqutest(df, column, 'class')
    chisqures.append((column, chi2, p_value))

results_df = pd.DataFrame(chisqures, columns=['Variable', 'Chi-Square Stat', 'p-value'])
print(results_df)

#Step 5
missing_count=df.isnull().sum()
print(missing_count)

# Perdorming Soft Impute on the missing continious variables

floatcols = df.select_dtypes(include='float64').columns
df_float = df[floatcols]

X_softimpute = SoftImpute().fit_transform(df_float)
df_imputedfloat = pd.DataFrame(X_softimpute, columns = df_float.columns)

df2 = df.copy()

df2[floatcols] = df_imputedfloat[floatcols]
df2

# Checking the number of missing values after data imputation on float vars:
missing_count_postimpute=df2.isnull().sum()
print(missing_count_postimpute)

# Data Imputation on the 13 categorical variables using mode method:

df3 = df2.copy()

catcols = df3.select_dtypes(include='category').columns
df_cat = df3[catcols]

for column in df_cat.columns:
    mode = df_cat[column].mode()[0] 
    df_cat[column].fillna(mode, inplace=True)

df3[catcols] = df_cat[catcols]  
df3

# Checking the number of missing values post data imputation on the categorical variables:
missing_count_postimpute_cat=df3.isnull().sum()
print(missing_count_postimpute_cat)

# I noticed some cat vars whose name suggests of binary values (Yes/No), contain a 3rd bin with small data. 
#I will remove that 3rd category since it does not make logical sense for it to exist.
# The result below shows that columns "dm" and "class" have unnecessary category.
for i in cat_cols:
    print(df3[i].value_counts())

df3['dm'].replace("\tno", np.nan, inplace=True)
df3['class'].replace("ckd\t", np.nan, inplace=True)

df4=df3.dropna()
df4

#Step 6

# These variables are continious and need to be investigated for outliers
print(floatcols)

# Identifying the outliers Visually: box plot for selected variables
sns.boxplot(data=df4[['age', 'bp', 'bgr', 'bu', 'sc', 'sod']])
plt.title('Boxplot of Float Variables Before Capping')
plt.show()

sns.boxplot(data=df4[['pot', 'hemo','rbcc']])
plt.title('Boxplot of Float Variables Before Capping')
plt.show()

#Identifying the outliers Statistically: IQR score

def identify_outliers(df4, column):
    Q1 = df4[column].quantile(0.25)
    Q3 = df4[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df4[(df4[column] < lower_bound) | (df4[column] > upper_bound)]
    return outliers


outliers_age = identify_outliers(df4, 'age')
outliers_bp = identify_outliers(df4, 'bp')
outliers_bgr = identify_outliers(df4, 'bgr')
outliers_bu= identify_outliers(df4, 'bu')
outliers_sc = identify_outliers(df4, 'sc')
outliers_sod = identify_outliers(df4, 'sod')
outliers_pot = identify_outliers(df4, 'pot')
outliers_hemo = identify_outliers(df4, 'hemo')
outliers_pcv = identify_outliers(df4, 'pcv')
outliers_wbcc = identify_outliers(df4, 'wbcc')
outliers_rbcc = identify_outliers(df4, 'rbcc')
outliers_dm = identify_outliers(df4, 'dm')

# Managing the identified outliers: Capping at 95th and 5th percentiles:

def cap_outliers(df4, column):
    lower_bound = df4[column].quantile(0.05)
    upper_bound = df4[column].quantile(0.95)
    df4[column] = np.where(df4[column] < lower_bound, lower_bound, df4[column])
    df4[column] = np.where(df4[column] > upper_bound, upper_bound, df4[column])
    return df4

df4 = cap_outliers(df4, 'age')
df4 = cap_outliers(df4, 'bp')
df4 = cap_outliers(df4, 'bgr')
df4 = cap_outliers(df4, 'bu')
df4 = cap_outliers(df4, 'sc')
df4 = cap_outliers(df4, 'sod')
df4 = cap_outliers(df4, 'pot')
df4 = cap_outliers(df4, 'hemo')
df4 = cap_outliers(df4, 'pcv')
df4 = cap_outliers(df4, 'wbcc')
df4 = cap_outliers(df4, 'rbcc')
df4 = cap_outliers(df4, 'dm')

catcols
floatcols

#Step 7
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

X = df4.drop('class', axis=1)
y = df4['class'].map({'ckd': 1, 'notckd': 0})  

cat_var = catcols[:-1]
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

num_var = floatcols
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_var),
        ('cat', cat_transformer, cat_var)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso_pipe.fit(X_train, y_train)

coefs = lasso_pipe.named_steps['classifier'].coef_
print("LASSO Coefficients:", coefs)

# Unable to execute the following code (Value Error: operands could not be broadcast together with shapes)

feature_names = num_var + lasso_pipe.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(cat_var)
coefficients = lasso_pipe.named_steps['classifier'].coef_

#Step 8
#See Step 12


#Step 11
rf_initial = RandomForestClassifier(n_estimators=100, random_state=1)
rf_initial.fit(X_train, y_train)

importances = rf_initial.feature_importances_

sorted_indices = np.argsort(importances)[::-1]

k = 5
top_k_features = X_train.columns[sorted_indices[:k]]

rf_retrained = RandomForestClassifier(n_estimators=100, random_state=1)
rf_retrained.fit(X_train[top_k_features], y_train)

y_pred_retrained = rf_retrained.predict(X_test[top_k_features])


accuracy_enhanced = accuracy_score(y_test, y_pred_retrained)
print(accuracy_enhanced)

#Step 12
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df4['class'] = df4['class'].map({'ckd': 1, 'notckd': 0})

formula = 'class ~ age + bp + sg + al + su + rbc + pc + pcc + ba + bgr + bu + sc + sod + pot + hemo + pcv + wbcc + rbcc + htn + dm + cad + appet + pe + ane'

train_data, test_data = train_test_split(df4, test_size=0.2, random_state=1)

model = smf.logit(formula=formula, data=train_data).fit()

test_data['pred_prob'] = model.predict(test_data) 
test_data['pred_label'] = (test_data['pred_prob'] > 0.7).astype(int)  

accuracy = accuracy_score(test_data['class'], test_data['pred_label'])
precision = precision_score(test_data['class'], test_data['pred_label'])
print('Accuracy is', accuracy)

print(model.summary())

label_encoder = LabelEncoder()
for col in catcols:
    df4[col] = label_encoder.fit_transform(df4[col])

X = df4.drop('class', axis=1)
y = df4['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

rf_model = RandomForestClassifier(n_estimators=100, random_state=1)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print("Accuracy is ", accuracy)
print("Precision is", precision)

#Step 13
# Retraining Random Forest on all data and analyzing feature importance:

rf_model_retrained2 = RandomForestClassifier(n_estimators=100, random_state=1)
rf_model_retrained2.fit(X, y)

importances = rf_model_retrained2.feature_importances_
feature_names = X.columns

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 6))
plt.title("Feature Importances in Random Forest")
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()





