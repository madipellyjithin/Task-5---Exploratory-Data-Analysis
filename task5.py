import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
df = pd.read_csv("train.csv")
df.head()
print("Rows and Columns:", df.shape)
df.info()
df.describe()
df.isnull().sum().sort_values(ascending=False)
print("Duplicate Rows:", df.duplicated().sum())
categorical = df.select_dtypes(include='object').columns
for col in categorical:
    print("\n", col)
    print(df[col].value_counts())
df.hist(figsize=(15,10), bins=20)
plt.tight_layout()
plt.show()
numeric_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(15,8))
for i,col in enumerate(numeric_cols):
    plt.subplot((len(numeric_cols)//3)+1,3,i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
sns.pairplot(df.select_dtypes(include=np.number))
plt.show()
plt.figure(figsize=(7,5))
sns.scatterplot(x=df[numeric_cols[0]],
                y=df[numeric_cols[1]])
plt.title("Scatter Plot")
plt.show()
for col in categorical:
    plt.figure(figsize=(8,5))
    sns.countplot(x=df[col])
    plt.xticks(rotation=45)
    plt.title(col)
    plt.show()
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.show()
if 'Survived' in df.columns:
    sns.countplot(x='Survived', data=df)
    plt.title("Target Variable Distribution")
    plt.show()
