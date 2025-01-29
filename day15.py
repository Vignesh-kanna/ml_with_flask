import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# Load the dataset
df = pd.read_csv("healthcare_data.csv")  # Replace with actual file name

# Initial Exploratory Data Analysis (EDA)
print("Dataset Overview:\n", df.info())
print("Missing Values per Column:\n", df.isna().sum())

# Calculate the percentage of missing values
missing_percentage = (df.isna().sum() / len(df)) * 100
print("Percentage of Missing Values:\n", missing_percentage)

# Visualizing Missing Data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Handling Missing Data
# Mean/Median/Mode Imputation for Numerical Columns
numerical_columns = ['age', 'blood_pressure']  # Update as needed
df['age'].fillna(df['age'].median(), inplace=True)
df['blood_pressure'].fillna(df['blood_pressure'].mean(), inplace=True)

# Mode Imputation for Categorical Columns
categorical_columns = ['gender']  # Update as needed
df['gender'].fillna(df['gender'].mode()[0], inplace=True)

# KNN Imputation
imputer = KNNImputer(n_neighbors=5)
df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

# Evaluating the Effect of Imputation
print("Dataset after Imputation:\n", df.info())
print("Missing Values after Imputation:\n", df.isna().sum())

# Visualizing Imputation Impact
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[numerical_columns])
plt.title("Boxplot After Imputation")
plt.show()

# Save the cleaned dataset
df.to_csv("cleaned_healthcare_data.csv", index=False)
print("Cleaned dataset saved successfully.")
