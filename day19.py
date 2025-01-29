import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# Load the dataset
df = pd.read_csv("ecommerce_orders.csv")  # Replace with actual filename

# Identify missing data
print("Dataset Overview:\n", df.info())
print("Missing Values per Column:\n", df.isna().sum())

# Compute percentage of missing values
missing_percentage = (df.isna().sum() / len(df)) * 100
print("Percentage of Missing Values:\n", missing_percentage)

# Visualize missing data patterns
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Handling Missing Values
# Mean/Median Imputation for Numerical Columns
numerical_columns = ['Product_Price']  # Update as needed
df['Product_Price'].fillna(df['Product_Price'].median(), inplace=True)

# Mode Imputation for Categorical Columns
categorical_columns = ['Product_Category']  # Update as needed
df['Product_Category'].fillna(df['Product_Category'].mode()[0], inplace=True)

# Forward Fill or Backward Fill for Date Fields
date_columns = ['Order_Date']  # Update as needed
df['Order_Date'].fillna(method='ffill', inplace=True)

# KNN Imputation for Complex Cases
imputer = KNNImputer(n_neighbors=5)
df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

# Evaluate the Impact of Imputation
print("Dataset after Imputation:\n", df.info())
print("Missing Values after Imputation:\n", df.isna().sum())

# Visualizing Imputed Values
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[numerical_columns])
plt.title("Boxplot After Imputation")
plt.show()

# Save the cleaned dataset
df.to_csv("cleaned_ecommerce_orders.csv", index=False)
print("Cleaned dataset saved successfully.")
