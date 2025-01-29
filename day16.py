import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv("healthcare_data.csv")  # Replace with actual file name

# 1. Handling Missing Data
print("Missing values before handling:\n", df.isnull().sum())

# Example: Impute numerical columns with median
df['age'].fillna(df['age'].median(), inplace=True)
df['blood_pressure'].fillna(df['blood_pressure'].mean(), inplace=True)  # Replace with relevant columns

# Example: Impute categorical columns with mode
df['gender'].fillna(df['gender'].mode()[0], inplace=True)

# 2. Detect and Handle Duplicates
print("Duplicate records before handling:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# 3. Detect and Handle Outliers using Boxplots
numerical_columns = ['age', 'blood_pressure']  # Add relevant numerical columns
for col in numerical_columns:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# Handling Outliers using Capping
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# 4. Standardize and Normalize Data
# Convert categorical variables into numerical representations
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])

# Scale numerical variables
scaler = MinMaxScaler()  # Or use StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# 5. Data Validation
print("Missing values after handling:\n", df.isnull().sum())
print("Duplicate records after handling:", df.duplicated().sum())
print("Data types:\n", df.dtypes)

# 6. Save the cleaned dataset
df.to_csv("cleaned_healthcare_data.csv", index=False)
print("Cleaned dataset saved successfully.")
