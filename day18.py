import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from textblob import TextBlob

# Load the dataset
df = pd.read_csv("travel_reviews.csv")  # Replace with actual filename

# Handling Missing Values
print("Missing values before handling:\n", df.isna().sum())

df['Customer_Age'].fillna(df['Customer_Age'].median(), inplace=True)
df['Rating'].fillna(df['Rating'].mode()[0], inplace=True)

def fill_review_text(text):
    if pd.isna(text):
        return "No review provided"
    return text

df['Review_Text'] = df['Review_Text'].apply(fill_review_text)

# Detect and Remove Duplicates
print("Duplicate records before handling:", df.duplicated().sum())
df.drop_duplicates(inplace=True)

# Standardize Rating Values
df['Rating'] = df['Rating'].clip(1, 5)

# Correct Spelling in Tour_Package Names
def correct_spelling(text):
    return str(TextBlob(text).correct()) if pd.notna(text) else text

df['Tour_Package'] = df['Tour_Package'].apply(correct_spelling)

# Identify and Handle Outliers
numerical_columns = ['Package_Price', 'Rating']
for col in numerical_columns:
    plt.figure(figsize=(5, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

# Capping Outliers
for col in numerical_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# Convert Categorical Data to Numerical Format
label_encoder = LabelEncoder()
df['Tour_Package'] = label_encoder.fit_transform(df['Tour_Package'])

# Save the Cleaned Dataset
df.to_csv("cleaned_travel_reviews.csv", index=False)
print("Cleaned dataset saved successfully.")
