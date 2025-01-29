# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv("Pharma_data.csv")

# Step 2: Data Cleaning
print("Dataset Information:")
print(data.info())

print("\nDataset Description:")
print(data.describe())

# Check for missing values
print("\nMissing Values Per Column:")
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Remove duplicate rows if any
data.drop_duplicates(inplace=True)

# Step 3: Visualizations

# Bar Plot - Total Sales per Region
plt.figure(figsize=(12, 6))
region_sales = data.groupby("Region")["Sales"].sum().reset_index()
sns.barplot(data=region_sales, x="Region", y="Sales", palette="Blues_d")
plt.title("Total Sales per Region", fontsize=16)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Total Sales", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scatter Plot - Marketing Spend vs Sales
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x="Marketing_Spend", y="Sales", hue="Region", palette="cool", alpha=0.7)
plt.title("Marketing Spend vs Sales", fontsize=16)
plt.xlabel("Marketing Spend", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.legend(title="Region")
plt.tight_layout()
plt.show()

# Boxplot - Drug Effectiveness Across Age Groups
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x="Age_Group", y="Effectiveness", palette="pastel")
plt.title("Drug Effectiveness Across Age Groups", fontsize=16)
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Effectiveness", fontsize=12)
plt.tight_layout()
plt.show()

# Line Plot - Sales Trend for Each Product Over Trial Periods
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x="Trial_Period", y="Sales", hue="Product", marker="o", palette="Set2")
plt.title("Sales Trend for Each Product Over Trial Periods", fontsize=16)
plt.xlabel("Trial Period", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.legend(title="Product")
plt.tight_layout()
plt.show()

# Heatmap - Correlation Between Sales, Marketing Spend, and Effectiveness
plt.figure(figsize=(8, 6))
corr_matrix = data[["Sales", "Marketing_Spend", "Effectiveness"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap", fontsize=16)
plt.tight_layout()
plt.show()

# Step 4: Analysis
print("Analysis:")
print("- The bar plot highlights which regions contribute the most to total sales.")
print("- The scatter plot shows the relationship between marketing spend and sales, potentially identifying diminishing returns.")
print("- The boxplot indicates age groups with higher drug effectiveness.")
print("- The line plot reveals sales trends for each product across trial periods.")
print("- The heatmap provides insights into correlations between sales, marketing spend, and effectiveness.")
