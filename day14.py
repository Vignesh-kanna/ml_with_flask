# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
# Replace "drug_data.csv" with the path to your dataset file
data = pd.read_csv("Pharma_data.csv")

# Step 2: Data Cleaning
print("Dataset Information:")
print(data.info())

print("\nDataset Description:")
print(data.describe())

# Handling missing values by dropping rows with missing data
data.dropna(inplace=True)

# Step 3: Bar Plot - Average Effectiveness by Drug and Region
plt.figure(figsize=(12, 6))
sns.barplot(data=data, x="Region", y="Effectiveness", hue="Product", ci=None, palette="viridis")
plt.title("Average Effectiveness by Drug and Region", fontsize=16)
plt.xlabel("Region", fontsize=12)
plt.ylabel("Effectiveness", fontsize=12)
plt.legend(title="Product", fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 4: Violin Plot - Distribution of Effectiveness and Side Effects
plt.figure(figsize=(12, 6))
sns.violinplot(data=data, x="Product", y="Effectiveness", palette="muted")
plt.title("Effectiveness Distribution by Product", fontsize=16)
plt.xlabel("Product", fontsize=12)
plt.ylabel("Effectiveness", fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(data=data, x="Product", y="Side_Effects", palette="cool")
plt.title("Side Effects Distribution by Product", fontsize=16)
plt.xlabel("Product", fontsize=12)
plt.ylabel("Side Effects", fontsize=12)
plt.tight_layout()
plt.show()

# Step 5: Pairplot - Relationships Between Variables
sns.pairplot(data, vars=["Effectiveness", "Side_Effects", "Marketing_Spend"], hue="Product", palette="Set2", diag_kind="kde")
plt.suptitle("Pairplot of Effectiveness, Side Effects, and Marketing Spend", y=1.02, fontsize=16)
plt.show()

# Step 6: Boxplot - Effectiveness Across Trial Periods
plt.figure(figsize=(12, 6))
sns.boxplot(data=data, x="Trial_Period", y="Effectiveness", palette="pastel")
plt.title("Effectiveness Across Trial Periods", fontsize=16)
plt.xlabel("Trial Period", fontsize=12)
plt.ylabel("Effectiveness", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 7: Regression Plot - Marketing Spend vs. Effectiveness
plt.figure(figsize=(12, 6))
sns.regplot(data=data, x="Marketing_Spend", y="Effectiveness", scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
plt.title("Marketing Spend vs. Effectiveness", fontsize=16)
plt.xlabel("Marketing Spend", fontsize=12)
plt.ylabel("Effectiveness", fontsize=12)
plt.tight_layout()
plt.show()

# Step 8: Analysis
print("Analysis:")
print("- From the bar plot, identify which product has the highest average effectiveness in each region.")
print("- The violin plot shows the distribution of effectiveness and side effects for each product.")
print("- The pairplot helps visualize the relationships between effectiveness, side effects, and marketing spend.")
print("- The boxplot highlights the effectiveness variation across different trial periods.")
print("- The regression plot indicates whether higher marketing spend correlates with better effectiveness.")
