# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Preview datasets
print("Customers Dataset:")
print(customers.head(), "\n")
print("Products Dataset:")
print(products.head(), "\n")
print("Transactions Dataset:")
print(transactions.head(), "\n")

# Check for missing values
print("Missing Values:")
print("Customers:\n", customers.isnull().sum(), "\n")
print("Products:\n", products.isnull().sum(), "\n")
print("Transactions:\n", transactions.isnull().sum(), "\n")

# Check for duplicates
print("Duplicate Rows:")
print("Customers:", customers.duplicated().sum())
print("Products:", products.duplicated().sum())
print("Transactions:", transactions.duplicated().sum(), "\n")

# Merge datasets for a comprehensive view
merged_data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")
print("Merged Dataset:")
print(merged_data.head(), "\n")

# Convert date columns to datetime
merged_data["TransactionDate"] = pd.to_datetime(merged_data["TransactionDate"])
customers["SignupDate"] = pd.to_datetime(customers["SignupDate"])

# Summary statistics
print("Summary Statistics for Transactions:")
print(merged_data.describe(), "\n")

# EDA Visualization

# 1. Transactions over time
plt.figure(figsize=(10, 5))
merged_data.groupby(merged_data["TransactionDate"].dt.date)["TransactionID"].count().plot()
plt.title("Transactions Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Transactions")
plt.grid()
plt.show()

# 2. Top-selling products
top_products = merged_data.groupby("ProductName")["Quantity"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(x=top_products.values, y=top_products.index, palette="viridis")
plt.title("Top 10 Best-Selling Products")
plt.xlabel("Quantity Sold")
plt.ylabel("Product Name")
plt.show()

# 3. Revenue by region
revenue_by_region = merged_data.groupby("Region")["TotalValue"].sum()
plt.figure(figsize=(10, 5))
sns.barplot(x=revenue_by_region.index, y=revenue_by_region.values, palette="coolwarm")
plt.title("Revenue by Region")
plt.xlabel("Region")
plt.ylabel("Total Revenue (USD)")
plt.show()

# 4. Product categories revenue
category_revenue = merged_data.groupby("Category")["TotalValue"].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(x=category_revenue.values, y=category_revenue.index, palette="Blues_r")
plt.title("Revenue by Product Category")
plt.xlabel("Total Revenue (USD)")
plt.ylabel("Category")
plt.show()

# 5. Signup trend over time
plt.figure(figsize=(10, 5))
customers.groupby(customers["SignupDate"].dt.year)["CustomerID"].count().plot(kind="bar", color="purple")
plt.title("Number of Customers Signing Up Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Customers")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(merged_data[["Quantity", "Price", "TotalValue"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

