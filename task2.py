# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
customers = pd.read_csv("Customers.csv")
products = pd.read_csv("Products.csv")
transactions = pd.read_csv("Transactions.csv")

# Merge datasets
merged_data = transactions.merge(customers, on="CustomerID").merge(products, on="ProductID")

# Feature engineering for customer profiles
customer_features = merged_data.groupby("CustomerID").agg({
    "TotalValue": ["sum", "mean"],  # Total and average transaction value
    "Quantity": "sum",             # Total quantity purchased
    "Category": lambda x: x.mode()[0],  # Most frequently purchased category
    "Region": "first",             # Customer's region
    "SignupDate": "first"          # Signup date
}).reset_index()

# Rename columns
customer_features.columns = [
    "CustomerID", "TotalValue_Sum", "TotalValue_Mean", 
    "TotalQuantity", "FrequentCategory", "Region", "SignupDate"
]

# Convert SignupDate to datetime and extract year
customer_features["SignupYear"] = pd.to_datetime(customer_features["SignupDate"]).dt.year

# One-hot encode categorical features (Region, FrequentCategory)
encoder = OneHotEncoder()
encoded_categories = encoder.fit_transform(customer_features[["Region", "FrequentCategory"]]).toarray()
encoded_category_df = pd.DataFrame(encoded_categories, columns=encoder.get_feature_names_out())

# Normalize numerical features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(customer_features[["TotalValue_Sum", "TotalValue_Mean", "TotalQuantity"]])
scaled_features_df = pd.DataFrame(scaled_features, columns=["TotalValue_Sum", "TotalValue_Mean", "TotalQuantity"])

# Combine all features into a single dataframe
final_features = pd.concat([scaled_features_df, encoded_category_df], axis=1)

# Compute cosine similarity
similarity_matrix = cosine_similarity(final_features)

# Generate recommendations for the first 20 customers
lookalike_map = {}
customer_ids = customer_features["CustomerID"].tolist()

for i in range(20):  # Loop through the first 20 customers
    similarities = list(enumerate(similarity_matrix[i]))
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)  # Sort by similarity score
    top_3 = [(customer_ids[idx], round(score, 4)) for idx, score in similarities[1:4]]  # Top 3 similar customers
    lookalike_map[customer_ids[i]] = top_3

# Save the lookalike map to CSV
lookalike_df = pd.DataFrame([
    {"cust_id": key, "lookalikes": str(value)} for key, value in lookalike_map.items()
])
lookalike_df.to_csv("Lookalike.csv", index=False)

# Display the lookalike map
print(lookalike_df.head())
