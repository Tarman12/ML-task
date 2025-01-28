# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers = pd.read_csv("Customers.csv")
transactions = pd.read_csv("Transactions.csv")

# Merge datasets
merged_data = transactions.merge(customers, on="CustomerID")

# Feature engineering for customer segmentation
customer_features = merged_data.groupby("CustomerID").agg({
    "TotalValue": ["sum", "mean"],  # Total and average transaction value
    "Quantity": "sum",             # Total quantity purchased
    "TransactionID": "count",      # Number of transactions
    "Region": "first",             # Region
    "SignupDate": "first"          # Signup date
}).reset_index()

# Rename columns
customer_features.columns = [
    "CustomerID", "TotalValue_Sum", "TotalValue_Mean", 
    "TotalQuantity", "TransactionCount", "Region", "SignupDate"
]

# Convert SignupDate to year
customer_features["SignupYear"] = pd.to_datetime(customer_features["SignupDate"]).dt.year

# Encode categorical features (Region)
customer_features = pd.get_dummies(customer_features, columns=["Region"], drop_first=True)

# Drop unnecessary columns
customer_features = customer_features.drop(["CustomerID", "SignupDate"], axis=1)

# Normalize features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(customer_features)

# Perform clustering (K-Means with optimal cluster range)
db_scores = []
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Evaluate clustering performance
    db_index = davies_bouldin_score(scaled_features, cluster_labels)
    silhouette = silhouette_score(scaled_features, cluster_labels)
    
    db_scores.append(db_index)
    silhouette_scores.append(silhouette)

# Find the optimal number of clusters (minimum DB Index)
optimal_clusters = np.argmin(db_scores) + 2
print(f"Optimal Number of Clusters: {optimal_clusters}")

# Re-run K-Means with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
customer_features["Cluster"] = kmeans.fit_predict(scaled_features)

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)
customer_features["PCA1"] = pca_components[:, 0]
customer_features["PCA2"] = pca_components[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=customer_features, x="PCA1", y="PCA2", hue="Cluster", palette="viridis", s=100
)
plt.title("Customer Segments (PCA Visualization)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

# Save clustering results
customer_features[["CustomerID", "Cluster"]].to_csv("Customer_Clusters.csv", index=False)

# Print metrics
print("Davies-Bouldin Scores for different cluster sizes:", db_scores)
print("Silhouette Scores for different cluster sizes:", silhouette_scores)
print("DB Index for optimal clusters:", min(db_scores))
