import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the saved data
df = pd.read_csv('dog_health_data.csv')

# Select relevant features for clustering
X = df[['Temperature', 'HeartRate', 'AccelMagnitude']]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering with an optimal number of clusters (assume 4 for now)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Define cluster labels based on observations
cluster_labels = {0: 'Healthy', 1: 'Fever', 2: 'Lethargy', 3: 'Hyperactive'}
df['Health_Status'] = df['Cluster'].map(cluster_labels)

# Count occurrences of each health status
status_counts = df['Health_Status'].value_counts()

# Plot pie chart of health statuses
plt.figure(figsize=(8, 6))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=['green', 'red', 'blue', 'purple'])
plt.title('Distribution of Dog Health Status')
plt.show()

# Scatter plot of clusters with labels
plt.figure(figsize=(10, 6))
for cluster, label in cluster_labels.items():
    subset = df[df['Cluster'] == cluster]
    plt.scatter(subset['Temperature'], subset['HeartRate'], label=label, alpha=0.6)

plt.xlabel('Temperature')
plt.ylabel('Heart Rate')
plt.title('Dog Health Clusters')
plt.legend()
plt.show()
