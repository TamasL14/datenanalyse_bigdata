from sklearn import cluster
import os
import json

def k_means_clustering(data, k):
    """Perform k-means clustering on the input data."""
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(data)
    return kmeans.labels_.tolist() # Convert numpy array to list

def dbscan_clustering(data, eps, min_samples):
    """Perform DBSCAN clustering on the input data."""
    dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)
    return dbscan.labels_.tolist() # Convert numpy array to list

json_file = "/Users/t.lukacs/Downloads/data_small/1b47d1e5-8f8d-4f52-98e8-2370a7e8d07f.json"
k = 3
eps = 0.5
min_samples = 5

# Load data from JSON file
with open(json_file, 'r') as f:
    data = json.load(f)

# Perform k-means clustering
kmeans_labels = k_means_clustering(data['data'], k)