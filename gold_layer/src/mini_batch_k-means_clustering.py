# Databricks notebook source
spark.conf.set("fs.azure.account.auth.type.storagename2024nulp.dfs.core.windows.net", "SAS")
spark.conf.set("fs.azure.sas.token.provider.type.storagename2024nulp.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
spark.conf.set("fs.azure.sas.fixed.token.storagename2024nulp.dfs.core.windows.net", "sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-04-16T19:21:59Z&st=2024-04-15T11:21:59Z&spr=https&sig=4Dc7HMctc4X3UC9W8IpqH%2FZG2W4MSNLD4ARuEItkko0%3D")

# COMMAND ----------

storage_account_name = "storagename2024nulp"
storage_account_access_key = "+EKAZgZBOntuuGxM8yM325zuDJxy5TLMuwXD5drix95rbTpXL5vYrxW6BrPAEOuNk4Cj5I5BDqD/+AStttexXQ=="

# COMMAND ----------

file_location = "wasbs://outputfiles@storagename2024nulp.blob.core.windows.net/mini_batch_k-means_output.csv"
file_type = "csv"

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)

# COMMAND ----------

rfm_data = spark.table('workspace1.analysis.silver_table').toPandas()
rfm_data

# COMMAND ----------

data = spark.table('workspace1.analysis.bronze_table').toPandas()

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])

# Determine the optimal number of clusters using silhouette scores
silhouette_scores = []
batch_size = 100
for i in range(2, 11):  # start from 2 clusters since silhouette score cannot be computed with 1 cluster
    mbk = MiniBatchKMeans(n_clusters=i, init='k-means++', batch_size=batch_size, random_state=42)
    mbk.fit(rfm_scaled)
    score = silhouette_score(rfm_scaled, mbk.labels_)
    silhouette_scores.append(score)

# Plot silhouette scores to find the best k
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Find the optimal number of clusters based on the highest silhouette score
k_optimal = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]

# Apply Mini Batch K-Means clustering with the optimal number of clusters
mbk = MiniBatchKMeans(n_clusters=k_optimal, random_state=42, batch_size=batch_size)
mbk.fit(rfm_scaled)
rfm_data['cluster'] = mbk.labels_


data_with_cluster = pd.merge(data, rfm_data[['customer_id', 'cluster']], on='customer_id', how='left')

# Plotting the clusters
plt.figure(figsize=(10, 6))

# Scatter plot for Frequency vs Recency
plt.scatter(rfm_scaled[:,1], rfm_scaled[:,0], c=mbk.labels_, cmap='viridis', alpha=0.5, edgecolor='k')
plt.xlabel('Frequency (Scaled)')
plt.ylabel('Recency (Scaled)')

# Plotting centroids
plt.scatter(mbk.cluster_centers_[:, 1], mbk.cluster_centers_[:, 0], c='red', marker='x', s=300, label='Centroids')
plt.title('Mini Batch K-Means Clustering with Optimal Clusters')
plt.legend()
plt.grid(True)
plt.show()


# COMMAND ----------

dbutils.fs.mkdirs('dbfs:/temp/output/')

# COMMAND ----------

data_with_cluster.to_csv('/dbfs/temp/output/mini_batch_k-means_output.csv', index=False)

# COMMAND ----------

dbutils.fs.cp('dbfs:/temp/output/mini_batch_k-means_output.csv', file_location)
