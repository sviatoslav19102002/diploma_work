# Databricks notebook source
spark.conf.set("fs.azure.account.auth.type.storagename2024nulp.dfs.core.windows.net", "SAS")
spark.conf.set("fs.azure.sas.token.provider.type.storagename2024nulp.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
spark.conf.set("fs.azure.sas.fixed.token.storagename2024nulp.dfs.core.windows.net", "sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-04-16T19:21:59Z&st=2024-04-15T11:21:59Z&spr=https&sig=4Dc7HMctc4X3UC9W8IpqH%2FZG2W4MSNLD4ARuEItkko0%3D")

# COMMAND ----------

storage_account_name = "storagename2024nulp"
storage_account_access_key = "+EKAZgZBOntuuGxM8yM325zuDJxy5TLMuwXD5drix95rbTpXL5vYrxW6BrPAEOuNk4Cj5I5BDqD/+AStttexXQ=="

# COMMAND ----------

file_location = "wasbs://outputfiles@storagename2024nulp.blob.core.windows.net/hierarchical_clustering_output.csv"
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
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])

# Hierarchical clustering
linked = linkage(rfm_scaled, method='ward')

# Create the dendrogram
plt.figure(figsize=(10, 7))
plt.title('Hierarchical Clustering Dendrogram')
dendrogram(linked, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)

# Draw a line to find the optimal number of clusters
max_d = 0.7 * max(linked[:, 2])
plt.axhline(y=max_d, c='k')
plt.show()

# Determine the number of clusters
optimal_clusters = sum(1 for h in linked[:, 2] if h >= max_d) + 1
optimal_clusters

print("Optimal number of clusters:", optimal_clusters)

# COMMAND ----------

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

linked = linkage(rfm_scaled, method='ward')

cluster_labels = fcluster(linked, optimal_clusters, criterion='maxclust')

# Assign cluster labels to the original data
rfm_data['cluster'] = cluster_labels

# Merge cluster information with the original dataset
data_with_cluster = pd.merge(data, rfm_data, on='customer_id', how='left')

# Plotting clustered data
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_with_cluster, x='Frequency', y='Recency', hue='cluster', palette='viridis', legend='full')
plt.title('Hierarchical Clustering of Customers')
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

# COMMAND ----------

dbutils.fs.mkdirs('dbfs:/temp/output/')

# COMMAND ----------

data_with_cluster.to_csv('/dbfs/temp/output/hierarchical_clustering_output.csv', index=False)

# COMMAND ----------

dbutils.fs.cp('dbfs:/temp/output/hierarchical_clustering_output.csv', file_location)
