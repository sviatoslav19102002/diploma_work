# Databricks notebook source
spark.conf.set("fs.azure.account.auth.type.storagename2024nulp.dfs.core.windows.net", "SAS")
spark.conf.set("fs.azure.sas.token.provider.type.storagename2024nulp.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
spark.conf.set("fs.azure.sas.fixed.token.storagename2024nulp.dfs.core.windows.net", "sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-04-16T19:21:59Z&st=2024-04-15T11:21:59Z&spr=https&sig=4Dc7HMctc4X3UC9W8IpqH%2FZG2W4MSNLD4ARuEItkko0%3D")

# COMMAND ----------

storage_account_name = "storagename2024nulp"
storage_account_access_key = "+EKAZgZBOntuuGxM8yM325zuDJxy5TLMuwXD5drix95rbTpXL5vYrxW6BrPAEOuNk4Cj5I5BDqD/+AStttexXQ=="

# COMMAND ----------

file_location = "wasbs://outputfiles@storagename2024nulp.blob.core.windows.net/k-means_output.csv"
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

diff = np.diff(wcss)
diff_r = diff[1:] / diff[:-1]
k_optimal = np.argmin(diff_r) + 2  # Add 2 due to zero-based indexing

# Apply K-Means clustering
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
kmeans.fit(rfm_scaled)
rfm_data['cluster'] = kmeans.labels_

data_with_cluster = pd.merge(data, rfm_data[['customer_id', 'cluster']], on='customer_id', how='left')


# Plotting the clusters
plt.figure(figsize=(10, 6))

plt.scatter(rfm_scaled[:,1], rfm_scaled[:,0], c=kmeans.labels_, cmap='viridis', alpha=0.5, edgecolor='k')
plt.xlabel('Frequency (Scaled)')
plt.ylabel('Recency (Scaled)')

plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], c='red', marker='x', s=300, label='Centroids')
plt.title('K-Means Clustering with Centroids')
plt.legend()
plt.grid(True)
plt.show()


# COMMAND ----------

dbutils.fs.mkdirs('dbfs:/temp/output/')

# COMMAND ----------

data_with_cluster.to_csv('/dbfs/temp/output/k-means_output.csv', index=False)

# COMMAND ----------

dbutils.fs.cp('dbfs:/temp/output/k-means_output.csv', file_location)
