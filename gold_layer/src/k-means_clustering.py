# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.text('secret_scope', 'storage-scope')
dbutils.widgets.text('secret_key', 'access-key')

# COMMAND ----------

secret_scope = dbutils.widgets.get('secret_scope')
secret_key = dbutils.widgets.get('secret_key')

storage_account_access_key = dbutils.secrets.get(secret_scope, secret_key)
storage_account_name = "storagename2024nulp"

file_location = "wasbs://outputfiles@storagename2024nulp.blob.core.windows.net/k-means_output.csv"
file_type = "csv"

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)

# COMMAND ----------

rfm_data = spark.table('workspace1.analysis.silver_table').toPandas()
data = spark.table('workspace1.analysis.bronze_table').toPandas()

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


kmeans = KMeans(n_clusters=k_optimal, random_state=42)
kmeans.fit(rfm_scaled)
rfm_data['cluster'] = kmeans.labels_
data_with_cluster = pd.merge(data, rfm_data[['customer_id', 'cluster']], on='customer_id', how='left')



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
