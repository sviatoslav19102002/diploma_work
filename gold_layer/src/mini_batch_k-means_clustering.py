# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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

file_location = "wasbs://outputfiles@storagename2024nulp.blob.core.windows.net/mini_batch_k-means_output.csv"
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


silhouette_scores = []
batch_size = 100
for i in range(2, 11):
    mbk = MiniBatchKMeans(n_clusters=i, init='k-means++', batch_size=batch_size, random_state=42)
    mbk.fit(rfm_scaled)
    score = silhouette_score(rfm_scaled, mbk.labels_)
    silhouette_scores.append(score)


plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()



k_optimal = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]
mbk = MiniBatchKMeans(n_clusters=k_optimal, random_state=42, batch_size=batch_size)
mbk.fit(rfm_scaled)
rfm_data['cluster'] = mbk.labels_
data_with_cluster = pd.merge(data, rfm_data[['customer_id', 'cluster']], on='customer_id', how='left')


plt.figure(figsize=(10, 6))

plt.scatter(rfm_scaled[:,1], rfm_scaled[:,0], c=mbk.labels_, cmap='viridis', alpha=0.5, edgecolor='k')
plt.xlabel('Frequency (Scaled)')
plt.ylabel('Recency (Scaled)')

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
