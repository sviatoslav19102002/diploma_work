# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

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

file_location = "wasbs://outputfiles@storagename2024nulp.blob.core.windows.net/hierarchical_clustering_output.csv"
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


linked = linkage(rfm_scaled, method='ward')
plt.figure(figsize=(10, 7))
plt.title('Hierarchical Clustering Dendrogram')
dendrogram(linked, truncate_mode='lastp', p=12, leaf_rotation=45., leaf_font_size=15., show_contracted=True)

max_d = 0.7 * max(linked[:, 2])
plt.axhline(y=max_d, c='k')
plt.show()

optimal_clusters = sum(1 for h in linked[:, 2] if h >= max_d) + 1
optimal_clusters

print("Optimal number of clusters:", optimal_clusters)

# COMMAND ----------

linked = linkage(rfm_scaled, method='ward')
cluster_labels = fcluster(linked, optimal_clusters, criterion='maxclust')

rfm_data['cluster'] = cluster_labels
data_with_cluster = pd.merge(data, rfm_data, on='customer_id', how='left')


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
