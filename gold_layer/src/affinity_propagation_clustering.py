# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AffinityPropagation
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

file_location = "wasbs://outputfiles@storagename2024nulp.blob.core.windows.net/affinity_propagation_output.csv"
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

parameters = [(0.5, -50), (0.7, -50), (0.9, -50), (0.7, -30), (0.7, -70)]


for damping, preference in parameters:
    affinity_propagation = AffinityPropagation(damping=damping, preference=preference, random_state=42)
    affinity_propagation.fit(rfm_scaled)
    rfm_data['cluster'] = affinity_propagation.labels_

    num_clusters = len(np.unique(affinity_propagation.labels_))

    plt.figure(figsize=(10, 6))


    sns.scatterplot(x=rfm_scaled[:,1], y=rfm_scaled[:,0], hue=affinity_propagation.labels_, palette='viridis', alpha=0.5, edgecolor=None)
    plt.xlabel('Frequency (Scaled)')
    plt.ylabel('Recency (Scaled)')
    plt.title(f'Affinity Propagation Clustering (Damping={damping}, Preference={preference}): {num_clusters} clusters')
    plt.legend(title='Cluster', loc='upper right')
    plt.grid(True)
    plt.show()

    print(f'Number of clusters (Damping={damping}, Preference={preference}): {num_clusters}')

# COMMAND ----------

affinity_propagation = AffinityPropagation(damping=0.5, preference=-50, random_state=42)
affinity_propagation.fit(rfm_scaled)
rfm_data['cluster'] = affinity_propagation.labels_

data_with_cluster = pd.merge(data, rfm_data, on='customer_id', how='left')
num_clusters = len(np.unique(affinity_propagation.labels_))


plt.figure(figsize=(10, 6))
sns.scatterplot(x=rfm_scaled[:,1], y=rfm_scaled[:,0], hue=affinity_propagation.labels_, palette='viridis', alpha=0.5, edgecolor=None)
plt.xlabel('Frequency (Scaled)')
plt.ylabel('Recency (Scaled)')
plt.title(f'Affinity Propagation Clustering (Damping={damping}, Preference={preference}): {num_clusters} clusters')
plt.legend(title='Cluster', loc='upper right')
plt.grid(True)
plt.show()

# COMMAND ----------

dbutils.fs.mkdirs('dbfs:/temp/output/')

# COMMAND ----------

data_with_cluster.to_csv('/dbfs/temp/output/affinity_propagation_output.csv', index=False)

# COMMAND ----------

dbutils.fs.cp('dbfs:/temp/output/affinity_propagation_output.csv', file_location)
