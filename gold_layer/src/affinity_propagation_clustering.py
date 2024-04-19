# Databricks notebook source
spark.conf.set("fs.azure.account.auth.type.storagename2024nulp.dfs.core.windows.net", "SAS")
spark.conf.set("fs.azure.sas.token.provider.type.storagename2024nulp.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
spark.conf.set("fs.azure.sas.fixed.token.storagename2024nulp.dfs.core.windows.net", "sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-04-16T19:21:59Z&st=2024-04-15T11:21:59Z&spr=https&sig=4Dc7HMctc4X3UC9W8IpqH%2FZG2W4MSNLD4ARuEItkko0%3D")

# COMMAND ----------

storage_account_name = "storagename2024nulp"
storage_account_access_key = "+EKAZgZBOntuuGxM8yM325zuDJxy5TLMuwXD5drix95rbTpXL5vYrxW6BrPAEOuNk4Cj5I5BDqD/+AStttexXQ=="

# COMMAND ----------

file_location = "wasbs://outputfiles@storagename2024nulp.blob.core.windows.net/affinity_propagation_output.csv"
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
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler

# Normalize the RFM scores
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_data[['Recency', 'Frequency', 'Monetary']])

# List of parameters to try
parameters = [(0.5, -50), (0.7, -50), (0.9, -50), (0.7, -30), (0.7, -70)]

# Apply Affinity Propagation clustering for each parameter combination
for damping, preference in parameters:
    affinity_propagation = AffinityPropagation(damping=damping, preference=preference, random_state=42)
    affinity_propagation.fit(rfm_scaled)
    rfm_data['cluster'] = affinity_propagation.labels_

    # Count number of clusters
    num_clusters = len(np.unique(affinity_propagation.labels_))

    # Plotting the clusters
    plt.figure(figsize=(10, 6))

    # Scatter plot for Frequency vs Recency
    sns.scatterplot(x=rfm_scaled[:,1], y=rfm_scaled[:,0], hue=affinity_propagation.labels_,
                    palette='viridis', alpha=0.5, edgecolor=None)
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

# Merge cluster information with the original dataset
data_with_cluster = pd.merge(data, rfm_data, on='customer_id', how='left')

# Count number of clusters
num_clusters = len(np.unique(affinity_propagation.labels_))

# Plotting the clusters
plt.figure(figsize=(10, 6))

# Scatter plot for Frequency vs Recency
sns.scatterplot(x=rfm_scaled[:,1], y=rfm_scaled[:,0], hue=affinity_propagation.labels_,
                palette='viridis', alpha=0.5, edgecolor=None)
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
