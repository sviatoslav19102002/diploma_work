# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

df = spark.table('workspace1.analysis.bronze_table').withColumn('sales', col('sales').cast(DoubleType()))
df.display()

# COMMAND ----------

data = df.toPandas()

data['order_date'] = pd.to_datetime(data['order_date'])

today = pd.to_datetime('today')
rfm_data = data.groupby('customer_id').agg({
    'order_date': lambda x: (today - x.max()).days,  # Recency
    'order_id': 'count',                             # Frequency
    'sales': 'sum'                                   # Monetary
}).reset_index().rename(columns={'order_date':'Recency', 'order_id':'Frequency', 'sales':'Monetary'})

rfm_data

# COMMAND ----------

output_df = spark.createDataFrame(rfm_data)
output_df.display()

# COMMAND ----------

output_df.write.format('delta').mode('overwrite').saveAsTable('workspace1.analysis.silver_table')
