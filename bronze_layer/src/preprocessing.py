# Databricks notebook source
dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.text('secret_scope', 'storage-scope')
dbutils.widgets.text('secret_key', 'access-key')

# COMMAND ----------

secret_scope = dbutils.widgets.get('secret_scope')
secret_key = dbutils.widgets.get('secret_key')

storage_account_access_key = dbutils.secrets.get(secret_scope, secret_key)
storage_account_name = "storagename2024nulp"

file_location = "wasbs://inputfiles@storagename2024nulp.blob.core.windows.net/Data_Sample_Full.csv"
file_type = "csv"

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)

# COMMAND ----------

df = spark.read.format(file_type).option("inferSchema", "true").option('header', 'true').load(file_location)

column_names = df.columns

df = df.toDF(*[i.replace(' ', '_').lower() for i in column_names])

df.display()

# COMMAND ----------

df.write.format("delta").mode('overwrite').saveAsTable("workspace1.analysis.bronze_table")
