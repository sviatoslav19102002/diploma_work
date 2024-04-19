# Databricks notebook source
spark.conf.set("fs.azure.account.auth.type.storagename2024nulp.dfs.core.windows.net", "SAS")
spark.conf.set("fs.azure.sas.token.provider.type.storagename2024nulp.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.sas.FixedSASTokenProvider")
spark.conf.set("fs.azure.sas.fixed.token.storagename2024nulp.dfs.core.windows.net", "sv=2022-11-02&ss=bfqt&srt=sco&sp=rwdlacupiytfx&se=2024-04-16T19:21:59Z&st=2024-04-15T11:21:59Z&spr=https&sig=4Dc7HMctc4X3UC9W8IpqH%2FZG2W4MSNLD4ARuEItkko0%3D")

# COMMAND ----------

storage_account_name = "storagename2024nulp"
storage_account_access_key = "+EKAZgZBOntuuGxM8yM325zuDJxy5TLMuwXD5drix95rbTpXL5vYrxW6BrPAEOuNk4Cj5I5BDqD/+AStttexXQ=="

# COMMAND ----------

file_location = "wasbs://inputfiles@storagename2024nulp.blob.core.windows.net/Data_Sample_Full.csv"
file_type = "csv"

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key."+storage_account_name+".blob.core.windows.net",
  storage_account_access_key)

# COMMAND ----------

df = spark.read.format(file_type).option("inferSchema", "true").option('header', 'true').load(file_location)\

column_names = df.columns

df = df.toDF(*[i.replace(' ', '_').lower() for i in column_names])

# COMMAND ----------

df.display()

# COMMAND ----------

df.write.format("delta").mode('overwrite').saveAsTable("workspace1.analysis.bronze_table")
