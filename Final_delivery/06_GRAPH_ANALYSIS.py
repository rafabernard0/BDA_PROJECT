# Databricks notebook source
# MAGIC %md
# MAGIC # Big Data Analytics Project
# MAGIC ### Group 55 – NOVA IMS
# MAGIC ##### Spring Semester 2024/2025

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📑 Table of Contents
# MAGIC - [1. Importing Libraries](#1-importing-libraries)
# MAGIC - [2. Loading Data](#2-loading-the-dataset)
# MAGIC - [3. EDA](#3-eda)
# MAGIC - [4. Degrees](#4-degrees)
# MAGIC - [5. Label Propagation Algorithm](#5-lpa)
# MAGIC - [6. Clustering](#6-clustering)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing Libraries

# COMMAND ----------

# import packages
import os
import pyspark
from pyspark.sql import SparkSession
from functools import reduce
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import col, avg, min, max, count
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pyspark.sql.types import IntegerType, NumericType, StringType
import builtins

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

from graphframes import GraphFrame
import networkx as nx


# COMMAND ----------

# Start Spark Session
spark = SparkSession.builder.appName("NYC Taxi Graph Analysis").getOrCreate()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading Data

# COMMAND ----------

# Load the dataset
data = spark.read.parquet('/dbfs/FileStore/clean/dataset_after_cleaning.parquet')

# COMMAND ----------

# select columns and cast data types
df = data.selectExpr("cast(PUlocationID as int) as src",
                   "cast(DOlocationID as int) as dst",
                   "cast(trip_distance as double) as distance",
                   "cast(total_amount as double) as amount")
df.limit(8).display()

# COMMAND ----------

# One of the location IDs is "other" -> casts to null
df.filter("src IS NULL OR dst IS NULL OR distance IS NULL OR amount IS NULL").show()

# COMMAND ----------

# drop any rows with null values
df = df.dropna()

# COMMAND ----------

# Vertices: unique locations
vertices = df.select("src").union(df.select("dst")).distinct().withColumnRenamed("src", "id")

# Edges: trips from pickup to dropoff
edges = df.select("src", "dst", "distance", "amount")

# Create the GraphFrame
gf = GraphFrame(vertices, edges)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA
# MAGIC Analyze the trips (most popular, expensive or longest)

# COMMAND ----------

# Aggregate to get average total_amount and distance as well as number of trips per edge (src → dst)
edge_agg = df.groupBy("src", "dst").agg(
    count("*").alias("trip_count"),
    avg("amount").alias("avg_amount"),
    avg("distance").alias("avg_distance")
)

# Find the most popular trips
edge_agg.orderBy('trip_count', ascending=False).limit(10).display()

# COMMAND ----------

# most expensive trips
edge_agg.orderBy('avg_amount', ascending=False).limit(10).display()

# COMMAND ----------

# longest trips
edge_agg.orderBy('avg_distance', ascending=False).limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualize

# COMMAND ----------

# Get the Top 100 Most Trafficked Routes
top_routes = edge_agg.orderBy("trip_count", ascending=False).limit(100)

# convert to pandas
top_routes_pd = top_routes.toPandas()

# COMMAND ----------

G = nx.DiGraph()

# Add edges and weights
for row in top_routes_pd.itertuples():
    G.add_edge(row.src, row.dst, weight=row.trip_count)

# Layout and draw
pos = nx.circular_layout(G) #nx.spring_layout(G, seed=42, k=10)
plt.figure(figsize=(10, 8))
edges = G.edges(data=True)
weights = [edata["weight"] for _, _, edata in edges]

# Normalize weights for better edge width scaling
min_width = 1
max_width = 9.0
min_val = builtins.min(weights)
max_val = builtins.max(weights)

# Normalize to [min_width, max_width]
scaled_widths = [
    min_width + (max_width - min_width) * ((w - min_val) / (max_val - min_val))
    for w in weights
]

nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color=weights, edge_cmap=plt.cm.Wistia, width=scaled_widths)
plt.title("Top 100 Most Trafficked Taxi Routes", fontsize=15)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Degrees (In-Degree, Out-Degree)

# COMMAND ----------

# Number of incoming trips
in_degrees = gf.inDegrees.orderBy('inDegree', ascending=False)   
in_degrees.limit(10).display()

# COMMAND ----------

# Number of outgoing trips
out_degrees = gf.outDegrees.orderBy('outDegree', ascending=False) 
out_degrees.limit(10).display()

# COMMAND ----------

# common ids among the top 10 values
set(row["id"] for row in in_degrees.limit(10).collect()).intersection(row["id"] for row in out_degrees.limit(10).collect())

# COMMAND ----------

# MAGIC %md
# MAGIC - 48,"Manhattan","Clinton East","Yellow Zone"
# MAGIC - 142,"Manhattan","Lincoln Square East","Yellow Zone"
# MAGIC - 161,"Manhattan","Midtown Center","Yellow Zone"
# MAGIC - 162,"Manhattan","Midtown East","Yellow Zone"
# MAGIC - 230,"Manhattan","Times Sq/Theatre District","Yellow Zone"
# MAGIC - 236,"Manhattan","Upper East Side North","Yellow Zone"
# MAGIC - 237,"Manhattan","Upper East Side South","Yellow Zone"

# COMMAND ----------

# MAGIC %md
# MAGIC ![](path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Label Propagation Algorithm (LPA) – Community Detection

# COMMAND ----------

# Filter edges with trip_count > threshold
threshold = 1000
filtered_edges = edge_agg.filter(edge_agg.trip_count > threshold)

# Get vertices connected by filtered edges
src_vertices = filtered_edges.select("src").distinct()
dst_vertices = filtered_edges.select("dst").distinct()
filtered_vertices = src_vertices.union(dst_vertices).distinct().withColumnRenamed("src", "id")

# Step 4: Create filtered GraphFrame
filtered_graph = GraphFrame(filtered_vertices, filtered_edges)

# COMMAND ----------

# apply label propagation algorithm
communities = filtered_graph.labelPropagation(maxIter=5)
communities.limit(10).display()

# COMMAND ----------

# only one community
communities.groupBy("label").agg(count("*").alias("count")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clustering with K-Means (Using MLlib)

# COMMAND ----------

# per-node averages using outgoing trips
trip_stats = df.groupBy("src").agg(
    avg("amount").alias("avg_amount_out"),
    avg("distance").alias("avg_distance_out")
).withColumnRenamed("src", "id")

# COMMAND ----------

# Create feature df
features_df = (
    in_degrees.join(out_degrees, "id", "outer")
    .join(trip_stats, "id", "inner")
    .na.fill(0)  # fill any missing values with 0
)

# COMMAND ----------

# Assemble features into a vector
assembler = VectorAssembler(
    inputCols=["inDegree", "outDegree", "avg_amount_out", "avg_distance_out"],
    outputCol="features"
)
feature_vector = assembler.transform(features_df)

# COMMAND ----------

# apply K-means for same K as in 03_CLUSTERING notebook
kmeans = KMeans(k=3, seed=1)
model = kmeans.fit(feature_vector)

clusters = model.transform(feature_vector)

# COMMAND ----------

# calculate number of locations in each cluster
cluster_counts = clusters.groupBy("prediction").agg(count("*").alias("count"))
cluster_counts.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Visualize Clusters

# COMMAND ----------

# create pandas df
pandas_df = clusters.toPandas()

# COMMAND ----------

# plot clustering results against Avg Distance and Avg Amount
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=pandas_df,
    x="avg_distance_out",
    y="avg_amount_out",
    hue="prediction",
    palette="tab10",  # nice categorical colors
    alpha=0.7
)
plt.title("Clusters of Locations by Avg Distance and Avg Amount")
plt.xlabel("Average Distance")
plt.ylabel("Average Amount")
plt.legend(title="Cluster")
plt.show()

# COMMAND ----------

# plot clustering results against In Degrees and Out Degrees
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=pandas_df,
    x="inDegree",
    y="outDegree",
    hue="prediction",
    palette="tab10",  # nice categorical colors
    alpha=0.7
)
plt.title("Clusters of Locations by In Degree and Out Degree")
plt.xlabel("In Degree")
plt.ylabel("Out Degree")
plt.legend(title="Cluster")
plt.show()