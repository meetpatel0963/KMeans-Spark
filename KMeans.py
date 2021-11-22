import os
import pandas as pd
import numpy as np
import folium  

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
from pyspark.sql.functions import udf, col
import pyspark.sql.functions as F

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.ml.clustering import KMeans, BisectingKMeans, GaussianMixture

# Visualization
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 400)

import matplotlib.pyplot as plt
plt.rc('figure', figsize=(18, 4))

import seaborn as sns
sns.set(context='notebook', style='whitegrid', rc={'figure.figsize': (18,4)})

np.set_printoptions(precision=4, suppress=True)


# setting random seed for notebook reproducability
rnd_seed=42
np.random.seed=rnd_seed
np.random.set_state=rnd_seed


spark = (SparkSession
         .builder
         .master("local[*]")
         .appName("cluster-uber-trip-data")
         .getOrCreate())

spark

UBER_DATA = './data/uber.csv'

# define the schema, corresponding to a line in the JSON data file.
schema = StructType([
    StructField("dt", TimestampType(), nullable=False),
    StructField("lat", DoubleType(), nullable=False),
    StructField("lon", DoubleType(), nullable=False),
    StructField("base", StringType(), nullable=True)]
  )

# Load training data
uber_df = spark.read.csv(path=UBER_DATA, schema=schema)
uber_df = uber_df.drop("dt")
uber_df.cache()

uber_df.show(10)

uber_df.printSchema()

# How may Records?
print("Dataset Size: ", uber_df.count(), " found")


uber_df.describe(["lat", "lon"]).show()


feature_columns = ['lat', 'lon']

# Vectorize the numerical features first
feature_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

uber_assembled_df = feature_assembler.setHandleInvalid("skip").transform(uber_df)
uber_assembled_df.cache()

uber_assembled_df.show(10)

train_df, test_df = uber_assembled_df.randomSplit([0.7, 0.3], seed=rnd_seed)

# cache the training and testing set
train_df.cache()
test_df.cache()

# remove the not needed dataframes
uber_df.unpersist()
uber_assembled_df.unpersist()


kmeans = KMeans(k=6, initMode='k-means||', featuresCol='features', predictionCol='cluster', maxIter=10)

kmModel = kmeans.fit(train_df)

centroids = []
for center in kmModel.clusterCenters():
    centroids.append((center[0], center[1]))

print("KMeans Cluster Centers: ", centroids)

# Plotting the centroids on google map using Folium library.
map_ = folium.Map(width=800, height=600, location=[40.79658011772687, -73.87341741832425], zoom_start = 25)

for point in range(0, len(centroids)):
    folium.Marker(centroids[point], popup = folium.Popup(centroids[point])).add_to(map_)

map_.save("./image.html")

test_preds = kmModel.transform(test_df)
test_preds.cache()
test_preds.show(10)

print(kmModel.summary.clusterSizes) # No of pints in each cluster


kmModel.write().overwrite().save("./data/model")

