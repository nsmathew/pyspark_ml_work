from pyspark.sql import SparkSession
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import PCA
import numpy as np
import pandas as pd

# Spark logistics
spark = SparkSession.builder \
	.master("local[6]") \
	.appName("COM6012 Assignment Question 4") \
	.config("spark.local.dir", "/fastdata/acr22nsm") \
    .config("spark.executor.extraJavaOptions", "-Xss10m") \
    .config("spark.driver.extraJavaOptions", "-Xss10m") \
    .config("spark.driver.maxResultSize", "30g") \
	.getOrCreate()
	# Use a few additional options considering the size of this dataset
	# Increased JVM stack size and max result size
sc = spark.sparkContext
sc.setLogLevel("WARN")

print("\n\n-----------------------------")
print("Reading in the data...")
# Read the data
rawdata = spark.read.csv('./Data/NIPS_1987-2015.csv', header=True)

# Since no direct function for transpose, use pyspark pandas
rawdata_t = rawdata.select("*").toPandas().T.reset_index()
# Create dataframe
new_rawdata = spark.createDataFrame(rawdata_t).filter(col("index") != '_c0')
del rawdata_t # Delete the pandas df

# Change datatype to double
new_rawdata = new_rawdata.select([col(c).cast("double") for c in new_rawdata.columns])

# Create a dense vector for the features
new_rawdata = new_rawdata.rdd.map(lambda r: [Vectors.dense(r[1:])]).toDF(['features'])

print("\n\n-----------------------------")
print("Starting question A...")

# Build out stages and then pipeline
# Since the dense vector is already created we will only do a scaling and then
# apply PCA
std_scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)
pca = PCA(k=2, inputCol="scaled_features", outputCol="pca_features")
stages = [std_scaler, pca]
pipeline = Pipeline(stages=stages)

# Perform model fitting
print("\nFitting to the PCA model...")
pipeline_model = pipeline.fit(new_rawdata)

# Start printing out the outcome
print("\n\nFitting complete, key info from model as below...")
result = pipeline_model.transform(new_rawdata).select("pca_features")
print("\nThe explained variance are as below:")
print(pipeline_model.stages[-1].explainedVariance)

print("\nThe first 10 entries of the transformed data are as below:")
result.show(n=10, truncate=False)

print("\nThe first 10 entries of the PCs are as below:")
pc_mat = pipeline_model.stages[-1].pc
[print(val) for val in pc_mat.toArray()[:10]]

print("\n\n-----------------------------")
print("Starting question B...")
# Plot the PCs on a plot
# Get the first and second PC
transf = result.toPandas().to_numpy()
ys = [v[1] for v in transf[:,0]]
xs = [v[0] for v in transf[:,0]]
plt.scatter(xs, ys)
plt.ylabel("PC2")
plt.xlabel("PC1")
plt.title("NIPS papers in terms of 2 Principal Components")
plt.savefig("Output/Q4_PC_Plot.png")