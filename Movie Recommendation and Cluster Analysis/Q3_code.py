from pyspark.sql import SparkSession
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, asc, split as sparkSplit, \
                                    explode, desc, avg
from pyspark.ml.recommendation import ALS
import numpy as np
import pandas as pd
from pyspark.ml.clustering import KMeans

# Spark logistics
spark = SparkSession.builder \
	.master("local[6]") \
	.appName("COM6012 Assignment Question 3") \
	.config("spark.local.dir", "/fastdata/acr22nsm") \
	.getOrCreate()
 
sc = spark.sparkContext
sc.setLogLevel("WARN")

print("\n\n-----------------------------")
print("Starting question A...")
print("\n-------------")
print("Starting question A 1 and 2...")
# Read the data
rawdata = spark.read.csv('./Data/ratings.csv', header=True)
# Read movies data for use later, CACHE this
moviesdata = spark.read.csv('./Data/movies.csv', header=True).cache()

# Peform the data type conversion to double
new_rawdata = rawdata
#new_rawdata = rawdata
for i in range(len(rawdata.columns)):
    new_rawdata = new_rawdata.withColumn(new_rawdata.columns[i],
                                         col(new_rawdata.columns[i])\
                                                .cast(DoubleType()))
new_rawdata.cache() # Cache this

# set the seed
seed=220187314
train_splits = [0.4, 0.6, 0.8]
"""
Function to accept a pre-defined ALS model, perform the time based split of
train/test data followed by training and evaluating of the model for the 
3 train splits provided in the assignment, [0.4, 0.6, 0.8]
"""
def als_train_evaluate(als):
    results = {}
    user_factors = {}
    training_data_collect = {}
    for split in train_splits:
        # Get number of rows for the current split
        rw_cnt = int(new_rawdata.count() * split)
        # Sort by timestamp in asc order and extract the top rows based on 
        # split percentage. This ensure the more recent records go into test
        # Cache training data for reuse later
        trainingData = new_rawdata.orderBy(asc("timestamp"))\
                                    .limit(rw_cnt).cache()
        testData = new_rawdata.subtract(trainingData)
        print("For split", str(split*100)+"%:", "Train count:", 
              trainingData.count(), 
              "Test count:", testData.count())
        model = als.fit(trainingData) # train
        # Cache userfactors for reuse later
        ufctrs = model.userFactors.cache()
        predictions = model.transform(testData) # predict
        evaluator = RegressionEvaluator(metricName="rmse", 
                                        labelCol="rating",
                                        predictionCol="prediction")
        rmse = evaluator.evaluate(predictions) # evaluate
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions) # evaluate
        evaluator.setMetricName("mse")
        mse = evaluator.evaluate(predictions) # evaluate
        print("RMSE = ",rmse, "| MAE = ", mae, "| MSE = ", mse)
        results[split] = [rmse, mae, mse]
        user_factors[split] = ufctrs
        training_data_collect[split] = trainingData
    return results, user_factors, training_data_collect

# Model as per lab settings
print("\nALS1 - with settings from Labs")
als = ALS(userCol="userId", itemCol="movieId", seed=seed, 
          coldStartStrategy="drop")
var1_results, _, _ = als_train_evaluate(als)

# Model with a different settting. Here we change rank slightly to 
# 15 which provides a slight improvement across all 3 splits
print("\n\nALS2 - with settings defined by me")
als = ALS(userCol="userId", itemCol="movieId", seed=seed, 
          coldStartStrategy="drop", rank=14, maxIter=17)
var2_results, user_factors2, training_data2 = als_train_evaluate(als)

print("\n-------------")
print("Starting question A 3...")
# Plot the metrics
xs = [str(i*100)+"%" for i in train_splits] # For x-axis
ys1_arr = np.zeros((len(train_splits), 3))
ys2_arr = np.zeros((len(train_splits), 3))


markers = ['bo--', 'go-.', 'rs--']
markers2 = ['bo--', 'go-.', 'rs--']
l = len(train_splits)
all_ys1 = np.concatenate(list(var1_results.values()), axis=0)
all_ys2 = np.concatenate(list(var2_results.values()), axis=0)
lbls = ("A - RMSE", "B - RMSE", "A - MAE", 
        "B - MAE", "A - MSE", "B - MSE")
table = []# To print the data as a table
print("Performance for 2 ALS settings across 3 splits...")
for i in range(l):
    table.append([all_ys1[i], all_ys1[i+l], all_ys1[i+l*2]])
    plt.plot(xs, [all_ys1[i], all_ys1[i+l], all_ys1[i+l*2]], 
             markers[i], alpha=0.4)
    table.append([all_ys2[i], all_ys2[i+l], all_ys2[i+l*2]])
    plt.plot(xs, [all_ys2[i], all_ys2[i+l], all_ys2[i+l*2]], 
             markers[i])
plt.title("Performance of ALS models with 2 settings")
plt.xlabel("Train Splits")
plt.ylabel("Metrics Score")
ax = plt.gca()
ax.legend(labels = lbls)
plt.savefig("Output/Q3A_MetricsPlot.png", bbox_inches='tight')
# print the table
print(pd.DataFrame(table, columns = ["40%", "60%", "80%"], index=lbls))
print("\n[A - ALS model with settings from Lab]")
print("[B - ALS model with settings defined by me]")
        
print("\n\n-----------------------------")
print("Starting question B...")

print("\n-------------")
print("Starting question B 1 and 2...")
print("\nPerforming training for Kmeans training for user "+\
      "factors for ALS Model B setting. k=25")
kmeans = KMeans(k=25, seed=seed)  # Two clusters with seed = 1
clusters = {}
genres = {}
# For each train split we will perform actions required for questions 
# B 1 and B 2. This will include the training of kmeans model, finding 
# the top 5 clustersfor each split as well the top 10 genres for the 
# users from the top cluster for each split
for split in train_splits:
    model = kmeans.fit(user_factors2.get(split))
    summary = model.summary # Get the summary to pull out cluster sizes
    # Get 5 largest clusters for the current split
    clusters[split] =  sorted(summary.clusterSizes, reverse=True)[:5]
    # Use transform to get the rows for the largest cluster
    transformed = model.transform(user_factors2.get(split))
    max_cluster = transformed.groupBy('prediction').count()\
                                .orderBy('count', ascending=False)\
                                .first()['prediction'] # max cluster no.
    # Get the distinct users from the largest cluster
    max_cluster_users = transformed.filter(transformed.prediction == max_cluster)\
                                    .select("id")\
                                    .distinct() # max cluster users
    # Get movies and average ratings for users from largest cluster - Q3 B 2
    movies_largest_cluster = training_data2.get(split)\
                                .join(max_cluster_users,
                                      col("userId") == col("id"), "inner")\
                                .groupBy("movieId")\
                                .agg(avg("rating").alias("avg_rating"))
    # Get movies with average ratings as 4 or higher - Q3 B 2
    top_movies = movies_largest_cluster.filter(col("avg_rating") >= 4)
    # Get the movie info for the 4 or higher rated movies by joining with the 
    # movies data set - Q3 B 2
    user_movies = moviesdata.withColumnRenamed("movieId", "movieId2")\
                            .join(top_movies, 
                                  col("movieId") == col("movieId2"), 
                                  "inner")\
                            .drop("movieId2")
    # Finally get the top 10 genres by first splitting the pipe 
    # delimited 'generes' column. Then convert the split column 
    # into rows. Then group by the genre, count and pick the 
    # generes that have 10 largest counts - Q3 B 2
    top_genres = user_movies.withColumn("genres_each", 
                                        sparkSplit("genres", "\|"))\
                            .select(col("movieId"), 
                                    col("title"), 
                                    explode(col("genres_each"))\
                            .alias("genre"))\
                            .groupBy("genre")\
                            .count()\
                            .orderBy(desc("count"))\
                            .limit(10)\
                            .select("genre")
    # Store the top 10 genres for use later
    genres[str(split*100)+"%"] = [row['genre'] for row in top_genres.collect()]

# Print results for Question B 1    
print("\nTop 5 clusers for each of the 3 splits with no. of users in them...\n")
print(pd.DataFrame(clusters))

# Start the plot
# Each split will be a point on the x-axis
# Each of the top 5 clusters sizes will be shown as a horizontal bar against the 
# split on the x-axis
plt.clf()
b_width = 0.15
xns = np.array(range(0, 3)) # For x-axis, numeric values to adjust pointion of bar values
fig, ax = plt.subplots()
i=0
for split, curr_clusters in clusters.items():
    # 1 vertical bar for each of the top 5 clusters   
    b1 = ax.bar(xns[i] - 2*b_width, curr_clusters[0], b_width, label='Rank 1')
    b2 = ax.bar(xns[i] - b_width, curr_clusters[1], b_width, label='Rank 2')
    b3 = ax.bar(xns[i], curr_clusters[2], b_width, label='Rank 3')
    b4 = ax.bar(xns[i] + b_width, curr_clusters[3], b_width, label='Rank 4')
    b5 = ax.bar(xns[i] + 2*b_width, curr_clusters[4], b_width, label='Rank 5')
    i += 1
    # Addin the values of the bars
    for bars in [b1, b2, b3, b4, b5]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, 
                     yval + 0.3, yval, 
                     ha="center", fontsize="x-small")
ax.set_ylabel('Cluster Sizes')
ax.set_xlabel('Train Data Splits')
ax.set_title('Top-5 Cluster Sizes for ALS Setting 2 and Train Splits')
ax.set_xticks(range(len(xs)), xs)
plt.savefig("Output/Q3B_1_ClustersPlot.png", bbox_inches='tight')

# Print results for Question B 2
print("\nTop 10 genres for users from largest cluster for each of the 3 splits...\n")
print(pd.DataFrame(genres))