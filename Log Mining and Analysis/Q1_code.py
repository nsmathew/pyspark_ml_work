from pyspark.sql import SparkSession
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from pyspark.sql.functions import col, split, regexp_replace, \
                                    concat, substring, \
                                    row_number, when, lit, round, coalesce
#from pyspark.sql.functions import count                                    
from pyspark.sql.window import Window
import pandas as pd
#from pyspark.sql.types import IntegerType
from pyspark.sql.functions import min as sparkMin
from pyspark.sql.functions import max as sparkMax
import numpy as np

# Spark logistics
spark = SparkSession.builder \
	.master("local[4]") \
	.appName("COM6012 Assignment Question 1") \
	.config("spark.local.dir", "/fastdata/acr22nsm") \
	.getOrCreate()
 
sc = spark.sparkContext
sc.setLogLevel("WARN")

# ---Utility functions---
# Function to plot the results as a bar graph.
# Caters to horizontal and vertical bar graphs
def plot_my_results(xs, ys, data, title, cat_label, 
                    val_label, val_idx, filename, horizontal=False):
    plt.clf()
    if horizontal:
        plt.barh(xs, ys, data=data)
        plt.xlabel(val_label)
        plt.ylabel(cat_label)
    else:
        plt.bar(xs, ys, data=data)
        plt.xlabel(cat_label)
        plt.ylabel(val_label)        
    plt.grid(True)
    plt.title(title)
    for i, d in enumerate(data.iterrows()):
        if horizontal:
            plt.text(d[1][val_idx], i, d[1][val_idx], ha='left', va='center')
        else:
            plt.text(i, d[1][val_idx], d[1][val_idx], ha='center', va='bottom')
    plt.savefig(filename, bbox_inches='tight')

# Load up the data from the archive to a dataframe. Do not cache as 
# used only once below
logFile = spark.read.text("./Data/NASA_access_log_Jul95.gz")

# ---Prepare the data---
# Create the dataframe with 5 columns
access_logs = logFile.select( \
    split("value", " ")[0].alias("host"), \
    concat(split("value", " ")[3], split("value"," ")[4]).alias("timestamp"), \
    split("value", '"')[1].alias("request"), \
    split("value", " ")[8].alias("reply_status"), \
    split("value", " ")[9].alias("reply_size") \
)
# Cleanup the timestamp column
access_logs = access_logs\
                .withColumn("timestamp", 
                            regexp_replace("timestamp", "[\\[\\]]", ""))
# Create a column for country based on the last 3 characters 
# of the host column, eg: .de, .sg etc
access_logs = access_logs.withColumn("host_cntry", substring("host", -3, 3))
access_logs = access_logs\
                .withColumn("host_cntry", 
                            when(access_logs.host_cntry == ".de","Germany") \
                .when(access_logs.host_cntry == ".ca","Canada") \
                .when(access_logs.host_cntry == ".sg","Singapore") \
                .otherwise("Other"))

# Maintain hosts of interest in a list
countries = ["Germany", "Canada", "Singapore"]

# Create a df with data filetered by country of interest
# CACHED
filteredLogs = access_logs.filter(col("host_cntry").isin(countries)).cache()

# ---For question 1A---
# a) filter the data for countries of interest 
# b) group by to get counts for each host country
hostsCounts = filteredLogs \
                .groupBy("host_cntry")\
                .count().withColumnRenamed("count", "Total_No_of_Requests")
# Print the results
print("\n\n--------------------------")
print("Results for question 1A...")
print("Requests originating from each of the 3 countries:")
hostsCounts.show(truncate=False)

# Plot the results
print("Creating plot for Question 1A...")
filename = "Output/Q1A_plot.png"
pd_plot = hostsCounts.toPandas()
pd_plot = pd_plot.sort_values("Total_No_of_Requests")
plot_my_results("host_cntry", "Total_No_of_Requests", pd_plot, \
                'Q1A - No. of requests by country in July 1995', \
                cat_label="Originating Country", \
                val_label="No. of Requests", val_idx=1, 
                filename=filename)
print("Plot saved as", filename)

print("\n\n--------------------------")
# ---For question 1B---

# For unique hosts, get distinct hosts 
# and their country and finally group by country
hostsUnique = filteredLogs\
                .select(["host", "host_cntry"]).distinct() \
                .groupBy("host_cntry")\
                .count().withColumnRenamed("count", "No_of_Unqiue_Hosts")

# For top 9 hosts by country
# Create a partition by host country and order the count of hosts in desc
winHCntry = Window.partitionBy("host_cntry").orderBy(col("count").desc())
# Create a reusable dataframe that is partitioned by host country
# a) group by to get counts for each host and host country, 
# b) create a column to indicate the ranking based on the counts and finally 
# CACHED
partioned_hosts = filteredLogs \
                .groupBy(["host", "host_cntry"]).count() \
                .withColumn("row",row_number().over(winHCntry)).cache()
# Next extract the top 9 hosts for each country, should have total 27 records
top9_hosts = partioned_hosts \
                .filter(col("row") <= 9)\
                .withColumnRenamed("count", "No. of Requests")\
                .withColumnRenamed("row", "Ranking")
print("Results for question 1B...")
print("No. of unqiue hosts by country:")
hostsUnique.show(truncate=False)
print("No. of rows in top9_hosts:", top9_hosts.count())
print("The top 9 hosts are:")
top9_hosts.show(truncate=False, n=27)

print("\n\n--------------------------")
# ---For question 1C---
# Create a dataframe with the top 9 hosts for each country and the rest of the 
# records aggregated by country with host='Other' and rank=10
# This is combined with the top 9 hosts for each country. This will give us 10 records 
# for each country, ie, top 9 hosts and rest of the hosts being aggregated
# Post this the the percentage of each record's no. of requests against the country's 
# total requests is computed by using the dataframe from earlier, 'hostsCounts'
top9_vs_rest = partioned_hosts \
        .filter(col("row")>9).groupby(["host_cntry"]).sum()\
        .select(["host_cntry", "sum(count)"])\
        .withColumnRenamed("sum(count)", "No_of_Requests")\
        .withColumn("host", lit("Other"))\
        .withColumn("rank", lit(10)).unionByName(partioned_hosts \
                                        .filter(col("row") <= 9)\
                                        .select(["host", "host_cntry", "count", "row"])\
                                        .withColumnRenamed("count", "No_of_Requests")\
                                        .withColumnRenamed("row", "rank"))\
        .join(hostsCounts, on='host_cntry', how='inner')\
        .withColumn('request_pct', col('No_of_Requests') / col('Total_No_of_Requests') * 100)\
        .withColumn("request_pct", round("request_pct", 2))\
        .orderBy(col("host_cntry").asc(), col("rank").asc()).cache()
print("Creating plots for question 1C...")
# Create 1 plot for each country 
for cntry in countries:
    filename="Output/Q1C_plot_"+cntry+".png"
    pd_plot = top9_vs_rest.filter(col("host_cntry") == cntry).toPandas()
    pd_plot = pd_plot.sort_values('request_pct')
    plot_my_results("host", "request_pct", pd_plot,\
                    'Q1C - Plot of Top 9 hosts vs rest for '+cntry, \
                    "Hosts", "Percentage of total requests(%)",  
                    filename=filename, 
                    horizontal=True, val_idx=5)
    print("Plot saved as", filename)
top9_vs_rest.unpersist() # Unpersist after use

print("\n\n--------------------------")
# ---For Question 1D---
# Identify the top host for each country. Create a dataframe with the timestamp
# split out to get day and hour. Next iterate for each country, create a full 
# grid of [min days, max days] X [ 00 hours, 23 hours] with the value being 
# the no. of requests for that day and hour or 0 if no requests.
# This is then displayed as heatmap using imshow() with days on x-axis and 
# hours on y-axis
top_hosts = partioned_hosts \
        .filter(col("row")==1).select(["host", "host_cntry"])

# Complete the below query
# CACHED
top_host_logs = filteredLogs.join(top_hosts, ["host", "host_cntry"], how="inner")\
    .select(["host", "host_cntry", "timestamp"])\
    .select("host", "host_cntry", split("timestamp", "/")[0].alias("day"),\
        split("timestamp", "/")[1].alias("month"),\
	    split("timestamp", ":")[1].alias("hour"))\
    .groupBy(["host", "host_cntry", "day", "hour"]).count()\
    .orderBy(col("host_cntry").asc(), col("day").asc(), col("hour").asc())\
    .cache()

# get top host for each country for the plot
top_hosts = {}
for row in top_host_logs.select(["host", "host_cntry"]).distinct().collect():
    top_hosts[row['host_cntry']] = row['host']

# get min and max days for each country for the plot
minmax_days = {}
for row in top_host_logs.groupBy(col("host_cntry"))\
                            .agg(sparkMin(col("day").cast("int")).alias("min_day"), 
                                 sparkMax(col("day").cast("int")).alias("max_day"))\
                            .collect():
    minmax_days[row['host_cntry']] = [row['min_day'], row['max_day']]
    
print("Creating plots for Question 1D...")
# Create the heatmaps, 1 country at a time
for cntry in countries:
    filename = "Output/Q1D_Heatmap_"+cntry+".png"
    
    # Create a template df with full extent of days and hours with 0 for the values
    # This will then be joined with the actual data to get the complete grid
    # to create the heatmap
    min_day = minmax_days.get(cntry)[0]
    max_day = minmax_days.get(cntry)[1]
    days_range = spark.range(min_day, max_day+1).selectExpr("id as day1")
    hours_range = spark.range(0, 24).selectExpr("id as hour1")
    template = days_range.crossJoin(hours_range)\
                .withColumn("host1", lit(top_hosts.get(cntry)))\
                .withColumn("host_cntry1", lit(cntry))\
                .withColumn("count1", lit(0))
    # Perform the join.
    # Example for a country with min_day = 2 and max_day = 29 dataframe will be like
    # below. Transpose of this is used for the plot.
    #    | 0|  1| ... |23|
    # | 2| 13   0  ...   9
    # | 3|  4   7  ...   0
    #  ...   ...   ... ...
    # |29|  0  35       22
    pd_plot = template.join(top_host_logs.filter(col("host_cntry")==cntry), 
                            (col("day1")==col("day")) & (col("hour1")==col("hour")), 
                            how = "outer")\
                    .withColumn("count_requests", coalesce(col("count"), col("count1")))\
                    .select(["day1", "hour1", "count_requests"]).toPandas()
    pd_plot_pivot = pd_plot.pivot(index='day1', columns='hour1', values='count_requests')
    
    # Create the heatmap
    plt.clf()
    plt.figure(figsize=(10, 8)) # Little wider to accomodate colour bar legend
    plt.yticks(np.arange(23, -1, -1), 
               labels=range(0, 24))
    plt.ylabel("Hours")
    plt.xlabel("Days")
    plt.xticks(np.arange(min_day, max_day+1, 1), 
               labels=range(min_day, max_day+1))
    plt.title("Heatmap of requests for "+str(top_hosts.get(cntry))+" from "+cntry+" in July '95")
    # Set extent and adjust to position ticks in centre of each cell
    extent = [min_day-0.5, max_day+0.5, 0-0.5, 23+0.5] 
    plt.imshow(pd_plot_pivot.T, cmap='Blues_r',  # Create heatmap
               extent=extent, aspect='auto')
    # Add the request values
    for x in range(pd_plot_pivot.shape[0]):
        for y in range(pd_plot_pivot.shape[1]):
                if pd_plot_pivot.iloc[x, y] != 0:
                        plt.text(x+min_day, 23-y, # Adjust for axis layout and spacing
                         pd_plot_pivot.iloc[x, y],
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.colorbar()
    plt.savefig(filename)
    
    print("Plot saved as", filename)
    