from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count                                    
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml import Pipeline, pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, \
                                   MulticlassClassificationEvaluator

# Spark logistics
spark = SparkSession.builder \
	.master("local[6]") \
	.appName("COM6012 Assignment Question 5") \
	.config("spark.local.dir", "/fastdata/acr22nsm") \
	.getOrCreate()
 
sc = spark.sparkContext
sc.setLogLevel("WARN")

print("\n\n-----------------------------")
print("Reading in the data...")
# Read the data
rawdata = spark.read.csv('./Data/HIGGS.csv').cache()
seed = 220187314
ncolumns = len(rawdata.columns)
# Headers as taken from the description for the dataset from
# http://archive.ics.uci.edu/ml/datasets/HIGGS
headers = ['label', 'lepton pT', 'lepton eta', 'lepton phi', 'missing energy magnitude',	
           'missing energy phi', 'jet 1 pt', 'jet 1 eta', 'jet 1 phi',
           'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta', 'jet 2 phi',	'jet 2 b-tag',
           'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt',	
           'jet 4 eta','jet 4 phi',	'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv',	'm_jlv', 
           'm_bb', 'm_wbb', 'm_wwbb']

# Now rename the columns
schemaNames = rawdata.schema.names
for i in range(ncolumns): # from column 1 onwards are the features, assign the names
    rawdata = rawdata.withColumnRenamed(schemaNames[i], headers[i])

# Check the schema and datatypes
print("\nSchema after adding in headers...")
rawdata.printSchema()

print("\nPerforming datatype conversion to double...")
# Convert the datatypes for all columns(feature+labels) to to double
for c in rawdata.columns:
    rawdata = rawdata.withColumn(c, col(c).cast("double"))

# Check if datatype has changed accordingly
print("\nSchema after change in datatype...")
rawdata.printSchema()

print("\n\n-----------------------------")
print("Starting Question 5 A...")

"""
Function to calculate a stratified split of training and test data to maintain class balance. 
For stratified split we will use proportionate allocation. We will calculate the fraction for 
each class and proportion the train and test data accordingly. train and test data frames are 
passed as arguments and are updated directly. Reused from Question 2
"""
def stratified_split(train_ratio, source_data):
    # Empty DFs to load train and test data
    trainingDataS = spark.createDataFrame([], source_data.schema)
    testDataS = spark.createDataFrame([], source_data.schema)
    # This will loop through a dataframe with 1 row for each class 
    # and it has a column indicating what is the fraction of that
    # class wrt to total data count
    for row in source_data.groupBy("label").count()\
                        .withColumn("sampl_frac", col("count") / source_data.count()).collect():
            train = row["sampl_frac"] * train_ratio # Calculate train ration within this class
            # Filter the data on this class and split the data randomly based on train and test ratio
            # calculated on the class' fraction
            (tempTrainingData, tempTestData) = source_data\
                                                    .filter(col("label")==row["label"])\
                                                    .randomSplit([train, (row["sampl_frac"]-train)], 
                                                            seed)
            trainingDataS = trainingDataS.unionAll(tempTrainingData)
            testDataS = testDataS.unionAll(tempTestData)
    return trainingDataS, testDataS

results_acc = {}
results_curve = {}

# Prepare data for CV
print("\nPreparing the train and test data for CV with 70/30 split on 1% data...")
# For balancing the classes in the train and test we will use stratified split
# The function is similar to that written for Question 2 in the assignment
print("\nClass balance for fulldata...")
rawdata.groupBy("label").agg(count("*").alias("count"))\
              .withColumn("percentage", col("count") / rawdata.count() * 100).show()
# Use the class percentages of 0:47 and 1:53 on 1% of data to perform random sampling
cv_data = rawdata.sampleBy("label", fractions={0: (0.02*0.47), 1: (0.02*0.53)}, seed=seed)
# Perform a stratified split on the same between test and train on 70/30 ratio
cv_trainingData, cv_testData = stratified_split(0.7, cv_data)
print("\nClass balance for train data within 1% random sample...")
cv_trainingData.groupBy("label").agg(count("*").alias("count"))\
              .withColumn("percentage", col("count") / cv_trainingData.count() * 100).show()
print("\nClass balance for test data within 1% random sample...")              
cv_testData.groupBy("label").agg(count("*").alias("count"))\
              .withColumn("percentage", col("count") / cv_testData.count() * 100).show()
              
# Setup vector assembler for features
vecAssemblerw = VectorAssembler(inputCols=rawdata.columns[1:], outputCol="features")

# Setup evaluators
evaluator1 = MulticlassClassificationEvaluator(labelCol="label",
                                                predictionCol="prediction",
                                                metricName="accuracy")
evaluator2 = BinaryClassificationEvaluator(rawPredictionCol="prediction", 
                                           labelCol="label", 
                                           metricName="areaUnderROC")
# Perform cross validation
print("Performing cross validation for RandomForestClassifier")
rfc = RandomForestClassifier(labelCol="label")
paramOpts = {"maxDepth": [3, 5, 8], 
             "numTrees": [3, 5, 8], 
             "maxBins": [28, 32, 36]}
print("\nParam options being cross validated:")
print('\n'.join([f'{k}: {v}' for k, v in paramOpts.items()]))
stages = [vecAssemblerw, rfc]
pipeline = Pipeline(stages=stages)
grid = ParamGridBuilder().addGrid(rfc.maxDepth, paramOpts.get("maxDepth"))\
                         .addGrid(rfc.numTrees, paramOpts.get("numTrees"))\
                         .addGrid(rfc.maxBins, paramOpts.get("maxBins"))\
                         .build()
# For cross validation evaluation we will use accuracy only, evaluator1
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator1,\
                        parallelism=2, numFolds=2)
cvModel = cv.fit(cv_trainingData)
paramDict1 = {param[0].name: param[1] for param in cvModel.bestModel\
                                                    .stages[-1].extractParamMap().items()}
print("\nBest params as below from the cross validation for RandomForestRegressor...")
print("maxDepth:", paramDict1.get("maxDepth"))
print("maxBins:", paramDict1.get("maxBins"))
print("numTrees:", paramDict1.get("numTrees"))

# Train and evaluate 
# With cv complete evaluate on test data for the 1%
print("\nEvaluating on best model and 1% data's test set...")
print("Getting predictions...")
predictions_test = cvModel.transform(cv_testData)
print("Performing evaluation...")
results_acc["CV_Data_RFC"] = evaluator1.evaluate(predictions_test)
print("Accuracy:", results_acc.get("CV_Data_RFC"))
results_curve["CV_Data_RFC"] = evaluator2.evaluate(predictions_test)
print("AUC:", results_curve.get("CV_Data_RFC"))


# Perform cross validation for GBT classifier
print("Performing cross validation for GBTClassifier")
gbc = GBTClassifier(labelCol="label")
paramOpts = {"maxDepth": [3, 5, 8], 
             "subsamplingRate": [0.75, 0.9, 1], 
             "maxBins": [28, 32, 36]}
print("\nParam options being cross validated:")
print('\n'.join([f'{k}: {v}' for k, v in paramOpts.items()]))
stages = [vecAssemblerw, gbc]
pipeline = Pipeline(stages=stages)
grid = ParamGridBuilder().addGrid(rfc.maxDepth, paramOpts.get("maxDepth"))\
                         .addGrid(rfc.subsamplingRate, paramOpts.get("subsamplingRate"))\
                         .addGrid(rfc.maxBins, paramOpts.get("maxBins"))\
                         .build()
# For cross validation evaluation we will use accuracy only, evaluator1                         
cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator1,\
                        parallelism=2, numFolds=2)
cvModel = cv.fit(cv_data)
cvModel.avgMetrics[0]
paramDict2 = {param[0].name: param[1] for param in cvModel.bestModel\
                                            .stages[-1].extractParamMap().items()}
print("\nBest params as below from the cross validation for RandomForestRegressor...")
print("maxDepth:", paramDict2.get("maxDepth"))
print("maxBins:", paramDict2.get("maxBins"))
print("subsamplingRate:", paramDict2.get("subsamplingRate"))

# With cv complete evaluate on test data for the 1%
print("\nEvaluating on best model and 1% data's test set...")
print("Performing predictions...")
predictions_test = cvModel.transform(cv_testData)
print("Performing evaluation...")
results_acc["CV_Data_GBC"] = evaluator1.evaluate(predictions_test)
print("Accuracy:", results_acc.get("CV_Data_GBC"))
results_curve["CV_Data_GBC"] = evaluator2.evaluate(predictions_test)
print("AUC:",results_curve.get("CV_Data_GBC"))

print("\n\n-----------------------------")
print("Starting Question 5 B...")

"""
Function to perform Train and evaluation taking as an input a model,
the training data and test data. Post training, evaluation is done on 
test data to print out metric in terms of accuracy and area under 
the curve.
"""
def train_and_evaluate(model, trainData, testData, variation):
    print("\nTraining and evaluating for", variation,"...")
    stages = [vecAssemblerw, model]
    pipeline = Pipeline(stages=stages)
    ml_pipelineModel = pipeline.fit(trainData) # Train
    print("Performing predictions...")
    predictions_test = ml_pipelineModel.transform(testData)
    print("Performing evaluation...")
    results_acc[variation] = evaluator1.evaluate(predictions_test)
    print("Accuracy:", results_acc.get(variation))
    results_curve[variation] = evaluator2.evaluate(predictions_test)
    print("AUC:", results_curve.get(variation))

# Prepare train and test data on full dataset
trainingData, testData = stratified_split(0.7, rawdata)
  
# Training and evaluating for RandomForestRegressor on full data set
rfc = RandomForestClassifier(labelCol="label", maxDepth=paramDict1.get("maxDepth"),
                            maxBins=paramDict1.get("maxBins"),
                            numTrees=paramDict1.get("numTrees"))
variation = "Full_Data_RFC"
train_and_evaluate(rfc, trainingData, testData, variation)


# Training and evaluating for GBTClassifier on full data set
gbc = GBTClassifier(labelCol="label", maxDepth=paramDict2.get("maxDepth"),
                            maxBins=paramDict2.get("maxBins"),
                            subsamplingRate=paramDict2.get("subsamplingRate"))
variation = "Full_Data_GBC"
train_and_evaluate(gbc, trainingData, testData, variation)

# Print out the final summary of the results
print("\n\nFinal Summary of results:")
print("\nAccuracy:")
print('\n'.join([f'{k}: {v}' for k, v in results_acc.items()]))
print("\nArea under the curve:")
print('\n'.join([f'{k}: {v}' for k, v in results_curve.items()]))