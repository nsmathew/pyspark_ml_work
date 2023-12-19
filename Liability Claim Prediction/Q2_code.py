from pyspark.sql import SparkSession
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import OneHotEncoder, StringIndexer, \
                                   StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GeneralizedLinearRegression,\
                                   LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator, \
                                   MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, log, count

# Spark logistics
spark = SparkSession.builder \
	.master("local[6]") \
	.appName("COM6012 Assignment Question 2") \
	.config("spark.local.dir", "/fastdata/acr22nsm") \
	.getOrCreate()
 
sc = spark.sparkContext
sc.setLogLevel("WARN")

# load data
rawdata = spark.read.csv('./Data/freMTPL2freq.csv', header=True).cache()

print("\n\n------------------------------")
print("Starting Question 2A...")

print("Schema after reading data...")
rawdata.printSchema()

# Exclude ID column
new_rawdata = rawdata.select(rawdata.columns[1:])

# Convert numeric columns to double data type
num_cols = ['ClaimNb', 'Exposure', 'VehPower', 'VehAge', 'DrivAge', 
            'BonusMalus', 'Density']
for column in num_cols:
    new_rawdata = new_rawdata.withColumn(column,
                                         new_rawdata[column].cast(DoubleType()))

print("Schema post data type conversion for numeric columns...")
new_rawdata.printSchema()

# Next Create NZClaim and LogClaimNb
# Create the binary column NZClaim
new_rawdata = new_rawdata.withColumn("NZClaim", 
                                     when(new_rawdata["ClaimNb"] > 0, 1.0)\
                                            .otherwise(0.0))
# Create a preprocessed ClaimNb replacing 0 with 0.5 
# Log(0.5) = -0.693... 
new_rawdata = new_rawdata.withColumn("PP_ClaimNb", 
                                     when(new_rawdata["ClaimNb"] == 0, 0.5)\
                                            .otherwise(new_rawdata["ClaimNb"]))\
                           .withColumn("PP_ClaimNb", col("PP_ClaimNb").cast("double"))        
# Create the LogClaimNb column
new_rawdata = new_rawdata.withColumn("LogClaimNb", log(col("PP_ClaimNb"))).drop(col("ClaimNb"))

print("Schema post creating columns for claims...")
new_rawdata.printSchema()

print("\n\n------------------------------")
print("Starting Question 2B...")
print("\n----------------")
print("Starting Question 2B a)...")

# Perform a traing and test split
seed = 14

"""
Function to calculate a stratified split of training and test data. For stratified split we 
will use proportionate allocation. We will calculate the fraction for each class and 
proportion the train and test data accordingly. train and test data frames are passed as 
arguments and are updated directly
"""
def stratified_split(train_ratio, source_data):
       # Empty DFs to load train and test data
       trainingData = spark.createDataFrame([], source_data.schema) 
       testData = spark.createDataFrame([], source_data.schema)
       # This will loop through a dataframe with 1 row for each class 
       # and it has a column indicating what is the fraction of that
       # class wrt to total data count
       for row in source_data.groupBy("PP_ClaimNb").count()\
                            .withColumn("sampl_frac", col("count") / source_data.count()).collect():
              train = row["sampl_frac"] * train_ratio # Calculate train ration within this class
              # Filter the data on this class and split the data randomly based on train and test ratio
              # calculated on the class' fraction
              (tempTrainingData, tempTestData) = source_data\
                                                        .filter(col("PP_ClaimNb")==row["PP_ClaimNb"])\
                                                        .randomSplit([train, (row["sampl_frac"]-train)], 
                                                               seed)
              trainingData = trainingData.unionAll(tempTrainingData)
              testData = testData.unionAll(tempTestData)
       return trainingData, testData


# Perform split of 70/30
trainingData, testData = stratified_split(0.7, new_rawdata)
# Cache both for reuse
trainingData.cache()
testData.cache()

# Print out some stats
print("Total Data count:", new_rawdata.count())
trc = trainingData.count()
tec = testData.count()
print("\nTrain split data count:", trc)
print("Test split data count:", tec)
print("Train + Test count:", trc+tec)

print("\nDistribution of the data wrt to the Claims classes in full data...")
new_rawdata.groupBy("PP_ClaimNb").agg(count("*").alias("count"))\
              .withColumn("percentage", col("count") / new_rawdata.count() * 100).show()
              
print("\nDistribution of the data in train split...")
trainingData.groupBy("PP_ClaimNb").agg(count("*").alias("count"))\
              .withColumn("percentage", col("count") / trainingData.count() * 100).show()
              
print("\nDistribution of the data test split...")
testData.groupBy("PP_ClaimNb").agg(count("*").alias("count"))\
              .withColumn("percentage", col("count") / testData.count() * 100).show()
              

print("\n----------------")
print("Starting Question 2B b)...")

# Create the stages for the ml pipeline. These will be used for the training
# and evaluation process:
#      String Indexing and One Hot Encoding of categorical columns
#      Assembling and Standardisation of numerical columns
#      Finally assembling into feature vectors
# For OHE
org_str_cols = ['Area', 'VehBrand', 'VehGas', 'Region']
idxd_cols = ['Area_idx', 'VehBrand_idx', 'VehGas_idx', 'Region_idx']
# Perform string indexing first
str_indexer = StringIndexer(inputCols=org_str_cols, outputCols=idxd_cols)
ohe_cols = ['Area_ohe', 'VehBrand_ohe', 'VehGas_ohe', 'Region_ohe']
# Folllowed by ohe
ohe = OneHotEncoder(inputCols=idxd_cols,outputCols=ohe_cols)
# Standardising numeric features
num_cols.remove('ClaimNb') # Exclude the original claims column from further pre-proc
num_assembler = VectorAssembler(inputCols=num_cols, outputCol="Num_Features")
std_scaler = StandardScaler(inputCol="Num_Features", outputCol="Scld_Num_Features")
# Vectorisation of features(numerical + categorical)
fv_assembler = VectorAssembler(inputCols=["Scld_Num_Features"]+ohe_cols, outputCol="features")

"""
Function to take in a already created ML model(before fit) and a feature vector assembler. 
The function then executes a pipeline to perform the vector assembley, training, 
predictions and reporting of RMSE and the coefficients. Argument, 'variation' is to be 
used to provide a brief descrition for the model variation to be used for printing
"""
def train_predict_report(train_data, test_data, ml_model, label, variation, eval_typ):
       print("\n*** Performing training for",variation,"...")
       # Feature vector assembler and ML Model
       # as the pipleine stages for the training and predictions
       ml_stages = [str_indexer, ohe, num_assembler, std_scaler, fv_assembler, ml_model]
       pipeline = Pipeline(stages=ml_stages)
       ml_pipelineModel = pipeline.fit(train_data) # Train
       print("Performing predictions...")
       predictions_test = ml_pipelineModel.transform(test_data)
       print("Performing evaluation...")
       if eval_typ == "RMSE":
              evaluator = RegressionEvaluator(labelCol=label, predictionCol="prediction",
                                                 metricName="rmse")
       elif eval_typ == "ACCURACY":
              evaluator = MulticlassClassificationEvaluator(labelCol=label,
                                                               predictionCol="prediction",
                                                               metricName="accuracy")
       result_test = evaluator.evaluate(predictions_test)
       # Print RMSE/Accuracy and coefficients
       print(eval_typ,":",result_test)
       print('Model Coefficients:')
       print(ml_pipelineModel.stages[-1].coefficients)
       # Also compute the training score, for CV later
       predictions_train = ml_pipelineModel.transform(train_data)
       result_train = evaluator.evaluate(predictions_train)
       return (result_train, result_test)

"""
Function which takes in a model_type indicating the type of ml model required and creates
on with the provided regParam or default value of 0.001 and returns it. The model_type caters
to the 5 variations as required in the assignment. Other hyperparameters are as specified or 
will take on the pyspark.ml default values.
"""
def get_ml_model(model_type, regParam=0.001):
       # Setup the model for Poisson Regression, using log as link function
       if model_type == "glm_poisson":
              return GeneralizedLinearRegression(featuresCol='features', 
                                                 labelCol='PP_ClaimNb', maxIter=50,\
                                                 regParam=regParam, family='poisson', link='log')
       # Setup the model for LR, including L1 reuglarisation
       elif model_type =="linear_l1":
              return LinearRegression(featuresCol='features', labelCol='LogClaimNb', maxIter=50,\
                                          regParam=regParam, elasticNetParam=1) # alpha=1, L1
       # Setup the model for LR, including L2 reuglarisation              
       elif model_type =="linear_l2":
              return LinearRegression(featuresCol='features', labelCol='LogClaimNb', maxIter=50,\
                                          regParam=regParam, elasticNetParam=0) # alpha=0, L2
       # Setup the model for Logistic Regression for classification, including L1 reuglarisation
       elif model_type =="logistic_l1":
              return LogisticRegression(featuresCol='features', labelCol='NZClaim', maxIter=50,\
                                          regParam=regParam, elasticNetParam=1)
       # Setup the model for Logistic Regression for classification, including L1 reuglarisation          
       elif model_type =="logistic_l2":
              return LogisticRegression(featuresCol='features', labelCol='NZClaim', maxIter=50,\
                                          regParam=regParam, elasticNetParam=0)
       # Invalid option
       else:
              return None
       
# Run the training and evaluation for Poisson regr, Linear regr L1 and L2, Logistic regr L1 and L2
train_predict_report(trainingData, testData, get_ml_model("glm_poisson", 0.001), 'PP_ClaimNb', 
                     "No. of claims(PP_ClaimNb) using Poisson Regression", eval_typ="RMSE")
train_predict_report(trainingData, testData, get_ml_model("linear_l1", 0.001), 'LogClaimNb', 
                     "LogClaimNb using Linear Regression and L1 Regularisation", eval_typ="RMSE")
train_predict_report(trainingData, testData, get_ml_model("linear_l2", 0.001), 'LogClaimNb', 
                     "LogClaimNb using Linear Regression and L2 Regularisation", eval_typ="RMSE")
train_predict_report(trainingData, testData, get_ml_model("logistic_l1", 0.001), 'NZClaim', 
                     "NZClaim using Logistic Regression and L1 Regularisation", eval_typ="ACCURACY")
train_predict_report(trainingData, testData, get_ml_model("logistic_l2", 0.001), 'NZClaim', 
                     "NZClaim using Logistic Regression and L2 Regularisation", eval_typ="ACCURACY")

print("\n----------------")
print("Starting Question 2B c)...")

# Here we will perform cross validation on training data with 10% 
# First we perform a 90/10 split on the training data
cv_trainingData = spark.createDataFrame([], trainingData.schema) 
cv_testData = spark.createDataFrame([], trainingData.schema)
# Perform the split
# Get 10% of the training data, maintaining the class balance
#--cv_data, _ = stratified_split(0.1, trainingData)
# Split this 10% as train/test for validation as 70/30
#--cv_trainingData, cv_testData = stratified_split(0.7, cv_data)
cv_trainingData, cv_testData = stratified_split(0.9, trainingData) #--
# Cahce both
cv_trainingData.cache()
cv_testData.cache()
# Print out some stats
#--print("Total Data count(for CV):", cv_data.count())
print("Total Data count(for CV):", trainingData.count()) #--
trc = cv_trainingData.count()
tec = cv_testData.count()
print("\nTrain split data count:", trc)
print("Test split data count:", tec)
print("Train + Test count:", trc+tec)

# Reg param options for CV
reg_param_opts = [0.001, 0.01, 0.1, 1, 10]
print("Regularisation Parameter Options for cross validation:", reg_param_opts)

# Next we perform validation for the 5 models

vc_results = {} # Hold all the rresults in the dict
for reg_opt in reg_param_opts: # Iterate through each regParam option and train+evaluate model perfomance
       results = []
       r = train_predict_report(cv_trainingData, cv_testData, get_ml_model("glm_poisson", reg_opt), 'PP_ClaimNb',
                            "PP_ClaimNb~Poisson Regression~regParam="+str(reg_opt), eval_typ="RMSE")
       results.append(r)
       r = train_predict_report(cv_trainingData, cv_testData, get_ml_model("linear_l1", reg_opt), 'LogClaimNb',
                            "LogClaimNb~Linear Regression(L1)~regParam="+str(reg_opt), eval_typ="RMSE")
       results.append(r)
       r = train_predict_report(cv_trainingData, cv_testData, get_ml_model("linear_l2", reg_opt), 'LogClaimNb',
                            "LogClaimNb~Linear Regression(L2)~regParam="+str(reg_opt), eval_typ="RMSE")
       results.append(r)

       r = train_predict_report(cv_trainingData, cv_testData, get_ml_model("logistic_l1", reg_opt), 'NZClaim',
                            "NZClaim~Logistic Regression(L1)~regParam="+str(reg_opt), eval_typ="ACCURACY")
       results.append(r)
       r = train_predict_report(cv_trainingData, cv_testData, get_ml_model("logistic_l2", reg_opt), 'NZClaim',
                            "NZClaim~Logistic Regression(L2)~regParam="+str(reg_opt), eval_typ="ACCURACY")
       results.append(r)
       vc_results[reg_opt] = results

# For plotting purposes
variations = ["PP_ClaimNb~Poisson Regression", "LogClaimNb~Linear Regression(L1)",
              "LogClaimNb~Linear Regression(L2)", "NZClaim~Logistic Regression(L1)",
              "NZClaim~Logistic Regression(L2)"]       
xs = [str(i) for i in reg_param_opts] # For x-axis
        
# Plot the the model performance for each regParam tested
for i, variation in enumerate(variations):
       train_s = []
       val_s = []
       for reg_opt, results in vc_results.items():
              val_s.append(results[i][1])
              train_s.append(results[i][0])
       print("Model:",variation, "Train Scores:", train_s, "Test Scores:", val_s)
       plt.clf()
       plt.figure(figsize=(6, 6))
       plt.plot(xs, train_s, label="Training Score", color="orange")
       plt.plot(xs, val_s, label="Cross-validation score", color="blue")
       ax = plt.gca()
       ax.legend()
       plt.ylim(round(min(train_s+val_s)-min(train_s+val_s)*0.001, 6), 
                round(max(train_s+val_s)+max(train_s+val_s)*0.001, 6))
       plt.title("Validation Curve for "+variation)
       plt.xlabel("regParam")
       plt.ylabel("Score(RMSE/Accuracy)")
       plt.savefig("Output/Q2_C_"+variation+".png", bbox_inches='tight')