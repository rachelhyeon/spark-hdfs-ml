"""
Script for predicting loan status using covariates in LendingClub data.
"""
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import RFormula

# Load in the data
spark = SparkSession \
    .builder \
    .appName("ML for Lending Club") \
    .getOrCreate()

# Use spark.read.parquet() to load the data into variable df
df = spark.read.parquet("hdfs://10.0.0.4:9000/lending-club.parquet")

# Extract features that will be used for prediction
extract_cols = ["id", "annual_inc", "delinq_2yrs", "purpose",
                "mort_acc", "pub_rec_bankruptcies", "fico_range_high",
                "fico_range_low", "max_bal_bc", "total_rec_late_fee",
                "loan_status"]

# Subset data frame for prediction
t_df = df \
    .filter(df.loan_status != "Current") \
    .select(*extract_cols)

# Function for categorizing loans into three categories


def loan_cat(loan_status, total_rec_late_fee):
    """ Create a new column categorizing loans into three categories
    Category 1: Loans fully paid off, with no late payments
    Category 2: Loans paid back, but late
    Category 3: Loans never fully paid back
    """
    if loan_status == "Fully Paid" and total_rec_late_fee == 0:
        return 0
    elif loan_status == "Fully Paid" and total_rec_late_fee > 0:
        return 1
    else:
        return 2


func_udf = F.udf(loan_cat, IntegerType())

t_df = t_df \
    .withColumn("loan_cat", func_udf(df.loan_status, df.total_rec_late_fee)) \
    .drop("id", "loan_status", "total_rec_late_fee") \
    .na.drop()

# Remove StringIndexer and VectorAssembler and replace it with RFormula
# RFormula will automatically convert strings to categorical variable
rf_features = RFormula(formula="loan_cat ~ .")

# Add label and feature columns according to the formula
prepared_data = rf_features.fit(t_df).transform(t_df)

# Set aside 30% of data as a test set
train, test = prepared_data.randomSplit([0.7, 0.3])

# Use MinMaxScaler to scale all variables
scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

# Fit decision tree and use default parameters
dt = DecisionTreeClassifier(labelCol="label", featuresCol="scaledFeatures")
dt_pipeline = Pipeline(stages=[scaler, dt])

# Train decision tree model
dt_model = dt_pipeline.fit(train)
dt_predictions = dt_model.transform(test)

# Fit random forest and use default parameters
rf = RandomForestClassifier(labelCol="label", featuresCol="scaledFeatures")
rf_pipeline = Pipeline(stages=[scaler, rf])

# Train random forest model
rf_model = rf_pipeline.fit(train)
rf_predictions = rf_model.transform(test)

# Fit multilayer perceptron - 3 layers of 40 nodes
# Input layer of size 9, Output of size 3
layers = [19, 40, 40, 40, 3]
mp = MultilayerPerceptronClassifier(labelCol="label",
                                    featuresCol="scaledFeatures",
                                    layers=layers)
mp_pipeline = Pipeline(stages=[scaler, mp])

# Train multilayer perceptron model
mp_model = mp_pipeline.fit(train)
mp_predictions = mp_model.transform(test)

# Evaluate using MulticlassClassificationEvaluator
# metricName = "f1" and metricName="accuracy
evaluator = MulticlassClassificationEvaluator() \
    .setPredictionCol("prediction") \
    .setLabelCol("loan_cat")

# F1 score for decision tree
f1_dt = evaluator.evaluate(dt_predictions)

# Accuracy for decision tree
acc_dt = evaluator.evaluate(dt_predictions, {evaluator.metricName: "accuracy"})

# F1 score for random forest
f1_rf = evaluator.evaluate(rf_predictions)

# Accuracy for random forest
acc_rf = evaluator.evaluate(rf_predictions, {evaluator.metricName: "accuracy"})

# F1 score for MLP
f1_mp = evaluator.evaluate(mp_predictions)

# Accuracy for MLP
acc_mp = evaluator.evaluate(mp_predictions, {evaluator.metricName: "accuracy"})

# Create data frame of test set metrics
test_metrics = spark.createDataFrame(
    [
        (f1_dt, acc_dt, f1_rf, acc_rf, f1_mp, acc_mp),
    ],
    ["F1 Score of Decision Tree", "Accuracy of Decision Tree",
     "F1 Score of Random Forest", "Accuracy of Random Forest",
     "F1 Score of Multilayer Perceptron", "Accuracy of Multilayer Perceptron"]
)

# Save test set metrics to a file
test_metrics.write.csv("hdfs://10.0.0.4:9000/mhyeon/mhyeon-hw03.csv",
                       header=True, mode="overwrite")
