# Databricks notebook source
# MAGIC %md
# MAGIC # Big Data Analytics Project
# MAGIC ### Group 55 – NOVA IMS
# MAGIC ##### Spring Semester 2024/2025

# COMMAND ----------

# MAGIC %md
# MAGIC ## 📑 Table of Contents
# MAGIC - 1. Set Up
# MAGIC - 2. EDA + feature selection
# MAGIC - 3. Feature Engineering, Vectorization, Train-Val-Test Split
# MAGIC - 4. Linear Regression
# MAGIC - 5. Random Forest Regressor
# MAGIC - 6. GBT Regressor
# MAGIC - 7. Update Features
# MAGIC - 8. Results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set Up

# COMMAND ----------

# Install missing dependencies
%pip install mlflow
%pip install ucimlrepo

# COMMAND ----------

# import packages
import os
import pyspark
from pyspark.sql import SparkSession
from functools import reduce
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, unix_timestamp, dayofweek, dayofmonth, udf, hour, minute, unix_timestamp, concat_ws
from IPython.display import Image, display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pyspark.sql.types import IntegerType, NumericType, StringType, FloatType, DoubleType
import textwrap
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_regression

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Correlation

import mlflow
import mlflow.pyspark.ml

# COMMAND ----------

# Start Spark Session
spark = SparkSession.builder.appName("Predict price") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()

# COMMAND ----------

# Load the dataset
data = spark.read.parquet('/dbfs/FileStore/clean/dataset_after_cleaning.parquet').limit(2000000)

# COMMAND ----------

# Check the data (number of rows, schema)
print(data.count())
data.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select columns that provide value and can be used for the analysis (are accessible at the time of analysis)
# MAGIC Exclude
# MAGIC - VendorID (indicates the TPEP provider that provided record) -> not relevant
# MAGIC - RatecodeID, payment_type, trip_duration_minutes, trip_speed_mph (at the end of the trip) -> not accessible
# MAGIC - fare_amount (at the end of the trip) -> not accessible
# MAGIC - tip_amount (substracted from the total amount since client can decide this. So we are predicting the price of the trip without tip.)

# COMMAND ----------

# drop above mentioned columns
data_filtered = data.drop("VendorID", "RatecodeID", "fare_amount", 'payment_type', 'trip_duration_minutes', 'trip_speed_mph')

# COMMAND ----------

# substract tip_amount
data_filtered = data_filtered.withColumn("total_amount", col("total_amount") - col("tip_amount")).drop("tip_amount")

# COMMAND ----------

# features with string datatypes can be turned to integers (except PU_DO_pair) 
[f[0] for f in data_filtered.dtypes if isinstance(data_filtered.schema[f[0]].dataType, (StringType))]

# COMMAND ----------

# Extract day of week and hour features, change datatypes to integer values
df = data_filtered.withColumn("hour", hour("tpep_pickup_datetime")) \
    .withColumn("day_of_week", dayofweek("tpep_pickup_datetime")) 

# cast string columns to integer    
for string_col in ['passenger_count', 'PULocationID', 'DOLocationID']:
    df = df.withColumn(string_col, col(string_col).cast("int"))

# check that dtypes ok
[f[0] for f in df.dtypes if isinstance(df.schema[f[0]].dataType, (StringType))]

# COMMAND ----------

# we notice that improvement_surcharge, mta_tax, congestion_surcharge are categorical -> we use stringIndexer
print(df.groupBy("improvement_surcharge").count().show())
print(df.groupBy("mta_tax").count().show())
print(df.groupBy("congestion_surcharge").count().show())

# COMMAND ----------

# cast improvement_surcharge, mta_tax, congestion_surcharge values to categorical using stringIndexer
for num_col in ['mta_tax', 'improvement_surcharge', 'congestion_surcharge']:
    df = df.withColumn(num_col, col(num_col).cast("string")) 
    stringIndexer = StringIndexer(inputCol=num_col, outputCol=f'{num_col}_indexed')
    df = stringIndexer.fit(df).transform(df)
    df = df.withColumn(f'{num_col}_indexed', col(f'{num_col}_indexed').cast("int")) 

# COMMAND ----------

# # of null values
number_of_nulls = df.filter(
    reduce(lambda x, y: x | y, [col(c).isNull() for c in df.columns])
).count()
print(number_of_nulls)

# COMMAND ----------

# remove all rows with any null values
df = df.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA + feature selection

# COMMAND ----------

# MAGIC %md
# MAGIC We split data to train, validation and test sets:
# MAGIC - Train set: Used to fit models.
# MAGIC - Validation set: Used to compare across model types (LR vsRF vs GBT) and feature sets.
# MAGIC - Test set: Used only once at the end to evaluate final chosen model — gives an unbiased estimate of generalization performance.
# MAGIC - In addition, CV on train set for RF and GBT: Used to tune hyperparameters (e.g. depth, max bins).

# COMMAND ----------

# Split data
train_data, val_data, test_data = df.randomSplit([0.6, 0.2, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical features

# COMMAND ----------

# select categorical features
categorical_features = [f[0] for f in df.dtypes if isinstance(train_data.schema[f[0]].dataType, (StringType, IntegerType))]
categorical_features

# COMMAND ----------

# Remove 'mta_tax', 'improvement_surcharge', 'congestion_surcharge' and use indexed versions; remove PU_DO_pairs as well, since we have PU and DO locations separately
categorical_features = ['passenger_count',
 'PULocationID',
 'DOLocationID',
 'airport_fee',
 'is_weekend',
 'is_holiday',
 'is_shared_ride',
 'has_toll',
 'hour',
 'day_of_week',
 'mta_tax_indexed',
 'improvement_surcharge_indexed',
 'congestion_surcharge_indexed']

# COMMAND ----------

# EDA: categorical features
pd_df = train_data.toPandas()

# Set up the figure for multiple bar plots
fig, axes = plt.subplots(3, 5, figsize=(20, 10))
axes = axes.flatten()

# Plot each categorical feature against the mean count
for i, feature in enumerate(categorical_features):
    plot_df = pd_df.groupby(feature)[['total_amount']].mean().reset_index()
    sns.barplot(data=pd_df, x=feature, y="total_amount", ax=axes[i], estimator=np.mean, ci=None)
    axes[i].set_title("\n".join(textwrap.wrap(f'Average price by {feature.replace("_", " ")}', width=40)))
    axes[i].set_xlabel(feature.replace("_", " "))
    axes[i].set_ylabel("Average price")

fig.delaxes(axes[-1])
fig.delaxes(axes[-2])
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The same values numerically:

# COMMAND ----------

# calculate average total_amounts for each group
for feature in categorical_features:
    train_data.groupBy(feature).avg("total_amount").orderBy("avg(total_amount)").show()

# COMMAND ----------

# MAGIC %md
# MAGIC Observations: for features is_weekend, is_holiday, is_shared_ride, and day_of_week the average price do not differ significantly between groups

# COMMAND ----------

# MAGIC %md
# MAGIC #### Mutual information
# MAGIC Calculate the mutual information scores

# COMMAND ----------

mi_features = ['passenger_count',
 #'PULocationID',
 #'DOLocationID',
 'airport_fee',
 'is_weekend',
 'is_holiday',
 'is_shared_ride',
 'has_toll',
 'hour',
 'day_of_week',
 'mta_tax_indexed',
 'improvement_surcharge_indexed',
 'congestion_surcharge_indexed']

mi_scores = mutual_info_regression(train_data.select(mi_features).toPandas(), train_data.select('total_amount').toPandas().squeeze())

mi_df = pd.DataFrame({
    'Feature': mi_features,
    'MI Score': mi_scores
}).sort_values(by='MI Score', ascending=False)

mi_df

# COMMAND ----------

# MAGIC %md
# MAGIC Exclude features: 
# MAGIC - 'day_of_week' (we keep 'is_weekend' that has higher MI-score)
# MAGIC - 'is_holiday' (no big differences on the average total_amount and low MI-score)
# MAGIC - 'has_toll' (we have numeric column 'tolls_amount')
# MAGIC - 'is_shared_ride' (we keep 'passenger_count' that has higher MI-score. The average total_amount did not differ significantly and the MI-score is low as well)
# MAGIC  

# COMMAND ----------

# final categorical features
categorical_features = ['passenger_count',
 'PULocationID',
 'DOLocationID',
 'airport_fee',
 'is_weekend',
 'hour',
 'mta_tax_indexed',
 'improvement_surcharge_indexed',
 'congestion_surcharge_indexed']

# COMMAND ----------

# Plot final features on trip_distance vs. total_amount scatter plot to see if we can visually detect some patterns
pandas_df = train_data.select(["trip_distance", "total_amount"] + categorical_features).toPandas()

# Set up the figure for multiple plots
fig, axes = plt.subplots(3, 3, figsize=(12, 8))
axes = axes.flatten()

# Plot trip_distance vs. total_amount and color points according to the groups in each categorical feature
for i, feature in enumerate(categorical_features):
    if pandas_df[feature].nunique() <= 10:
        axes[i].scatter(
        pandas_df['trip_distance'],
        pandas_df['total_amount'],
        c=pandas_df[feature],
        cmap='tab10',
        alpha=0.5,
        s=10)
    else:
        axes[i].scatter(
            pandas_df['trip_distance'],
            pandas_df['total_amount'],
            c=pandas_df[feature],
            cmap=plt.cm.get_cmap('hsv', pandas_df[feature].nunique()), 
            alpha=0.3,
            s=10)
    axes[i].set_title(f'{feature.replace("_", " ")}')
    axes[i].set_xlabel('Trip distance')
    axes[i].set_ylabel("Total Amount")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical features

# COMMAND ----------

# select numerical features
numerical_features = [f[0] for f in df.dtypes if isinstance(train_data.schema[f[0]].dataType, DoubleType)]
numerical_features

# COMMAND ----------

# remove 'total_amount' (target) and 'pickup_hour_decimal' (we have categorical variable hour)
numerical_features = ['trip_distance', 'extra', 'tolls_amount']

# COMMAND ----------

# Plot each numerical feature against total_amount
pandas_df = train_data.select(numerical_features + ['total_amount']).toPandas() 

# Set up the figure for multiple plots
fig, axes = plt.subplots(1, 3, figsize=(9,4))
axes = axes.flatten()

# Plot each feature against the total_amount
for i, feature in enumerate(numerical_features):
    axes[i].scatter(
            pandas_df[feature],
            pandas_df['total_amount'],
            alpha=0.3,
            s=10)
    axes[i].set_title(f'{feature.replace("_", " ")}')
    axes[i].set_ylabel('Total amount')
    axes[i].set_xlabel(f'{feature.replace("_", " ")}')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We observe that there are some relationships between the numerical features and the target variable. Let us next calculate correlation matrix.

# COMMAND ----------

# Create correlation matrix
assembler = VectorAssembler(inputCols=numerical_features + ['total_amount'],
outputCol='features')
df_corr_matrix = assembler.transform(train_data)
corr_matrix = Correlation.corr(df_corr_matrix, "features", "pearson").head()[0]

# COMMAND ----------

# Convert Spark DenseMatrix to NumPy array
corr_array = corr_matrix.toArray()

# Create DataFrame for seaborn
columns = numerical_features + ['total_amount']
corr_df = pd.DataFrame(corr_array, columns=columns, index=columns)

# COMMAND ----------

# plot correlation matrix
plt.figure(figsize=(6, 5))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Since all numerical features show at least some correlation with the target variable, even if weak in the case of extra, we retain all of them at this stage. From the scatter plot, we observe that as the value of extra increases, the spread of total_amount decreases.

# COMMAND ----------

# final numerical features
numerical_features = ['trip_distance', 'extra', 'tolls_amount']

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering, Vectorization, Train-Test Split

# COMMAND ----------

# no need for indexer as all values doubles or integers
df.select(numerical_features+categorical_features).printSchema()

# COMMAND ----------

# features for baseline model
features = ['trip_distance'] 
# features for LR with all numerical features
features2 = numerical_features 
# all categorical and numerical features
features3 = categorical_features + numerical_features 

assembler = VectorAssembler(inputCols=features,
outputCol='features')

assembler2 = VectorAssembler(inputCols=features2, 
outputCol='features2')

assembler3 = VectorAssembler(inputCols=features3, 
outputCol='features3')

# COMMAND ----------

# Evaluators
evaluator_rmse = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="r2")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression

# COMMAND ----------

# Define Linear Regression model and scaler
mlflow.pyspark.ml.autolog(disable=True) 

lr = LinearRegression(featuresCol="features_scaled", labelCol="total_amount")
scaler = StandardScaler(inputCol='features', outputCol="features_scaled", withMean=True, withStd=True)

with mlflow.start_run(run_name="LR_Baseline_Model"):
    # Train the model
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    lr_model = pipeline.fit(train_data)

    # Make predictions
    predictions = lr_model.transform(val_data)

    # Evaluate model
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    mlflow.log_metrics({
        "rmse_val_data": rmse,
        "r2_val_data": r2,
        "mae_val_data": mae
    })

    # Show results
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2: {r2}")
    print(f"MAE: {mae}")
    predictions.select("features", "total_amount", "prediction").show(10)

    # Model coefficients and intercept
    print(f"Coefficients: {lr_model.stages[-1].coefficients}")
    print(f"Intercept: {lr_model.stages[-1].intercept}")

# COMMAND ----------

mlflow.pyspark.ml.autolog(disable=True) 

# Define Linear Regression model and scaler
lr = LinearRegression(featuresCol="features2_scaled", labelCol="total_amount")
scaler = StandardScaler(inputCol='features2', outputCol="features2_scaled", withMean=True, withStd=True)

with mlflow.start_run(run_name="LR_Model_Numerical_Features"):
    # Train the model
    pipeline = Pipeline(stages=[assembler2, scaler, lr])
    lr_model = pipeline.fit(train_data)

    # Make predictions
    predictions = lr_model.transform(val_data)

    # Evaluate model
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    mlflow.log_metrics({
        "rmse_val_data": rmse,
        "r2_val_data": r2,
        "mae_val_data": mae
    })

    # Show results
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R^2: {r2}")
    print(f"MAE: {mae}")
    predictions.select("features2", "total_amount", "prediction").show(10)

    # Model coefficients and intercept
    print(f"Coefficients: {lr_model.stages[-1].coefficients}")
    print(f"Intercept: {lr_model.stages[-1].intercept}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Regressor

# COMMAND ----------

# RF model
rf = RandomForestRegressor(featuresCol="features3", labelCol="total_amount", seed=42)

# Grid search
rf_paramGrid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [20, 50])
    .addGrid(rf.maxDepth, [3, 5])
    .addGrid(rf.maxBins, [16, 32])
    .build()
)

mlflow.pyspark.ml.autolog(disable=True) 

# rf pipeline and cross validator
rf_pipeline = Pipeline(stages = [assembler3, rf])
rf_cv = CrossValidator(
    estimator=rf_pipeline,
    estimatorParamMaps=rf_paramGrid,
    evaluator=evaluator_rmse,
    numFolds=3
)

with mlflow.start_run(run_name="RF_Model"):
    rf_cv_model = rf_cv.fit(train_data)
    predictions = rf_cv_model.transform(val_data)

    # Log all tested parameter combinations
    for params, metric in zip(rf_cv_model.getEstimatorParamMaps(), rf_cv_model.avgMetrics):
        param_str = "_".join([f"{param.name}_{val}" for param, val in params.items()])
        mlflow.log_metric(f"cv_rmse_{param_str}", metric)

    # Evaluate 
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    mlflow.log_metrics({
        "rmse_val_data": rmse,
        "r2_val_data": r2,
        "mae_val_data": mae
    })

    # Show results
    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")
    print(f"MAE: {mae}")

# COMMAND ----------

# Look into feature importance in random forest regression
best_rf_model = rf_cv_model.bestModel
rf_model = best_rf_model.stages[-1]

# Get feature importance from the Random Forest model
feature_importances = rf_model.featureImportances

# Sort features by importance
sorted_features = sorted(zip(features3, feature_importances), key=lambda x: x[1], reverse=True)
features, importances = zip(*sorted_features)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## GBT Regressor

# COMMAND ----------

# GBT model and grid search
gbt = GBTRegressor(featuresCol="features3", labelCol="total_amount", seed=42)
gbt_paramGrid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [3, 5])
    .addGrid(gbt.maxBins, [16, 32])
    .build()
)

# Create a Pipeline (gbt_pipeline)
gbt_pipeline = Pipeline(stages = [assembler3, gbt])

mlflow.pyspark.ml.autolog(disable=True) 

gbt_cv = CrossValidator(
    estimator=gbt_pipeline,
    estimatorParamMaps=gbt_paramGrid,
    evaluator=evaluator_rmse,
    numFolds=3
)

with mlflow.start_run(run_name="GBT_Model"):
    gbt_cv_model = gbt_cv.fit(train_data)
    predictions = gbt_cv_model.transform(val_data)

    # Log all tested parameter combinations
    for params, metric in zip(gbt_cv_model.getEstimatorParamMaps(), gbt_cv_model.avgMetrics):
        param_str = "_".join([f"{param.name}_{val}" for param, val in params.items()])
        mlflow.log_metric(f"cv_rmse_{param_str}", metric)

    # Evaluate 
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    mlflow.log_metrics({
        "rmse_val_data": rmse,
        "r2_val_data": r2,
        "mae_val_data": mae
    })

    # Show results
    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")
    print(f"MAE: {mae}")

# COMMAND ----------

# Look into feature importance in GBT regression
best_gbt_model = gbt_cv_model.bestModel.stages[-1]

# Get feature importances from the GBT model
feature_importances = best_gbt_model.featureImportances

# Sort features by importance
sorted_features = sorted(zip(features3, feature_importances), key=lambda x: x[1], reverse=True)
features, importances = zip(*sorted_features)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("GBT Feature Importance")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Update features
# MAGIC Update features based on feature importances to see if we can improve our model

# COMMAND ----------

features4 = [#'passenger_count',
 'PULocationID',
 'DOLocationID',
 'airport_fee',
 #'is_weekend',
 'hour',
 'mta_tax_indexed',
 'congestion_surcharge_indexed',
 'trip_distance',
 'extra',
 'tolls_amount']

assembler4 = VectorAssembler(inputCols=features4, 
outputCol='features4')

# COMMAND ----------

# RF model
rf = RandomForestRegressor(featuresCol="features4", labelCol="total_amount", seed=42)

# Grid search
rf_paramGrid = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, [20, 50])
    .addGrid(rf.maxDepth, [3, 5])
    .addGrid(rf.maxBins, [16, 32])
    .build()
)

mlflow.pyspark.ml.autolog(disable=True) 

# rf pipeline and cross validator
rf_pipeline = Pipeline(stages = [assembler4, rf])
rf_cv = CrossValidator(
    estimator=rf_pipeline,
    estimatorParamMaps=rf_paramGrid,
    evaluator=evaluator_rmse,
    numFolds=3
)

with mlflow.start_run(run_name="RF_Model_features4"):
    rf_cv_model = rf_cv.fit(train_data)
    predictions = rf_cv_model.transform(val_data)

    # Log all tested parameter combinations
    for params, metric in zip(rf_cv_model.getEstimatorParamMaps(), rf_cv_model.avgMetrics):
        param_str = "_".join([f"{param.name}_{val}" for param, val in params.items()])
        mlflow.log_metric(f"cv_rmse_{param_str}", metric)

    # Evaluate 
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    mlflow.log_metrics({
        "rmse_val_data": rmse,
        "r2_val_data": r2,
        "mae_val_data": mae
    })

    # Show results
    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")
    print(f"MAE: {mae}")

# COMMAND ----------

# Look into feature importance in random forest regression
best_rf_model_features4 = rf_cv_model.bestModel
rf_model = best_rf_model_features4.stages[-1]

# Get feature importance from the Random Forest model
feature_importances = rf_model.featureImportances

# Sort features by importance
sorted_features = sorted(zip(features4, feature_importances), key=lambda x: x[1], reverse=True)
features, importances = zip(*sorted_features)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()

# COMMAND ----------

# GBT model and grid search
gbt = GBTRegressor(featuresCol="features4", labelCol="total_amount", seed=42)
gbt_paramGrid = (
    ParamGridBuilder()
    .addGrid(gbt.maxDepth, [3, 5])
    .addGrid(gbt.maxBins, [16, 32])
    .build()
)

# Create a Pipeline (gbt_pipeline)
gbt_pipeline = Pipeline(stages = [assembler4, gbt])

mlflow.pyspark.ml.autolog(disable=True) 

gbt_cv = CrossValidator(
    estimator=gbt_pipeline,
    estimatorParamMaps=gbt_paramGrid,
    evaluator=evaluator_rmse,
    numFolds=3
)

with mlflow.start_run(run_name="GBT_Model_features4"):
    gbt_cv_model = gbt_cv.fit(train_data)
    predictions = gbt_cv_model.transform(val_data)

    # Log all tested parameter combinations
    for params, metric in zip(gbt_cv_model.getEstimatorParamMaps(), gbt_cv_model.avgMetrics):
        param_str = "_".join([f"{param.name}_{val}" for param, val in params.items()])
        mlflow.log_metric(f"cv_rmse_{param_str}", metric)

    # Evaluate 
    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)
    mae = evaluator_mae.evaluate(predictions)

    mlflow.log_metrics({
        "rmse_val_data": rmse,
        "r2_val_data": r2,
        "mae_val_data": mae
    })

    # Show results
    print(f"RMSE: {rmse}")
    print(f"R^2: {r2}")
    print(f"MAE: {mae}")

# COMMAND ----------

# Look into feature importance in GBT regression
best_gbt_model_features4 = gbt_cv_model.bestModel.stages[-1]

# Get feature importances from the GBT model
feature_importances = best_gbt_model_features4.featureImportances

# Sort features by importance
sorted_features = sorted(zip(features4, feature_importances), key=lambda x: x[1], reverse=True)
features, importances = zip(*sorted_features)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.title("GBT Feature Importance")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results 
# MAGIC Calculate test error and visualize predictions for the final model (GBT Regressor with features 3)

# COMMAND ----------

# Get predictions
test_data_assembled = assembler3.transform(test_data)
predictions = best_gbt_model.transform(test_data_assembled)

# Select actual and predicted values
results = predictions.select("total_amount", "prediction").toPandas()

# COMMAND ----------

rmse = evaluator_rmse.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)

# Show results
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")
print(f"MAE: {mae}")

# COMMAND ----------

# Predicted vs Actual
plt.figure(figsize=(8, 6))
sns.scatterplot(data=results, x="total_amount", y="prediction", alpha=0.5)
plt.plot([results.total_amount.min(), results.total_amount.max()],
         [results.total_amount.min(), results.total_amount.max()],
         color='red', linestyle='--')
plt.xlabel("Actual Total Amount")
plt.ylabel("Predicted Total Amount")
plt.title("GBT Regression: Actual vs Predicted")
plt.grid(True)
plt.show()

# COMMAND ----------

results["residuals"] = results["total_amount"] - results["prediction"]

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(results["prediction"], results["residuals"])
plt.axhline(0, color='red', linestyle="--")  # horizontal line at zero for reference
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("GBT Residuals vs Predicted")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Observations: The scatter plots reveal the presence of some trips for which the model's predictions are less accurate. However, there are no strong or systematic patterns in the residuals. For instance, we notice a gap around the predicted value of 150, but overall, the distribution appears balanced. This suggests that the final model performs consistently across the majority of data points.