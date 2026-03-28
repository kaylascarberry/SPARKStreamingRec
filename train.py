from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Create Spark session
spark = SparkSession.builder \
    .appName("MovieLensALS") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

# Load your processed data (IMPORTANT: must exist)
ratings_agg = spark.read.parquet("ratings_parquet")

# Split the data into training and test sets
train, test = ratings_agg.randomSplit([0.8, 0.2], seed=42)

# Train ALS model
als = ALS(
    userCol="user_id",
    itemCol="movie_id",
    ratingCol="interactions",
    coldStartStrategy="drop"
)

# Train model on training set
model = als.fit(train)
print("Model trained successfully!")

#Generate predictions on test set
predictions = model.transform(test)

#Evaluate the model
evaluator = RegressionEvaluator(
    metricName="rmse", 
    labelCol="interactions", 
    predictionCol="prediction"
    )

rmse = evaluator.evaluate(predictions)
print("RMSE = " + str(rmse))

# Generate user recommendations
user_recs = model.recommendForAllUsers(10)
user_recs.show(5, truncate=False)

input("Press ENTER to stop Spark session.")




