from pyspark.sql import SparkSession

# Configure Spark session
spark = SparkSession.builder \
    .appName("MovieLensRecommender") \
    .master("local[*]") \
    .getOrCreate()

#Movie Ratings Data
ratings_df = spark.read.csv(
    "C:/Users/kayla/StreamRec/ml-latest-small/ratings.csv",
    header=True,
    inferSchema=True
)

#Rename Columns
ratings_df = ratings_df \
    .withColumnRenamed("userId", "user_id") \
    .withColumnRenamed("movieId", "movie_id")

#Create the dataframe

#Check the number of partitions for this dataframe

#Inspect the Data
ratings_df.show(5)
ratings_df.printSchema()

#Feature Aggregation
ratings_df.createOrReplaceTempView("ratings")

ratings_agg = spark.sql("""
    SELECT user_id, movie_id, COUNT(*) as interactions
    FROM ratings
    GROUP BY user_id, movie_id
""")

#Convert to Parquet
ratings_agg.write.parquet("ratings_parquet")


