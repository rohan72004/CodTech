from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when, isnan
spark = SparkSession.builder.appName("Netflix Data Analysis").getOrCreate()
df = spark.read.csv("data/your_dataset.csv", header=True, inferSchema=True)
print("Initial Data Preview:")
df.show(5)
df_clean = df.withColumn(
    "release_year",
    when(
        col("release_year").cast("string").rlike("^[0-9]{4}$"),
        col("release_year").cast("int")
    ).otherwise(None)
)
df_clean = df_clean.na.drop(subset=["release_year"])
df_clean = df_clean.fillna({"director": "Unknown", "country": "Unknown", "rating": "Not Rated"})
print("Cleaned Data Preview:")
df_clean.show(5)
print("Count by Type:")
df_clean.groupBy("type").count().show()
print("Top Countries by Number of Shows:")
df_clean.groupBy("country").count().orderBy("count", ascending=False).show(10)
print("Shows Released per Year:")
df_clean.groupBy("release_year").count().orderBy("release_year").show(20)
from pyspark.sql.functions import regexp_extract
df_movies = df_clean.filter(col("type") == "Movie")
df_movies = df_movies.withColumn(
    "duration_min",
    regexp_extract(col("duration"), r"(\d+)", 1).cast("int")
)
print("Average duration of Movies (minutes):")
df_movies.agg({"duration_min": "avg"}).show()
spark.stop()
