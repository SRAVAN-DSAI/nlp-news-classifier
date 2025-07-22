

import os
import re
from datetime import datetime

# Spark Imports
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType

# Local utility import
import sys
# Add parent directory (nlp_news_pipeline) to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.spark_utils import create_spark_session

def load_raw_data_into_spark(spark, input_filepath: str):
    """
    Loads raw JSON Lines data into a Spark DataFrame.
    """
    schema = StructType([
        StructField("id", LongType(), True),
        StructField("timestamp", StringType(), True),
        StructField("text", StringType(), True),
        StructField("source", StringType(), True),
        StructField("category", StringType(), True)
    ])

    raw_df = (
        spark.read.format("json")
        .schema(schema)
        .load(input_filepath)
    )
    return raw_df

def clean_and_prepare_data(raw_df):
    """
    Cleans and prepares the raw DataFrame.
    - Converts timestamp string to TimestampType using a Python UDF.
    - Cleans the text column (lowercase, removes special chars, extra spaces).
    """
    @F.udf(TimestampType())
    def parse_timestamp_udf(ts_str):
        if ts_str is None:
            return None
        try:
            return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    cleaned_df = raw_df.withColumn(
        "timestamp",
        parse_timestamp_udf(F.col("timestamp"))
    )

    @F.udf(StringType())
    def clean_text_udf(text):
        if text is None:
            return None
        text = text.lower()
        text = re.sub(r'http\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    cleaned_df = cleaned_df.withColumn("cleaned_text", clean_text_udf(F.col("text")))

    final_df = cleaned_df.select("id", "timestamp", "source", "category", "text", "cleaned_text")

    return final_df

def write_dataframe_to_parquet(df, output_path, partition_by=None):
    """
    Writes a Spark DataFrame to Parquet files.
    """
    print(f"\nAttempting to write data to Parquet files at: {output_path}")
    writer = df.write.mode("overwrite")

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.parquet(output_path)
    print(f"Successfully wrote data to Parquet files at: {output_path}")

if __name__ == "__main__":
    print("\n--- Running Data Processing Script ---")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_data_path = os.path.join(project_root, "data/raw_news/ag_news_train.jsonl")
    output_cleaned_parquet_path = os.path.join(project_root, "data/cleaned_news_parquet")
    cleaned_parquet_partition_cols = ["source", "category"]

    if not os.path.exists(input_data_path):
        print(f"Error: Input data file not found at '{input_data_path}'.")
        print("Please run data_ingestion.py first to generate the data.")
    else:
        spark = create_spark_session(app_name="NewsDataProcessor", executor_memory="3g", driver_memory="3g")
        print("SparkSession created successfully.")

        raw_dataframe = load_raw_data_into_spark(spark, input_data_path)
        print(f"Raw data loaded into Spark DataFrame. Total records: {raw_dataframe.count()}")

        cleaned_dataframe = clean_and_prepare_data(raw_dataframe)
        print("\nCleaned data DataFrame created.")

        print("\nCleaned DataFrame Schema:")
        cleaned_dataframe.printSchema()
        print("\nFirst 5 rows of the Cleaned DataFrame:")
        cleaned_dataframe.show(5, truncate=False)
        print("\nTimestamp NULL check (should be 0 if parsing works):")
        num_null_timestamps = cleaned_dataframe.filter(F.col("timestamp").isNull()).count()
        print(f"Number of NULL timestamps after parsing: {num_null_timestamps}")

        print(f"\n--- Writing ALL CLEANED data to Parquet at: {output_cleaned_parquet_path} ---")
        write_dataframe_to_parquet(cleaned_dataframe, output_cleaned_parquet_path, cleaned_parquet_partition_cols)
        print("Successfully wrote cleaned data to Parquet.")

        spark.stop()
        print("SparkSession stopped.")