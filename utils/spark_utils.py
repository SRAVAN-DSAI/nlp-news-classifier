from pyspark.sql import SparkSession

def create_spark_session(app_name="BigDataNLPProject", master="local[*]", executor_memory="2g", driver_memory="2g"):
    """
    Creates and returns a SparkSession.
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.executor.memory", executor_memory)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark