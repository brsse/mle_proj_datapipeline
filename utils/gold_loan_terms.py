import logging
from pyspark.sql.functions import col, when, regexp_extract
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.types import DoubleType

def process_loan_terms(spark, df):
    logging.info("Processing loan terms data")

    # Drop columns that aren't useful
    df = df.drop("issue_d", "sub_grade")

    # Convert term data into int
    df = df.withColumn("term", regexp_extract(col("term"), r"(\d+)", 1).cast("int"))

    # Process grade into numerical labels
    logging.info("Converting alphabetical grade to numerical values...")
    df = df.withColumn("grade", (
        when(col("grade") == "A", 0)
        .when(col("grade") == "B", 1)
        .when(col("grade") == "C", 2)
        .when(col("grade") == "D", 3)
        .when(col("grade") == "E", 4)
        .when(col("grade") == "F", 5)
        .when(col("grade") == "G", 6)
    ))

    # 3. Parse or encode categorical fields
    logging.info("One-hot-encoding categorical columns...")
    categorical_cols = ["purpose", "loan_status"]
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in categorical_cols]

    pipeline = Pipeline(stages=indexers + encoders)
    df = pipeline.fit(df).transform(df)

    # Drop original and indexed versions
    df = df.drop(*categorical_cols)
    df = df.drop(*[f"{c}_idx" for c in categorical_cols])

    logging.info("Finished processing loan terms data")
    return df
