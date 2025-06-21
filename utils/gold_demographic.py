import logging
from pyspark.sql.functions import col, when
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

def process_demographic(spark, df):
    logging.info("Processing demographic data")

    # Drop columns that are too noisy or not useful
    drop_cols = ["emp_title", "emp_length", "addr_state", "zip_code"]
    df = df.drop(*[c for c in drop_cols if c in df.columns])

    # Columns to OHE
    cols_to_ohe = [
        "home_ownership",
        "verification_status",
        "application_type"
    ]
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cols_to_ohe]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in cols_to_ohe]

    pipeline = Pipeline(stages=indexers + encoders)
    logging.info("One-hot-encoding categorical data...")
    df = pipeline.fit(df).transform(df)

    # Drop original and indexed columns, keep OHE columns
    cols_to_remove = cols_to_ohe + [f"{c}_idx" for c in cols_to_ohe]
    df = df.drop(*cols_to_remove)
    
    logging.info("Finished processing demographic data")
    return df
