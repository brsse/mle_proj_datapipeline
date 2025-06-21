import logging
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

def process_financial(spark, df):
    logging.info("Processing financial data")

    logging.info("Casting data to double type...")
    for c in df.columns:
        if c not in ["member_id", "snapshot_date"] and df.schema[c].dataType.simpleString() not in ["double"]:
            df = df.withColumn(c, col(c).cast(DoubleType()))
            
    logging.info("Finished processing financial data")
    return df