import logging
from pyspark.sql.functions import col, when, trim, upper, lit
from pyspark.sql import DataFrame


def process_demographic(spark, path: str) -> DataFrame:
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    logging.info(f"Loaded demographic data with {df.count()} rows and {len(df.columns)} columns")
    
    # Drop Features Marked In Red According to Data Dictionary File
    drop_columns = [
        "annual_inc_joint", "verification_status_joint", "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc", 
        "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts"
    ]
    df = df.drop(*drop_columns)

    # Fill Empty/Missing Values
    df = df.withColumn("emp_title", when(col("emp_title").isNull(), "MISSING")  # Fill NA
                                     .otherwise(trim(upper(col("emp_title")))))   # strip whitespace and convert to uppercase
    df = df.fillna({
        "emp_length": "MISSING",
        "home_ownership": "MISSING"
    })

    logging.info("Demographic processing complete")
    return df