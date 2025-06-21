import logging
from pyspark.sql.functions import col, when, lit, to_date, months_between
from pyspark.sql import DataFrame


def process_credit_history(spark, path: str) -> DataFrame:
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    logging.info(f"Loaded credit history data with {df.count()} rows and {len(df.columns)} columns")

    # Drop Features Marked In Red According to Data Dictionary File
    drop_columns = [
        "last_credit_pull_d", "mths_since_last_record", "mths_since_last_major_derog", "mths_since_recent_bc_dlq", 
        "mths_since_recent_revol_delinq", "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med", 
        "sec_app_mths_since_last_major_derog"
    ]
    df = df.drop(*drop_columns)
    
    # Add Missing Flags
    df = df.withColumn("mort_acc_missing", when(col("mort_acc").isNull(), lit(1)).otherwise(lit(0)))

    # Handle Format And Missing Values for earliest_cr_line
    df = df.withColumn("earliest_cr_line", to_date(col("earliest_cr_line"), "MMM-yyyy"))
    df = df.withColumn(
        "months_since_earliest_cr_line",
        months_between(col("snapshot_date"), col("earliest_cr_line"))
    )
    df = df.fillna({"months_since_earliest_cr_line": 999})
    
    
    # Fill NA for these features with mode
    fill_mode = [
        "inq_last_6mths", "acc_now_delinq", "delinq_2yrs", "pub_rec", "collections_12_mths_ex_med",
        "chargeoff_within_12_mths", "tax_liens", "pub_rec_bankruptcies", "delinq_amnt"
    ]
    for col_name in fill_mode:
        mode_val = df.groupBy(col_name).count().orderBy("count", ascending=False).first()[0]
        df = df.fillna({col_name: mode_val})

    # Fill NA for these features with -1
    fill_neg1 = [
        "inq_last_12m", "num_tl_op_past_12m", "inq_fi", "mths_since_last_delinq", "mths_since_recent_inq", "mths_since_rcnt_il", 
        "mths_since_recent_bc", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_accts_ever_120_pd"
    ]
    df = df.fillna({col: -1 for col in fill_neg1})

        # Fill NA for these features with 0
    fill_zero = [
        "mort_acc"]
    df = df.fillna({col: 0 for col in fill_zero})

    logging.info("Credit history processing complete")
    return df