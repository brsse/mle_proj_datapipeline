import logging
from pyspark.sql.functions import col, when
from pyspark.sql import DataFrame


def process_loan_terms(spark, path: str) -> DataFrame:
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    logging.info(f"Loaded loan terms data with {df.count()} rows and {len(df.columns)} columns")

    # Drop Features Marked In Red According to Data Dictionary File
    drop_columns = [
        "url", "desc", "title", "hardship_flag", "hardship_type", "hardship_reason", "hardship_status", "deferral_term",
        "hardship_amount", "hardship_start_date", "hardship_end_date", "payment_plan_start_date", "hardship_length",
        "hardship_dpd", "hardship_loan_status", "orig_projected_additional_accrued_interest", "hardship_payoff_balance_amount",
        "hardship_last_payment_amount", "debt_settlement_flag_date", "settlement_status", "settlement_date",
        "settlement_amount", "settlement_percentage", "settlement_term", "out_prncp", "out_prncp_inv", "total_pymnt", 
        "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", 
        "last_pymnt_d", "next_pymnt_d", "last_pymnt_amnt", "policy_code"
    ]
    df = df.drop(*drop_columns)

    # Convert binary string values to 0/1
    df = df.withColumn("pymnt_plan", when(col("pymnt_plan") == "y", 1).otherwise(0))
    df = df.withColumn("debt_settlement_flag", when(col("debt_settlement_flag") == "Y", 1).otherwise(0))
    df = df.withColumn("initial_list_status", when(col("initial_list_status") == "w", 1).otherwise(0))
    df = df.withColumn("disbursement_method", when(col("disbursement_method") == "Cash", 1).otherwise(0))
    
    logging.info("Loan terms processing complete")
    return df