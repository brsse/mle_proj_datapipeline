import logging
from pyspark.sql.functions import col, when, lit
from pyspark.sql import DataFrame


def process_financial(spark, path: str) -> DataFrame:
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)
    logging.info(f"Loaded financial data with {df.count()} rows and {len(df.columns)} columns")

    # Drop Features Marked In Red According to Data Dictionary File
    drop_columns = ["dti_joint", "revol_bal_joint"]
    df = df.drop(*drop_columns)
    
    # Add Missing Flags
    missing_flag_columns = ["all_util", "il_util", "bc_util"]
    for col_name in missing_flag_columns:
        df = df.withColumn(f"{col_name}_missing", when(col(col_name).isNull(), lit(1)).otherwise(lit(0)))

    # Fill NA for these features with mean
    mean_val = df.selectExpr("avg(dti) as mean_dti").first()["mean_dti"]
    df = df.fillna({"dti": mean_val})

    # Fill NA for these features with 0
    fill_zero = [
        "revol_util", "total_rev_hi_lim", "tot_coll_amt", "tot_cur_bal", "avg_cur_bal", "all_util", "max_bal_bc", 
        "open_acc", "total_acc", "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "open_rv_12m", "open_rv_24m", 
        "acc_open_past_24mths", "num_actv_bc_tl", "num_actv_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_il_tl", 
        "num_bc_tl", "num_op_rev_tl", "num_sats", "num_bc_sats", "total_cu_tl"
    ]
    df = df.fillna({col_name: 0 for col_name in fill_zero})

    # Fill NA for these features with -1
    fill_neg1 = [
        "il_util", "bc_util", "total_bal_il", "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit", 
        "tot_hi_cred_lim", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", 
        "bc_open_to_buy", "percent_bc_gt_75", "pct_tl_nvr_dlq"
    ]
    df = df.fillna({col_name: -1 for col_name in fill_neg1})

    logging.info("Financial processing complete")
    return df