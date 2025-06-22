import logging
import re
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import to_date, col, months_between, trunc, udf
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder


# def load_category_by_date(base_path, data_window, spark):
#     path = Path(base_path)
#     pattern = re.compile(r".*_(\d{4}-\d{2}-\d{2})")
#     folders = [f for f in path.iterdir() if f.is_dir() and pattern.match(f.name)]
#     if data_window:
#         date_strs = [d if isinstance(d, str) else d.strftime("%Y-%m-%d") for d in data_window]
#         folders = [f for f in folders if pattern.match(f.name).group(1) in date_strs]
#     if not folders:
#         return None
#     return spark.read.parquet(*[str(f) for f in folders])

# type conversion
def enforce_schema(df, schema_dict):
    for col_name, dtype in schema_dict.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast(dtype))
    return df

def credit_history_processing(df, spark):
    # type conversion
    schema = {
        "member_id": StringType(),
        "snapshot_date": DateType(),
        "mort_acc": IntegerType(),
        "num_tl_op_past_12m": IntegerType(),
        "inq_last_6mths": IntegerType(),
        "inq_last_12m": IntegerType(),
        "inq_fi": IntegerType(),
        "mths_since_last_delinq": IntegerType(),
        "mths_since_recent_inq": IntegerType(),
        "mths_since_rcnt_il": IntegerType(),
        "mths_since_recent_bc": IntegerType(),
        "acc_now_delinq": IntegerType(),
        "delinq_2yrs": IntegerType(),
        "pub_rec": IntegerType(),
        "collections_12_mths_ex_med": IntegerType(),
        "chargeoff_within_12_mths": IntegerType(),
        "tax_liens": IntegerType(),
        "pub_rec_bankruptcies": IntegerType(),
        "num_tl_120dpd_2m": IntegerType(),
        "num_tl_30dpd": IntegerType(),
        "num_tl_90g_dpd_24m": IntegerType(),
        "num_accts_ever_120_pd": IntegerType(),
        "delinq_amnt": IntegerType(),
        "mort_acc_missing": IntegerType(),
    }
    df = enforce_schema(df, schema)
    df = df.withColumn("earliest_cr_line", to_date("earliest_cr_line", "MMM-yyyy"))

    # get difference between snapshot_date and earliest_cr_line
    df = df.withColumn("earliest_cr_date", to_date(col("earliest_cr_line"), "MMM-yyyy"))

    df = df.withColumn("snapshot_month", trunc("snapshot_date", "MM"))
    df = df.withColumn("earliest_cr_month", trunc("earliest_cr_date", "MM"))

    df = df.withColumn("months_since_earliest_cr", months_between("snapshot_month", "earliest_cr_month").cast("int"))

    # drop unnecessary columns
    df = df.drop("earliest_cr_line")

    return df

def demographic_processing(df, spark):
    # type conversion
    schema = {
        "member_id": StringType(),
        "snapshot_date": DateType(),
        "emp_title": StringType(),
        "emp_length": StringType(),
        "home_ownership": StringType(),
        "annual_inc": FloatType(),
        "verification_status": StringType(),
        "zip_code": StringType(),
        "addr_state": StringType(),
        "application_type": StringType(),
    }
    df = enforce_schema(df, schema)

    # get top 10 appearing emp_title, and do one-hot encoding
    # clean 'emp_title' and limit to top 10 most frequent values
    top_10_titles = (
        df.groupBy("emp_title")
        .count()
        .orderBy(F.desc("count"))
        .limit(10)
        .rdd.map(lambda row: row["emp_title"])
        .collect()
    )
    # replace nulls and normalize
    df = df.withColumn("emp_title", F.upper(F.trim(F.coalesce("emp_title", F.lit("MISSING")))))
    # create new column with only top 10 or 'OTHER'
    df = df.withColumn(
        "emp_title_limited",
        F.when(F.col("emp_title").isin(top_10_titles), F.col("emp_title")).otherwise("OTHER")
    )
    # StringIndex and OneHotEncode
    indexer = StringIndexer(inputCol="emp_title_limited", outputCol="emp_title_index", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol="emp_title_index", outputCol="emp_title_ohe")
    pipeline = Pipeline(stages=[indexer, encoder])
    df = pipeline.fit(df).transform(df)
    # convert one-hot vector to array
    df = df.withColumn("emp_title_ohe_array", vector_to_array("emp_title_ohe"))
    # expand to separate binary columns
    num_categories = len(top_10_titles) + 1
    for i in range(num_categories):
        df = df.withColumn(f"emp_title_ohe_{i}", F.col("emp_title_ohe_array")[i])
    # drop intermediate columns
    df = df.drop("emp_title_limited", "emp_title_index", "emp_title_ohe", "emp_title_ohe_array")

    # format emp_length
    def parse_emp_length(val):
        if val is None:
            return None
        val = val.strip().lower()
        if val == "10+ years":
            return 10
        elif val == "< 1 year":
            return 0
        elif val == "missing":
            return -1
        elif "year" in val:
            try:
                return int(val.split()[0])
            except:
                return -1
        else:
            return -1
    parse_emp_length_udf = udf(parse_emp_length, IntegerType())
    df = df.withColumn("emp_length", parse_emp_length_udf(col("emp_length")))

    # one-hot encode categorical columns
    # example categorical columns
    categorical_cols = ["home_ownership", "verification_status", "addr_state", "application_type"]
    index_output_cols = [f"{col}_index" for col in categorical_cols]
    ohe_output_cols = [f"{col}_ohe" for col in categorical_cols]
    # stringIndexers
    indexers = [
        StringIndexer(inputCol=c, outputCol=idx, handleInvalid="keep")
        for c, idx in zip(categorical_cols, index_output_cols)
    ]
    # oneHotEncoders
    encoders = [
        OneHotEncoder(inputCol=idx, outputCol=ohe)
        for idx, ohe in zip(index_output_cols, ohe_output_cols)
    ]
    # combine all stages into one pipeline
    pipeline = Pipeline(stages=indexers + encoders)
    model = pipeline.fit(df)
    df = model.transform(df)
    # optionally convert OHE vectors to individual binary columns
    for ohe_col in ohe_output_cols:
        array_col = f"{ohe_col}_array"
        df = df.withColumn(array_col, vector_to_array(col(ohe_col)))
        num_categories = df.select(ohe_col).first()[0].size
        for i in range(num_categories):
            df = df.withColumn(f"{ohe_col}_{i}", col(array_col)[i])
        df = df.drop(ohe_col, array_col)

    # drop unnecessary columns
    df = df.drop("earliest_cr_line", "emp_title", "zip_code", "home_ownership", "verification_status", "addr_state", "application_type")

    return df

def financial_processing(df, spark):
    # type conversion
    schema = {
        "member_id": StringType(),
        "snapshot_date": DateType(),
        "dti": FloatType(),
        "revol_bal": FloatType(),
        "revol_util": FloatType(),
        "total_rev_hi_lim": FloatType(),
        "tot_coll_amt": FloatType(),
        "tot_cur_bal": FloatType(),
        "avg_cur_bal": FloatType(),
        "all_util": FloatType(),
        "max_bal_bc": FloatType(),
        "il_util": FloatType(),
        "bc_util": FloatType(),
        "total_bal_il": FloatType(),
        "total_bal_ex_mort": FloatType(),
        "total_bc_limit": FloatType(),
        "total_il_high_credit_limit": FloatType(),
        "tot_hi_cred_lim": FloatType(),
        "open_acc": IntegerType(),
        "total_acc": IntegerType(),
        "open_acc_6m": IntegerType(),
        "open_act_il": IntegerType(),
        "open_il_12m": IntegerType(),
        "open_il_24m": IntegerType(),
        "open_rv_12m": IntegerType(),
        "open_rv_24m": IntegerType(),
        "acc_open_past_24mths": IntegerType(),
        "mo_sin_old_il_acct": IntegerType(),
        "mo_sin_old_rev_tl_op": IntegerType(),
        "mo_sin_rcnt_rev_tl_op": IntegerType(),
        "mo_sin_rcnt_tl": IntegerType(),
        "num_actv_bc_tl": IntegerType(),
        "num_actv_rev_tl": IntegerType(),
        "num_rev_accts": IntegerType(),
        "num_rev_tl_bal_gt_0": IntegerType(),
        "num_il_tl": IntegerType(),
        "num_bc_tl": IntegerType(),
        "num_op_rev_tl": IntegerType(),
        "num_sats": IntegerType(),
        "num_bc_sats": IntegerType(),
        "total_cu_tl": IntegerType(),
        "bc_open_to_buy": FloatType(),
        "percent_bc_gt_75": FloatType(),
        "pct_tl_nvr_dlq": FloatType(),
        "all_util_missing": IntegerType(),
        "il_util_missing": IntegerType(),
        "bc_util_missing": IntegerType()
    }
    df = enforce_schema(df, schema)

    return df

def loan_terms_processing(df, spark):
    # type conversion
    schema = {
        "id": StringType(),
        "member_id": StringType(),
        "snapshot_date": DateType(),
        "loan_amnt": FloatType(),
        "funded_amnt": FloatType(),
        "funded_amnt_inv": FloatType(),
        "term": StringType(),
        "int_rate": FloatType(),
        "installment": FloatType(),
        "grade": StringType(),
        "sub_grade": StringType(),
        "issue_d": DateType(),
        "loan_status": StringType(),
        "pymnt_plan": StringType(),
        "purpose": StringType(),
        "initial_list_status": StringType(),
        "disbursement_method": StringType(),
        "debt_settlement_flag": StringType()
    }
    df = enforce_schema(df, schema)

    # one-hot encoding for categorical column
    categorical_cols = ["term", "loan_status", "pymnt_plan", "purpose", "initial_list_status", "disbursement_method", "debt_settlement_flag"]
    # create indexers and encoders
    indexers = [StringIndexer(inputCol=col_name, outputCol=f"{col_name}_idx", handleInvalid="keep")
                for col_name in categorical_cols]
    encoders = [OneHotEncoder(inputCol=f"{col_name}_idx", outputCol=f"{col_name}_ohe", dropLast=False)
                for col_name in categorical_cols]
    # pipeline
    pipeline = Pipeline(stages=indexers + encoders)
    df = pipeline.fit(df).transform(df)
    # convert each OHE vector column into separate binary columns
    for col_name in categorical_cols:
        ohe_col = f"{col_name}_ohe"
        arr_col = f"{col_name}_arr"
        df = df.withColumn(arr_col, vector_to_array(col(ohe_col)))
        # get number of categories
        vector_size = df.select(arr_col).head()[arr_col].__len__()
        # create new columns for each category value
        for i in range(vector_size):
            df = df.withColumn(f"{col_name}_{i}", col(arr_col)[i])
        # drop intermediate columns
        df = df.drop(col_name, f"{col_name}_idx", f"{col_name}_ohe", arr_col)
    
    # Keep grade column for filtering, drop other unnecessary columns
    df = df.drop("issue_d", "sub_grade", "term", "loan_status", "pymnt_plan", "purpose", "initial_list_status", "disbursement_method", "debt_settlement_flag")

    return df

def create_feature_store(
    silver_root: str,
    output_path: str,
    snapshot_date_str: str
):
    spark = SparkSession.builder.appName("FeatureStoreCreation").getOrCreate()

    Path("/opt/airflow/logs").mkdir(exist_ok=True)
    logging.basicConfig(
        filename="/opt/airflow/logs/gold_feature_store.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_str = snapshot_date.strftime('%Y_%m_%d')
    # === Load Each Feature Category ===
    logging.info("Loading silver-level parquet folders")
    loan_df        = spark.read.parquet(f"{silver_root}/loan_terms/silver_loan_terms_week_{date_str}")
    demo_df        = spark.read.parquet(f"{silver_root}/demographic/silver_demographic_week_{date_str}")
    fin_df         = spark.read.parquet(f"{silver_root}/financial/silver_financial_week_{date_str}")
    credit_df      = spark.read.parquet(f"{silver_root}/credit_history/silver_credit_history_week_{date_str}")

    if not all([loan_df, demo_df, fin_df, credit_df]):
        logging.warning("One or more silver categories could not be loaded.")
        return
    
    # === Feature engineering ===
    demo_df = demographic_processing(demo_df, spark)
    credit_df = credit_history_processing(credit_df, spark)
    fin_df = financial_processing(fin_df, spark)
    loan_df = loan_terms_processing(loan_df, spark)

    # === Join all on member_id and snapshot_date ===
    logging.info("Joining all dataframes on member_id and snapshot_date")
    # Start with loan_df and use left joins to ensure we only include members with loan terms (and grades)
    # but allow missing data in other tables
    df = loan_df.join(demo_df, ["member_id", "snapshot_date"], "left") \
                .join(fin_df, ["member_id", "snapshot_date"], "left") \
                .join(credit_df, ["member_id", "snapshot_date"], "left")

    # Drop duplicates that may have been introduced by the joins
    df = df.distinct()
    
    # === Apply same filtering as label store for consistency ===
    logging.info("Applying grade filter to ensure consistency with label store")
    df = df.filter("grade IS NOT NULL")
    
    # Drop grade column after filtering, along with member_id and snapshot_date
    df = df.drop("member_id", "snapshot_date", "grade")
    print(df.columns)
    
    # === Write Feature Store to Parquet ===
    # Create output directory
    gold_table_dir = os.path.join(output_path, "feature_store")
    os.makedirs(gold_table_dir, exist_ok=True)
    
    # Write processed data to gold layer
    partition_name = f"feature_store_week_{date_str}"
    output_path = os.path.join(gold_table_dir, partition_name)
    
    logging.info(f"Writing gold feature store to: {output_path}")
    df.write.mode("overwrite").parquet(output_path)

    logging.info("Gold feature store created successfully.")
    print(f"Gold feature store written to: {output_path}")

    