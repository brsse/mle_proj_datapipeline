import logging
from pyspark.sql.functions import col

def process_credit_history(spark, df):
    logging.info("Processing credit history data")
    
    drop_cols = ["earliest_cr_line"]
    df = df.drop(*[c for c in drop_cols if c in df.columns])

    # Confirm all remaining columns are numeric or join keys
    numeric_types = ["double", "int", "bigint"]
    bad_cols = [(c, dtype) for c, dtype in df.dtypes
                if c not in ["member_id", "snapshot_date"] and dtype not in numeric_types]
    if bad_cols:
        logging.info(f"Non-numeric columns found: {bad_cols}")
        print("Warning: Non-numeric columns:", bad_cols)

    logging.info("Finished processing credit history data")
    return df