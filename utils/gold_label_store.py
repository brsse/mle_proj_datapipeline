import logging
import re
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_date, weekofyear, month, date_sub, dayofweek, lit, concat_ws
from datetime import datetime
import os


def create_gold_label_store(input_dir, output_dir, snapshot_date_str):
    spark = SparkSession.builder.appName("GoldLabelStore").getOrCreate()

    # === Logging ===
    Path("/opt/airflow/logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="/opt/airflow/logs/gold_label_store.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )

    logging.info(f"Creating Gold Label Store from {input_dir}...")

    # Parse the snapshot date and format for file path
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    date_str = snapshot_date.strftime('%Y_%m_%d')
    
    # === Load Loan Terms Data ===
    logging.info("Loading silver-level loan terms parquet folder")
    loan_terms_path = f"{input_dir}/loan_terms/silver_loan_terms_week_{date_str}"
    
    try:
        df = spark.read.parquet(loan_terms_path)
        logging.info(f"Successfully loaded loan terms data from: {loan_terms_path}")
    except Exception as e:
        logging.error(f"Failed to load loan terms data from {loan_terms_path}: {str(e)}")
        return

    # === Select Columns For Label Store ===
    selected_columns = ["id", "snapshot_date", "grade"]
    df = df.select(*selected_columns).filter("grade IS NOT NULL")
    df = df.withColumn("snapshot_date", to_date("snapshot_date"))

    # === Write Label Store to Parquet ===
    # Create output directory
    gold_table_dir = os.path.join(output_dir, "label_store")
    os.makedirs(gold_table_dir, exist_ok=True)
    
    # Write processed data to gold layer
    partition_name = f"label_store_week_{date_str}"
    output_path = os.path.join(gold_table_dir, partition_name)
    
    logging.info(f"Writing gold label store to: {output_path}")
    df.write.mode("overwrite").parquet(output_path)

    logging.info("Gold label store created successfully.")
    print(f"Gold label store written to: {output_path}")