import logging
import re
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_date, weekofyear, month, date_sub, dayofweek, lit, concat_ws


def create_gold_label_store(input_dir, output_dir, data_window=None):
    spark = SparkSession.builder.appName("GoldLabelStore").getOrCreate()

    # === Logging ===
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/gold_label_store.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True
    )

    logging.info(f"Creating Gold Label Store from {input_dir}...")

    input_dir = Path(input_dir) / "loan_terms"
    pattern = re.compile(r"loan_terms_(\d{4}-\d{2}-\d{2})")
    
    all_folders = [f for f in input_dir.iterdir() if f.is_dir() and pattern.match(f.name)]
    for f in input_dir.iterdir():
        logging.info(f"Checking folder: {f.name}")
        if pattern.match(f.name):
            logging.info(f"Match: {f.name}")

    # === Filter Dates For Data ===
    if data_window:
        date_list = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in data_window]
        logging.info(f"Filtering for snapshot_date in dates list...")
        matched_folders = [
            str(f) for f in all_folders if pattern.match(f.name).group(1) in date_list
        ]
    else:
        logging.info("No date filter provided; reading all folders")
        matched_folders = [str(f) for f in all_folders]

    if not matched_folders:
        logging.warning("No matching folders found. Exiting.")
        print("No matching folders found. Exiting.")
        return

    logging.info(f"Processing data from matched folders")
    df = spark.read.parquet(*matched_folders)

    # === Select Columns For Label Store ===
    selected_columns = ["id", "snapshot_date", "grade"]
    df = df.select(*selected_columns).filter("grade IS NOT NULL")
    df = df.withColumn("snapshot_date", to_date("snapshot_date"))

    # === Write Label Store to Parquet ===
    output_path = f"{output_dir}/label_store"
    logging.info(f"Writing gold label store to: {output_path}")
    df.write.mode("overwrite").parquet(str(output_path))

    logging.info("Gold label store created successfully.")
    print(f"Gold label store written to: {output_path}")