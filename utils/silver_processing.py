import pandas as pd
import os
import logging
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from utils.silver_credit_history import process_credit_history
from utils.silver_demographic import process_demographic
from utils.silver_financial import process_financial
from utils.silver_loan_terms import process_loan_terms


def silver_processing(data_dir, output_dir, data_window=None):
    spark = SparkSession.builder.appName("SilverProcessing").getOrCreate()

    # === Logging ===
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/silver_processing.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True 
    )

    # === Files and Directories ===
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = {
        "credit_history": process_credit_history,
        "demographic": process_demographic,
        "financial": process_financial,
        "loan_terms": process_loan_terms,
    }

    # === Process Data ===
    for category, processor in categories.items():
        logging.info(f"Processing {category} data")

        df = processor(spark, str(data_dir / f"features_{category}.csv"))

        if data_window is not None:
            df = df.filter(col("snapshot_date").cast("date").isin(data_window))

        category_output_dir = output_dir / category
        category_output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Writing {category} data...")
        for row in df.select("snapshot_date").distinct().collect():
            snapshot_date = row["snapshot_date"]
            date_str = snapshot_date.strftime("%Y-%m-%d")

            daily_df = df.filter(col("snapshot_date") == snapshot_date).orderBy("member_id", "snapshot_date")

            output_path = category_output_dir / f"{category}_{date_str}"
            daily_df.write.mode("overwrite").parquet(str(output_path))

        logging.info(f"Finished writing {category} data to {category_output_dir}")

    logging.info("Silver-level processing complete.")
    print("Silver-level processing complete.")