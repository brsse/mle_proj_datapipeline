import logging
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from utils.gold_credit_history import process_credit_history
from utils.gold_demographic import process_demographic
from utils.gold_financial import process_financial
from utils.gold_loan_terms import process_loan_terms


def gold_processing(data_dir, output_dir, data_window=None):
    spark = SparkSession.builder.appName("GoldProcessing").getOrCreate()

    # === Logging ===
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="logs/gold_processing.log",
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
        silver_category_dir = data_dir / category
        gold_category_dir = output_dir / category
        gold_category_dir.mkdir(parents=True, exist_ok=True)

        date_folders = sorted([f for f in silver_category_dir.iterdir() if f.is_dir()])

        # Filter for data_window (if any)
        if data_window is not None:
            data_window = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in data_window]
            date_folders = [f for f in date_folders if f.name.split("_")[-1] in data_window]

        # Read all daily folders into one DataFrame
        if date_folders:
            df = spark.read.parquet(*[str(f) for f in date_folders])
            logging.info(f"Loaded {df.count()} rows and {len(df.columns)} columns for {category}")
        else:
            logging.warning(f"No folders found for {category} in {silver_category_dir}")
            continue

        df = processor(spark, df)

        logging.info(f"Writing {category} data...")
        for row in df.select("snapshot_date").distinct().collect():
            snapshot_date = row["snapshot_date"]
            try:
                date_str = snapshot_date.to_pydatetime().date().strftime("%Y-%m-%d")
            except Exception:
                date_str = str(snapshot_date)[:10]

            daily_df = df.filter(col("snapshot_date") == snapshot_date).orderBy("member_id", "snapshot_date")
            output_path = gold_category_dir / f"{category}_{date_str}"
            daily_df.write.mode("overwrite").parquet(str(output_path))

        logging.info(f"Finished writing {category} data for {date_str} to {output_path}")

    logging.info("Gold-level processing complete")
    print("Gold-level processing complete.")