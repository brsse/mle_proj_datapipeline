import os
import glob
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Import gold processing functions
from gold_feature_store import create_feature_store
from gold_label_store import create_gold_label_store


def process_gold_table(snapshot_date_str, silver_directory, gold_directory, table_name, spark=None):
    """
    Process gold table data for a specific date and table by reading from silver datamart
    
    Args:
        snapshot_date_str (str): Date string in YYYY-MM-DD format
        silver_directory (str): Directory containing silver data
        gold_directory (str): Directory to save gold data
        table_name (str): Name of the table to process (feature_store, label_store)
        spark: Spark session (optional, will create one if not provided)
    """
    try:
        # Initialize Spark session if not provided
        if spark is None:
            spark = SparkSession.builder.appName(f"GoldProcessing_{table_name}").getOrCreate()
            should_stop_spark = True
        else:
            should_stop_spark = False
        
        # Setup logging
        log_dir = Path("/opt/airflow/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_dir / "gold_processing.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True 
        )
        
        # Parse the snapshot date
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        
        # Validate table name
        supported_tables = ["feature_store", "label_store"]
        if table_name not in supported_tables:
            raise ValueError(f"Unsupported table name: {table_name}. Supported tables: {supported_tables}")
        
        # Process feature_store
        if table_name == "feature_store":
            logging.info(f"Creating gold feature store for week starting {snapshot_date_str}")
            
            # Create output directory
            gold_output_dir = os.path.join(gold_directory, "feature_store")
            os.makedirs(gold_output_dir, exist_ok=True)
            
            # Call the feature store creation function
            create_feature_store(
                silver_root=silver_directory,
                output_path=gold_directory,
                snapshot_date_str=snapshot_date_str
            )
            
            output_path = os.path.join(gold_output_dir, "feature_store")
            logging.info(f"Successfully created gold feature store: {output_path}")
            print(f"Gold feature store created: {output_path}")
            return output_path
            
        # Process label_store
        elif table_name == "label_store":
            logging.info(f"Creating gold label store for week starting {snapshot_date_str}")
            
            # Create output directory
            gold_output_dir = os.path.join(gold_directory, "label_store")
            os.makedirs(gold_output_dir, exist_ok=True)
            
            # Call the label store creation function
            create_gold_label_store(
                input_dir=silver_directory,
                output_dir=gold_directory,
                data_window=None  # Process all available data
            )
            
            output_path = os.path.join(gold_output_dir, "label_store")
            logging.info(f"Successfully created gold label store: {output_path}")
            print(f"Gold label store created: {output_path}")
            return output_path
        
    except Exception as e:
        logging.error(f"Error processing gold table {table_name} for date {snapshot_date_str}: {str(e)}")
        raise
    finally:
        # Stop Spark session only if we created it
        if 'should_stop_spark' in locals() and should_stop_spark and 'spark' in locals():
            spark.stop() 