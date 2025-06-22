import os
import glob
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Import silver processing functions
from silver_credit_history import process_credit_history
from silver_demographic import process_demographic
from silver_financial import process_financial
from silver_loan_terms import process_loan_terms


def process_silver_table(snapshot_date_str, bronze_directory, silver_directory, table_name, spark=None):
    """
    Process silver table data for a specific date and table by reading from bronze datamart
    
    Args:
        snapshot_date_str (str): Date string in YYYY-MM-DD format
        bronze_directory (str): Directory containing bronze data
        silver_directory (str): Directory to save silver data
        table_name (str): Name of the table to process (credit_history, demographic, financial, loan_terms)
        spark: Spark session (optional, will create one if not provided)
    """
    try:
        # Initialize Spark session if not provided
        if spark is None:
            spark = SparkSession.builder.appName(f"SilverProcessing_{table_name}").getOrCreate()
            should_stop_spark = True
        else:
            should_stop_spark = False
        
        # Setup logging
        log_dir = Path("/opt/airflow/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=str(log_dir / "silver_processing.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            force=True 
        )
        
        # Parse the snapshot date
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        
        # Define the mapping of table names to their processing functions
        table_processors = {
            "credit_history": process_credit_history,
            "demographic": process_demographic,
            "financial": process_financial,
            "loan_terms": process_loan_terms
        }
        
        if table_name not in table_processors:
            raise ValueError(f"Unsupported table name: {table_name}. Supported tables: {list(table_processors.keys())}")
        
        processor_func = table_processors[table_name]
        
        # Construct bronze input path - bronze processing creates weekly files
        bronze_table_dir = os.path.join(bronze_directory, table_name)
        date_str = snapshot_date.strftime('%Y_%m_%d')
        bronze_file = os.path.join(bronze_table_dir, f"bronze_{table_name}_week_{date_str}.csv")
        
        if not os.path.exists(bronze_file):
            logging.warning(f"Bronze file not found for week starting {snapshot_date_str} in table {table_name}: {bronze_file}")
            return None
        
        logging.info(f"Processing bronze file: {bronze_file}")
        
        # Process the bronze file using the appropriate silver processing function
        processed_df = processor_func(spark, bronze_file)
        
        # Create output directory
        silver_table_dir = os.path.join(silver_directory, table_name)
        os.makedirs(silver_table_dir, exist_ok=True)
        
        # Write processed data to silver layer
        partition_name = f"silver_{table_name}_week_{snapshot_date.strftime('%Y_%m_%d')}"
        output_path = os.path.join(silver_table_dir, partition_name)
        
        logging.info(f"Writing silver data to: {output_path}")
        processed_df.write.mode("overwrite").parquet(output_path)
        
        logging.info(f"Successfully processed silver table {table_name} for week starting {snapshot_date_str}")
        print(f"Silver processing complete for {table_name}: {output_path}")
        
        return output_path
        
    except Exception as e:
        logging.error(f"Error processing silver table {table_name} for date {snapshot_date_str}: {str(e)}")
        raise
    finally:
        # Stop Spark session only if we created it
        if 'should_stop_spark' in locals() and should_stop_spark and 'spark' in locals():
            spark.stop() 