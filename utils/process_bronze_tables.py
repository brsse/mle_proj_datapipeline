import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse
import logging

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_bronze_table(snapshot_date_str, bronze_directory, spark, table_name, processing_mode='daily'):
    """
    Process bronze table data for a specific date and table
    
    Args:
        snapshot_date_str (str): Date string in YYYY-MM-DD format
        bronze_directory (str): Directory to save bronze data
        spark: Spark session (not used in this implementation, can be None)
        table_name (str): Name of the table to process
        processing_mode (str): 'daily' for single date, 'weekly' for entire week
    """
    try:
        # Parse the snapshot date
        snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
        
        # Construct the input filename
        filename = f'/opt/airflow/data/features_{table_name}.csv'
        
        # Check if input file exists
        if not os.path.exists(filename):
            logging.error(f"Input file not found: {filename}")
            raise FileNotFoundError(f"Input file not found: {filename}")
        
        # Get file size for debugging
        file_size = os.path.getsize(filename)
        logging.info(f"Processing file: {filename} (size: {file_size / (1024*1024):.2f} MB)")
        
        # Read CSV in chunks to handle large files
        chunk_size = 5000  # Reduced chunk size for better memory management
        all_chunks = []
        
        if processing_mode == 'weekly':
            # For weekly processing, get the entire week's data
            # snapshot_date is Sunday, so week runs from Sunday to Saturday
            week_start = snapshot_date  # Sunday
            week_end = snapshot_date + timedelta(days=6)  # Saturday (6 days after Sunday)
            
            logging.info(f"Processing weekly data from {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')} (7 days)")
            
            try:
                chunk_count = 0
                for chunk in pd.read_csv(filename, chunksize=chunk_size, low_memory=False):
                    chunk_count += 1
                    logging.info(f"Processing chunk {chunk_count} for {table_name}")
                    
                    # Filter for the entire week
                    chunk['snapshot_date'] = pd.to_datetime(chunk['snapshot_date'])
                    filtered_chunk = chunk[
                        (chunk['snapshot_date'] >= week_start) & 
                        (chunk['snapshot_date'] <= week_end)
                    ]
                    if not filtered_chunk.empty:
                        all_chunks.append(filtered_chunk)
                        
                    # Clear memory after each chunk
                    del chunk
            except Exception as e:
                logging.error(f"Error reading CSV file {filename}: {str(e)}")
                raise
        else:
            # For daily processing, get only the specific date
            try:
                chunk_count = 0
                for chunk in pd.read_csv(filename, chunksize=chunk_size, low_memory=False):
                    chunk_count += 1
                    logging.info(f"Processing chunk {chunk_count} for {table_name}")
                    
                    # Filter for the specific snapshot date
                    filtered_chunk = chunk[chunk['snapshot_date'] == snapshot_date_str]
                    if not filtered_chunk.empty:
                        all_chunks.append(filtered_chunk)
                        
                    # Clear memory after each chunk
                    del chunk
            except Exception as e:
                logging.error(f"Error reading CSV file {filename}: {str(e)}")
                raise
        
        if not all_chunks:
            if processing_mode == 'weekly':
                logging.warning(f"No data found for week starting {snapshot_date_str} in table {table_name}")
            else:
                logging.warning(f"No data found for date {snapshot_date_str} in table {table_name}")
            # Create empty DataFrame with same columns as original
            try:
                sample_df = pd.read_csv(filename, nrows=1)
                df = pd.DataFrame(columns=sample_df.columns)
            except Exception as e:
                logging.error(f"Error reading sample from {filename}: {str(e)}")
                raise
        else:
            # Combine all chunks
            df = pd.concat(all_chunks, ignore_index=True)
        
        if processing_mode == 'weekly':
            logging.info(f"Week starting {snapshot_date_str} row count for {table_name}: {len(df)}")
        else:
            logging.info(f"{snapshot_date_str} row count for {table_name}: {len(df)}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(bronze_directory, table_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output filename
        if processing_mode == 'weekly':
            partition_name = f"bronze_{table_name}_week_{snapshot_date.strftime('%Y_%m_%d')}.csv"
        else:
            partition_name = f"bronze_{table_name}_{snapshot_date_str.replace('-', '_')}.csv"
        filepath = os.path.join(output_dir, partition_name)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f'Saved to: {filepath}')
        
        return filepath
        
    except Exception as e:
        logging.error(f"Error processing bronze table {table_name} for date {snapshot_date_str}: {str(e)}")
        raise