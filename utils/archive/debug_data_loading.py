#!/usr/bin/env python3
"""
Debug script to identify where data loading is hanging.
"""

import time
import os
import sys
from datetime import datetime, timedelta
from typing import List

def get_date_range_for_training(end_date, num_weeks: int) -> List[str]:
    """Calculates the list of weekly partition strings for data loading."""
    weeks = []
    for i in range(num_weeks):
        partition_date = end_date - timedelta(weeks=i)
        weeks.append(partition_date.strftime('%Y_%m_%d'))
    return sorted(weeks)

def debug_data_loading():
    """Debug the data loading process step by step."""
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] === DATA LOADING DEBUG ===")
    
    # Step 1: Check if we can import Spark
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 1: Testing Spark import...")
    try:
        from pyspark.sql import SparkSession
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Spark import successful")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Spark import failed: {e}")
        return
    
    # Step 2: Test Spark initialization
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 2: Testing Spark initialization...")
    try:
        spark = SparkSession.builder \
            .appName("DebugDataLoading") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Spark initialization successful")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Spark initialization failed: {e}")
        return
    
    # Step 3: Check data paths
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 3: Checking data paths...")
    FEATURE_STORE_PATH = "/opt/airflow/datamart/gold/feature_store"
    LABEL_STORE_PATH = "/opt/airflow/datamart/gold/label_store"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Feature store path: {FEATURE_STORE_PATH}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Label store path: {LABEL_STORE_PATH}")
    
    # Check if directories exist
    if os.path.exists(FEATURE_STORE_PATH):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Feature store directory exists")
        feature_dirs = [d for d in os.listdir(FEATURE_STORE_PATH) if d.startswith('feature_store_week_')]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(feature_dirs)} feature directories")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Feature store directory does not exist")
        return
    
    if os.path.exists(LABEL_STORE_PATH):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Label store directory exists")
        label_dirs = [d for d in os.listdir(LABEL_STORE_PATH) if d.startswith('label_store_week_')]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(label_dirs)} label directories")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Label store directory does not exist")
        return
    
    # Step 4: Test reading a single week
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 4: Testing single week reading...")
    try:
        # Get first week
        SNAPSHOT_DATE = datetime(2023, 1, 1)
        training_weeks = get_date_range_for_training(SNAPSHOT_DATE, 1)
        first_week = training_weeks[0]
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing week: {first_week}")
        
        feature_path = os.path.join(FEATURE_STORE_PATH, f"feature_store_week_{first_week}")
        label_path = os.path.join(LABEL_STORE_PATH, f"label_store_week_{first_week}")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Reading feature path: {feature_path}")
        features_df = spark.read.parquet(feature_path)
        feature_count = features_df.count()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Features loaded: {feature_count} records")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Reading label path: {label_path}")
        labels_df = spark.read.parquet(label_path)
        label_count = labels_df.count()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Labels loaded: {label_count} records")
        
        # Test join
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing join...")
        joined_df = features_df.join(labels_df, "id")
        joined_count = joined_df.count()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Join successful: {joined_count} records")
        
        # Test pandas conversion
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing pandas conversion...")
        pandas_df = joined_df.toPandas()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Pandas conversion successful: {pandas_df.shape}")
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Single week test failed: {e}")
        return
    
    # Step 5: Test chunked loading
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 5: Testing chunked loading...")
    try:
        # Test with just 2 weeks
        test_weeks = get_date_range_for_training(SNAPSHOT_DATE, 2)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Testing with weeks: {test_weeks}")
        
        # Import the function
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from model_operations import load_data_for_training
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Calling load_data_for_training...")
        result_df = load_data_for_training(spark, FEATURE_STORE_PATH, LABEL_STORE_PATH, test_weeks)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Chunked loading successful: {result_df.shape}")
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Chunked loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] === DEBUG COMPLETE ===")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] All tests passed! The issue might be with the full 50-week load.")

if __name__ == "__main__":
    debug_data_loading() 