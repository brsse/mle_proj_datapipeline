import glob
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession

def analyze_gold_data():
    """Analyze feature store and label store data"""
    
    # Initialize Spark
    spark = SparkSession.builder.appName("GoldDataAnalysis").getOrCreate()
    
    # Paths
    feature_store_path = "/opt/airflow/datamart/gold/feature_store"
    label_store_path = "/opt/airflow/datamart/gold/label_store"
    
    print("=== GOLD LAYER DATA ANALYSIS ===\n")
    
    # Analyze Feature Store
    print("1. FEATURE STORE ANALYSIS")
    print("-" * 50)
    
    # Get all feature store directories
    feature_dirs = sorted(glob.glob(f"{feature_store_path}/feature_store_week_*"))
    print(f"Found {len(feature_dirs)} feature store directories")
    
    feature_data = []
    for dir_path in feature_dirs:
        try:
            # Extract date from directory name
            date_str = dir_path.split("_week_")[-1]
            date_obj = datetime.strptime(date_str, "%Y_%m_%d")
            
            # Read parquet and count records
            df = spark.read.parquet(dir_path)
            record_count = df.count()
            
            # Check if grade column exists (it shouldn't after processing)
            has_grade = 'grade' in df.columns
            
            # Feature store doesn't have snapshot_date column, so we use directory date
            feature_data.append({
                'directory': dir_path.split('/')[-1],
                'date': date_obj,
                'date_str': date_str,
                'record_count': record_count,
                'snapshot_dates': [date_obj],  # Use directory date as snapshot date
                'has_grade': has_grade
            })
            
            print(f"  {dir_path.split('/')[-1]}: {record_count} records, has_grade: {has_grade}")
            
        except Exception as e:
            print(f"  Error reading {dir_path}: {e}")
    
    # Analyze Label Store
    print("\n2. LABEL STORE ANALYSIS")
    print("-" * 50)
    
    # Get all label store directories
    label_dirs = sorted(glob.glob(f"{label_store_path}/label_store_week_*"))
    print(f"Found {len(label_dirs)} label store directories")
    
    label_data = []
    for dir_path in label_dirs:
        try:
            # Extract date from directory name
            date_str = dir_path.split("_week_")[-1]
            date_obj = datetime.strptime(date_str, "%Y_%m_%d")
            
            # Read parquet and count records
            df = spark.read.parquet(dir_path)
            record_count = df.count()
            
            # Check if label store has snapshot_date column
            has_snapshot_date = 'snapshot_date' in df.columns
            has_grade = 'grade' in df.columns
            
            if has_snapshot_date:
                # Get distinct snapshot dates in this file
                distinct_dates = df.select("snapshot_date").distinct().collect()
                date_list = [row['snapshot_date'] for row in distinct_dates]
            else:
                # Use directory date if no snapshot_date column
                date_list = [date_obj]
            
            label_data.append({
                'directory': dir_path.split('/')[-1],
                'date': date_obj,
                'date_str': date_str,
                'record_count': record_count,
                'snapshot_dates': date_list,
                'has_snapshot_date': has_snapshot_date,
                'has_grade': has_grade
            })
            
            print(f"  {dir_path.split('/')[-1]}: {record_count} records, {len(date_list)} snapshot dates, has_grade: {has_grade}")
            
        except Exception as e:
            print(f"  Error reading {dir_path}: {e}")
    
    # Compare data
    print("\n3. COMPARISON ANALYSIS")
    print("-" * 50)
    
    # Create DataFrames for comparison
    feature_df = pd.DataFrame(feature_data)
    label_df = pd.DataFrame(label_data)
    
    print(f"Feature store directories: {len(feature_df)}")
    print(f"Label store directories: {len(label_df)}")
    
    # Compare dates
    feature_dates = set(feature_df['date_str'].tolist())
    label_dates = set(label_df['date_str'].tolist())
    
    print(f"\nFeature store dates: {len(feature_dates)}")
    print(f"Label store dates: {len(label_dates)}")
    
    # Find mismatches
    feature_only = feature_dates - label_dates
    label_only = label_dates - feature_dates
    common_dates = feature_dates & label_dates
    
    print(f"\nCommon dates: {len(common_dates)}")
    print(f"Feature store only: {len(feature_only)}")
    print(f"Label store only: {len(label_only)}")
    
    if feature_only:
        print(f"\nDates only in feature store: {sorted(feature_only)}")
    if label_only:
        print(f"Dates only in label store: {sorted(label_only)}")
    
    # Compare record counts for common dates
    print("\n4. RECORD COUNT COMPARISON (Common Dates)")
    print("-" * 50)
    
    comparison_data = []
    for date_str in sorted(common_dates):
        feature_row = feature_df[feature_df['date_str'] == date_str].iloc[0]
        label_row = label_df[label_df['date_str'] == date_str].iloc[0]
        
        feature_count = feature_row['record_count']
        label_count = label_row['record_count']
        difference = feature_count - label_count
        
        comparison_data.append({
            'date': date_str,
            'feature_count': feature_count,
            'label_count': label_count,
            'difference': difference,
            'feature_snapshots': len(feature_row['snapshot_dates']),
            'label_snapshots': len(label_row['snapshot_dates']),
            'feature_has_grade': feature_row['has_grade'],
            'label_has_grade': label_row['has_grade']
        })
        
        print(f"  {date_str}: Feature={feature_count}, Label={label_count}, Diff={difference}, Feature_grade={feature_row['has_grade']}, Label_grade={label_row['has_grade']}")
    
    # Summary statistics
    print("\n5. SUMMARY STATISTICS")
    print("-" * 50)
    
    total_feature_records = feature_df['record_count'].sum()
    total_label_records = label_df['record_count'].sum()
    
    print(f"Total feature store records: {total_feature_records}")
    print(f"Total label store records: {total_label_records}")
    print(f"Total difference: {total_feature_records - total_label_records}")
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        print(f"\nAverage difference per week: {comp_df['difference'].mean():.2f}")
        print(f"Max difference: {comp_df['difference'].max()}")
        print(f"Min difference: {comp_df['difference'].min()}")
    
    # Show sample of differences
    print("\n6. SAMPLE DIFFERENCES (First 10 weeks)")
    print("-" * 50)
    
    for i, comp in enumerate(comparison_data[:10]):
        print(f"  Week {i+1} ({comp['date']}): Feature={comp['feature_count']}, Label={comp['label_count']}, Diff={comp['difference']}")
    
    # Check for patterns in differences
    print("\n7. DIFFERENCE PATTERN ANALYSIS")
    print("-" * 50)
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        positive_diff = comp_df[comp_df['difference'] > 0]
        negative_diff = comp_df[comp_df['difference'] < 0]
        zero_diff = comp_df[comp_df['difference'] == 0]
        
        print(f"Weeks with more feature records: {len(positive_diff)}")
        print(f"Weeks with more label records: {len(negative_diff)}")
        print(f"Weeks with equal records: {len(zero_diff)}")
        
        if len(positive_diff) > 0:
            print(f"Average excess in feature store: {positive_diff['difference'].mean():.2f}")
        if len(negative_diff) > 0:
            print(f"Average excess in label store: {abs(negative_diff['difference'].mean()):.2f}")
    
    spark.stop()
    return feature_data, label_data, comparison_data