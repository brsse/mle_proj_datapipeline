from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

def debug_missing_records():
    """Debug which records are missing from label store compared to feature store"""
    
    # Initialize Spark
    spark = SparkSession.builder.appName("MissingRecordsDebug").getOrCreate()
    
    # Read the data
    df_feature = spark.read.parquet("datamart/gold/feature_store/feature_store_week_2024_12_29")
    df_label = spark.read.parquet("datamart/gold/label_store/label_store_week_2024_12_29")
    
    print("=== DEBUGGING MISSING RECORDS ===")
    print(f"Feature store: {df_feature.count()} records")
    print(f"Label store: {df_label.count()} records")
    print(f"Difference: {df_feature.count() - df_label.count()} records")
    
    # Check if 'id' column exists in both datasets
    print("\n=== COLUMN CHECK ===")
    feature_has_id = 'id' in df_feature.columns
    label_has_id = 'id' in df_label.columns
    
    print(f"Feature store has 'id' column: {feature_has_id}")
    print(f"Label store has 'id' column: {label_has_id}")
    
    if not feature_has_id or not label_has_id:
        print("ERROR: Both datasets must have 'id' column for comparison")
        return
    
    # Get the IDs from both datasets
    feature_ids = df_feature.select("id").distinct()
    label_ids = df_label.select("id").distinct()
    
    print(f"\nFeature store unique IDs: {feature_ids.count()}")
    print(f"Label store unique IDs: {label_ids.count()}")
    
    # Find IDs that are in feature store but not in label store
    missing_in_label = feature_ids.subtract(label_ids)
    missing_in_feature = label_ids.subtract(feature_ids)
    
    print(f"\nIDs in feature store but missing in label store: {missing_in_label.count()}")
    print(f"IDs in label store but missing in feature store: {missing_in_feature.count()}")
    
    # Show some examples of missing IDs
    if missing_in_label.count() > 0:
        print("\n=== SAMPLE MISSING IDs (in feature but not in label) ===")
        missing_in_label.show(10, truncate=False)
        
        # Get the full records for some missing IDs from feature store
        print("\n=== SAMPLE MISSING RECORDS FROM FEATURE STORE ===")
        sample_missing_ids = missing_in_label.limit(5)
        missing_records = df_feature.join(sample_missing_ids, "id", "inner")
        missing_records.show(5, truncate=False)
    
    if missing_in_feature.count() > 0:
        print("\n=== SAMPLE MISSING IDs (in label but not in feature) ===")
        missing_in_feature.show(10, truncate=False)
        
        # Get the full records for some missing IDs from label store
        print("\n=== SAMPLE MISSING RECORDS FROM LABEL STORE ===")
        sample_missing_ids = missing_in_feature.limit(5)
        missing_records = df_label.join(sample_missing_ids, "id", "inner")
        missing_records.show(5, truncate=False)
    
    # Check for any patterns in the missing records
    print("\n=== PATTERN ANALYSIS ===")
    
    # Check if missing records have any common characteristics
    if missing_in_label.count() > 0:
        print("Analyzing patterns in records missing from label store...")
        
        # Get all missing records from feature store
        all_missing_records = df_feature.join(missing_in_label, "id", "inner")
        
        # Check for null values in key columns
        print("\nNull value analysis for missing records:")
        for col_name in all_missing_records.columns[:10]:  # Check first 10 columns
            null_count = all_missing_records.filter(col(col_name).isNull()).count()
            total_count = all_missing_records.count()
            if total_count > 0:
                null_percentage = (null_count / total_count) * 100
                print(f"{col_name}: {null_count} nulls ({null_percentage:.2f}%)")
    
    spark.stop()

if __name__ == "__main__":
    debug_missing_records() 