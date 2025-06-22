from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

def find_duplicate_ids():
    """Finds and displays records with duplicate IDs in the feature store,
       and analyzes which columns are unique vs. non-unique."""
    
    spark = SparkSession.builder.appName("FindDuplicateIDs").getOrCreate()
    
    feature_store_path = "datamart/gold/feature_store/feature_store_week_2024_12_29"
    print(f"Reading feature store from: {feature_store_path}")
    
    df_feature = spark.read.parquet(feature_store_path)
    df_feature.cache() # Cache for better performance
    
    print(f"\nTotal records in feature store: {df_feature.count()}")
    
    # Group by 'id' and count occurrences
    id_counts = df_feature.groupBy("id").count()
    
    # Filter for IDs that appear more than once
    duplicate_ids_df = id_counts.filter(col("count") > 1)
    
    num_duplicate_ids = duplicate_ids_df.count()
    print(f"Number of IDs with duplicates: {num_duplicate_ids}")
    
    if num_duplicate_ids > 0:
        print("\n--- IDs with Duplicate Records ---")
        duplicate_ids_df.orderBy(col("count").desc()).show()
        
        print("\n--- Analysis of Duplicate Records (Sample) ---")
        # Take a sample of 5 duplicate IDs to inspect
        duplicate_ids_to_check = [row.id for row in duplicate_ids_df.limit(5).collect()]
        
        for dup_id in duplicate_ids_to_check:
            print(f"\n--- Analyzing duplicates for ID: {dup_id} ---")
            
            # Get all records for this specific duplicate ID
            duplicate_records = df_feature.filter(col("id") == dup_id).collect()
            
            if not duplicate_records:
                continue
                
            # Display the duplicate records
            print("Duplicate rows found:")
            df_feature.filter(col("id") == dup_id).show(truncate=False)

            first_record = duplicate_records[0].asDict()
            all_columns = list(first_record.keys())
            non_unique_cols = set()
            
            # Compare the first record with all other duplicate records for this ID
            for i in range(1, len(duplicate_records)):
                current_record = duplicate_records[i].asDict()
                for col_name in all_columns:
                    if first_record[col_name] != current_record[col_name]:
                        non_unique_cols.add(col_name)

            unique_cols = set(all_columns) - non_unique_cols
            
            print(f"For ID {dup_id}:")
            if non_unique_cols:
                print(f"  Columns with DIFFERENT values: {sorted(list(non_unique_cols))}")
            else:
                print("  All columns have the same values (exact duplicates).")

            if unique_cols:
                print(f"  Columns with the SAME values: {sorted(list(unique_cols))}")

    else:
        print("\nNo duplicate IDs found in the feature store.")
        
    df_feature.unpersist()
    spark.stop()

if __name__ == "__main__":
    find_duplicate_ids() 