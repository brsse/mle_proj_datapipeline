import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import xgboost as xgb
import mlflow
import mlflow.sklearn
import json
from datetime import datetime, timedelta
import os
import sys
import time
from typing import List

# Import from parent utils directory since we're in archive/
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model_operations import load_data_for_training

def get_date_range_for_training(end_date: datetime, num_weeks: int) -> List[str]:
    """
    Calculates the list of weekly partition strings for data loading.
    """
    weeks = []
    for i in range(num_weeks):
        partition_date = end_date - timedelta(weeks=i)
        weeks.append(partition_date.strftime('%Y_%m_%d'))
    return sorted(weeks)

def load_and_prepare_data():
    """
    Load and prepare the data for training.
    """
    start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting data loading...")
    
    # Initialize Spark
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing Spark...")
    spark = SparkSession.builder \
        .appName("ModelTraining") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # Configuration
    SNAPSHOT_DATE = datetime(2023, 1, 1)
    TRAINING_WEEKS = 24  # 6 months of data for robust training
    FEATURE_STORE_PATH = "/opt/airflow/datamart/gold/feature_store"
    LABEL_STORE_PATH = "/opt/airflow/datamart/gold/label_store"
    
    # Get training weeks
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Calculating training weeks...")
    training_weeks = get_date_range_for_training(SNAPSHOT_DATE, TRAINING_WEEKS)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data for {len(training_weeks)} weeks, from {training_weeks[0]} to {training_weeks[-1]}")
    
    # Load data using the original approach
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Calling load_data_for_training...")
    load_start = time.time()
    
    try:
        full_df = load_data_for_training(spark, FEATURE_STORE_PATH, LABEL_STORE_PATH, training_weeks)
        load_time = time.time() - load_start
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Data loading completed in {load_time:.2f} seconds")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully loaded {full_df.shape[0]} records.")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR during data loading: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] This might be due to:")
        print(f"  - Missing parquet files")
        print(f"  - Memory issues with .toPandas()")
        print(f"  - Spark configuration problems")
        raise
    
    # Prepare features and target
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Preparing features and target...")
    # Drop columns that cause issues with XGBoost (date columns and other non-numeric)
    features_to_drop = [
        'id',  # UNIQUE_ID_COLUMN
        'grade',  # TARGET_COLUMN
        'snapshot_date', 
        'earliest_cr_date',
        'snapshot_month',
        'earliest_cr_month',
        'months_since_earliest_cr_line'
    ]
    
    # Drop the problematic columns
    full_df_cleaned = full_df.drop(columns=features_to_drop, errors='ignore')
    
    # Get remaining feature columns
    feature_columns = [col for col in full_df_cleaned.columns if col != 'grade_encoded']
    X = full_df_cleaned[feature_columns]
    y = full_df['grade']
    
    # Create grade mapping
    grade_mapping = {grade: idx for idx, grade in enumerate(sorted(y.unique()))}
    y_encoded = y.map(grade_mapping)
    
    total_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Data preparation completed in {total_time:.2f} seconds")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Data shape: {X.shape}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Grade distribution: {y.value_counts().to_dict()}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Grade mapping: {grade_mapping}")
    
    return X, y_encoded, grade_mapping

def main():
    """
    Main training function with fast hyperparameter tuning.
    """
    total_start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting fast XGBoost training...")
    
    # Load data
    X, y, grade_mapping = load_and_prepare_data()
    
    # Split data
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training set size: {X_train.shape[0]}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Test set size: {X_test.shape[0]}")
    
    # Create a smaller sample for fast hyperparameter tuning
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating tuning sample...")
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train, y_train, train_size=0.1, random_state=42, stratify=y_train
    )
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tuning sample size: {X_train_sample.shape[0]}")
    
    # Define a much simpler parameter space for very fast tuning
    param_dist = {
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8],
        'min_child_weight': [1, 5]
    }
    
    # Initialize base model
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initializing XGBoost model...")
    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(grade_mapping),
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    
    # Fast hyperparameter tuning with RandomizedSearchCV
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting fast hyperparameter tuning...")
    tuning_start = time.time()
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=5,  # Reduced from 10 to 5 for speed
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,  # Use all CPU cores
        random_state=42,
        verbose=1  # Reduced verbosity
    )
    
    # Fit the random search
    random_search.fit(X_train_sample, y_train_sample)
    tuning_time = time.time() - tuning_start
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Hyperparameter tuning completed in {tuning_time:.2f} seconds")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Best parameters: {random_search.best_params_}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Train final model on full training data with best parameters
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training final model on full dataset...")
    final_start = time.time()
    final_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(grade_mapping),
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        **random_search.best_params_
    )
    
    # Use early stopping for final training
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    final_model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val_final, y_val_final)],
        verbose=False
    )
    final_time = time.time() - final_start
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Final model training completed in {final_time:.2f} seconds")
    
    # Evaluate on test set
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluating model...")
    y_pred = final_model.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Final Test Performance:")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Log to MLflow
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Logging to MLflow...")
    try:
        # Use the service name 'mlflow' instead of 'localhost' for Docker networking
        mlflow.set_tracking_uri("http://mlflow:5000")

        # Set the experiment for this run. Will be created if it doesn't exist.
        mlflow.set_experiment("test")

        # Set a descriptive run name for easy identification
        run_name = f"xgboost_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(random_search.best_params_)
            mlflow.log_param("num_classes", len(grade_mapping))
            mlflow.log_param("training_samples", X_train.shape[0])
            mlflow.log_param("test_samples", X_test.shape[0])
            
            # Log metrics
            mlflow.log_metric("macro_f1_score", macro_f1)
            
            # Log model
            mlflow.sklearn.log_model(final_model, "model")
            
            # Log grade mapping
            mlflow.log_dict(grade_mapping, "grade_mapping.json")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully logged to MLflow")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: MLflow logging failed: {e}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Continuing with local metrics save...")
    
    # Save baseline metrics locally
    baseline_metrics = {
        "model_type": "XGBoost",
        "macro_f1_score": macro_f1,
        "best_params": random_search.best_params_,
        "grade_mapping": grade_mapping,
        "training_timestamp": datetime.now().isoformat(),
        "training_samples": X_train.shape[0],
        "test_samples": X_test.shape[0]
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = f"model_bank/baseline_metrics_{timestamp}.json"
    
    os.makedirs("model_bank", exist_ok=True)
    with open(metrics_file, 'w') as f:
        json.dump(baseline_metrics, f, indent=2)
    
    total_time = time.time() - total_start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Baseline metrics saved to: {metrics_file}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Total training time: {total_time:.2f} seconds")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed successfully!")

if __name__ == "__main__":
    main() 