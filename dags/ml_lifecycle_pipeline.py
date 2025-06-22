from airflow import DAG
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2022, 1, 16),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'ml_lifecycle_pipeline',
    default_args=default_args,
    description='End-to-end ML lifecycle pipeline for credit scoring',
    schedule_interval='0 6 * * 1',  # Weekly on Mondays at 6 AM
    catchup=True,
    tags=['p2p', 'credit-scoring', 'ml-lifecycle']
) as dag:

    # === Start of Pipeline ===
    start = DummyOperator(task_id='start_pipeline')

    # === Phase 1: Data Availability Check ===
    # See Parked Item #2: Check if at least 6 months (24 weeks) of data is available.
    # If not, all downstream tasks should be skipped.
    check_data_availability = DummyOperator(task_id='check_data_availability')

    # === Phase 2: Data Preprocessing ===
    # This phase mirrors the structure from the original `dag.py`.
    # It converts raw data through bronze, silver, and gold layers.
    
    start_preprocessing = DummyOperator(task_id='start_preprocessing')

    # --- Bronze Layer ---
    # Original script: process_bronze_tables.py
    process_bronze = DummyOperator(task_id='process_bronze_layer')

    # --- Silver Layer ---
    # Original scripts: silver_*.py, process_silver_tables.py
    process_silver = DummyOperator(task_id='process_silver_layer')
    
    # --- Gold Layer ---
    # Original scripts: gold_*.py, process_gold_tables.py
    process_gold_tables = DummyOperator(task_id='process_gold_layer_tables')
    
    # --- Gold Layer: Feature & Label Stores ---
    # Original scripts: gold_feature_store.py, gold_label_store.py
    create_feature_store = DummyOperator(task_id='create_feature_store')
    create_label_store = DummyOperator(task_id='create_label_store')

    end_preprocessing = DummyOperator(task_id='end_preprocessing')

    # === Phase 3: Model Retraining Logic ===
    # This phase will handle the logic for model retraining, including checking triggers
    # and preparing the final training dataset.
    
    start_retraining_logic = DummyOperator(task_id='start_retraining_logic')
    
    # See Parked Item #3: Check for retraining triggers (performance threshold or 3-month refresh).
    check_retraining_trigger = DummyOperator(task_id='check_retraining_trigger')

    # See Parked Item #3: Prepare the 6-month training data window.
    prepare_training_data = DummyOperator(task_id='prepare_training_data')

    # === Phase 4: Parallel Model Training ===
    # See Parked Item #8: Train LightGBM and CatBoost models in parallel.
    # Note: XGBoost is excluded as per Parked Item #5.

    # Original script: LightGBM_training_run.py
    train_lightgbm = DummyOperator(task_id='train_lightgbm_model')
    
    # Original script: CatBoost_training_run.py
    train_catboost = DummyOperator(task_id='train_catboost_model')

    # === Phase 5: Model Selection and Registration ===
    # This phase selects the best model from the parallel runs and registers it.

    # See Parked Item #5: Select the best performing model (LightGBM vs. CatBoost).
    select_best_model = DummyOperator(task_id='select_best_model')

    # See Parked Item #6: Register the winning model in the MLflow Model Registry.
    register_model = DummyOperator(task_id='register_model_in_registry')

    end_retraining_logic = DummyOperator(task_id='end_retraining_logic')

    # === End of Pipeline ===
    end = DummyOperator(task_id='end_pipeline')


    # --- Define DAG Dependencies ---

    # Preprocessing Dependencies
    start >> check_data_availability >> start_preprocessing
    start_preprocessing >> process_bronze >> process_silver >> process_gold_tables
    process_gold_tables >> [create_feature_store, create_label_store] >> end_preprocessing

    # Retraining Dependencies
    end_preprocessing >> start_retraining_logic
    start_retraining_logic >> check_retraining_trigger >> prepare_training_data

    # Parallel Training Dependencies
    prepare_training_data >> [train_lightgbm, train_catboost]

    # Model Selection Dependencies (waits for both to finish)
    [train_lightgbm, train_catboost] >> select_best_model >> register_model
    
    register_model >> end_retraining_logic >> end