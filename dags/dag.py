from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.state import State
from datetime import datetime, timedelta
import os
import json
import sys
import pandas as pd
from pyspark.sql import SparkSession
from typing import List
from airflow.exceptions import AirflowSkipException

# Import DAG functions from separate module
sys.path.append('/opt/airflow/utils')
from dag_functions import (
    decide_pipeline_path,
    check_retraining_trigger,
    select_best_model_initial,
    register_model_initial,
    evaluate_production_model,
    train_lightgbm_weekly,
    train_catboost_weekly,
    select_best_model_weekly,
    register_model_weekly,
    run_model_inference,
    # check_data_availability
)
from silver_credit_history import process_credit_history
from silver_demographic import process_demographic
from silver_financial import process_financial
from silver_loan_terms import process_loan_terms
from gold_credit_history import process_credit_history as process_gold_credit_history
from gold_demographic import process_demographic as process_gold_demographic
from gold_financial import process_financial as process_gold_financial
from gold_loan_terms import process_loan_terms as process_gold_loan_terms
from process_bronze_tables import process_bronze_table
from process_silver_tables import process_silver_table
from process_gold_tables import process_gold_table
from gold_feature_store import create_feature_store
from gold_label_store import create_gold_label_store

class SafeExternalTaskSensor(ExternalTaskSensor):
    def poke(self, context):
        prev_execution_date = context['execution_date'] - timedelta(weeks=1)
        dag = context['dag']
        dag_start_date = getattr(dag, 'start_date', None) or (dag.default_args.get('start_date') if hasattr(dag, 'default_args') else None)
        if dag_start_date is None:
            # If we can't determine the start date, always succeed
            return True
        if prev_execution_date < dag_start_date:
            # Mark as success explicitly
            context['ti'].set_state(State.SUCCESS)
            return True
        return super().poke(context)

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': True,  # Critical: Each run waits for previous week's run to complete
    'start_date': datetime(2022, 12, 25),
    'end_date': datetime(2024, 12, 31),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'data_ml_pipeline',
    default_args=default_args,
    description='End-to-end ML lifecycle pipeline for credit scoring',
    schedule_interval='0 6 * * 0',
    catchup=True,
    tags=['data-preprocessing', 'credit-scoring', 'ml-lifecycle']
) as dag:
    """
    ML Lifecycle Pipeline with Weekly Dependencies
    
    CRITICAL: This DAG uses an ExternalTaskSensor to ensure proper weekly dependencies.
    - Each week's run waits for the previous week's 'end_pipeline' task to complete.
    - This prevents race conditions and ensures data consistency.
    - The 'end_pipeline' task uses trigger_rule='one_success' to act as a reliable
      join point for all possible DAG branches.
    """

    # This sensor ensures that the current week's run only starts after the
    # previous week's run has fully completed.
    wait_for_previous_run = SafeExternalTaskSensor(
        task_id='wait_for_previous_run',
        external_dag_id='data_ml_pipeline',
        external_task_id='end_pipeline',
        execution_delta=timedelta(weeks=1),
        allowed_states=['success'],
        mode='poke',
        timeout=60 * 60 * 3,
        poke_interval=5,  # Check every 5 seconds for faster backfill
    )

    # === Start of Pipeline ===
    start = DummyOperator(task_id='start_pipeline')

    # === Phase 1: Data Availability Check ===

    # Source data checks
    dep_check_source_credit_history = FileSensor(
        task_id='dep_check_source_credit_history',
        filepath='/opt/airflow/data/features_credit_history.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )
    
    dep_check_source_demographic = FileSensor(
        task_id='dep_check_source_demographic',
        filepath='/opt/airflow/data/features_demographic.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )
    
    dep_check_source_financial = FileSensor(
        task_id='dep_check_source_financial',
        filepath='/opt/airflow/data/features_financial.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )
    
    dep_check_source_loan_terms = FileSensor(
        task_id='dep_check_source_loan_terms',
        filepath='/opt/airflow/data/features_loan_terms.csv',
        poke_interval=10,
        timeout=600,
        mode='poke'
    )

    # Bronze layer processing
    # bronze_table_cred_history = DummyOperator(task_id='bronze_table_credit_history')
    bronze_table_cred_history = PythonOperator(
        task_id='bronze_table_cred_history',
        python_callable=process_bronze_table,
        op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', None, 'credit_history', 'weekly']
    )
    
    # bronze_table_demographic = DummyOperator(task_id='bronze_table_demographic')
    bronze_table_demographic = PythonOperator(
        task_id='bronze_table_demographic',
        python_callable=process_bronze_table,
        op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', None, 'demographic', 'weekly']
    )
    
    # bronze_table_financial = DummyOperator(task_id='bronze_table_financial')
    bronze_table_financial = PythonOperator(
        task_id='bronze_table_financial',
        python_callable=process_bronze_table,
        op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', None, 'financial', 'weekly']
    )
    
    # bronze_table_loan_term = DummyOperator(task_id='bronze_table_loan_term')
    bronze_table_loan_term = PythonOperator(
        task_id='bronze_table_loan_term',
        python_callable=process_bronze_table,
        op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', None, 'loan_terms', 'weekly']
    )

    # Silver layer processing
    # silver_table_cred_history = DummyOperator(task_id='silver_table_cred_history')
    silver_table_cred_history = PythonOperator(
        task_id='silver_table_cred_history',
        python_callable=process_silver_table,
        op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', '/opt/airflow/datamart/silver/', 'credit_history', None]
    )
    
    # silver_table_demographic = DummyOperator(task_id='silver_table_demographic')
    silver_table_demographic = PythonOperator(
        task_id='silver_table_demographic',
        python_callable=process_silver_table,
        op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', '/opt/airflow/datamart/silver/', 'demographic', None]
    )
    
    # silver_table_financial = DummyOperator(task_id='silver_table_financial')
    silver_table_financial = PythonOperator(
        task_id='silver_table_financial',
        python_callable=process_silver_table,
        op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', '/opt/airflow/datamart/silver/', 'financial', None]
    )
    
    # silver_table_loan_term = DummyOperator(task_id='silver_table_loan_term')
    silver_table_loan_term = PythonOperator(
        task_id='silver_table_loan_term',
        python_callable=process_silver_table,
        op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', '/opt/airflow/datamart/silver/', 'loan_terms', None]
    )

    # Gold layer processing - FEATURE AND LABEL STORE (ONLY ACTIVE TASKS)
    # gold_feature_store = DummyOperator(task_id='gold_feature_store')
    gold_feature_store = PythonOperator(
        task_id='gold_feature_store',
        python_callable=create_feature_store,
        op_args=['/opt/airflow/datamart/silver/', '/opt/airflow/datamart/gold/', '{{ ds }}']
    )
    
    # gold_label_store = DummyOperator(task_id='gold_label_store')
    gold_label_store = PythonOperator(
        task_id='gold_label_store',
        python_callable=create_gold_label_store,
        op_args=['/opt/airflow/datamart/silver/', '/opt/airflow/datamart/gold/', '{{ ds }}']
    )

    # === Phase 2: Data Preprocessing ===
    # This phase mirrors the structure from the original `dag.py`.
    # Each task would be a PythonOperator calling a function from a utils script.
    
    start_preprocessing = DummyOperator(task_id='start_preprocessing')

    # Calls a function from utils/process_bronze_tables.py
    # process_bronze = DummyOperator(task_id='process_bronze_layer')

    # Calls a function from utils/process_silver_tables.py
    # process_silver = DummyOperator(task_id='process_silver_layer')
    
    # Calls a function from utils/process_gold_tables.py
    # process_gold_tables = DummyOperator(task_id='process_gold_layer_tables')
    
    # Calls a function from utils/gold_feature_store.py
    # create_feature_store = DummyOperator(task_id='create_feature_store')
    
    # Calls a function from utils/gold_label_store.py
    # create_label_store = DummyOperator(task_id='create_label_store')

    end_preprocessing = DummyOperator(task_id='end_preprocessing')

    # === Gate 1: Decide on the main pipeline path ===
    decide_pipeline_path_task = BranchPythonOperator(
        task_id='decide_pipeline_path',
        python_callable=decide_pipeline_path,
    )

    # --- Path 1: Skip (for historical runs before 2023) ---
    skip_run = DummyOperator(task_id='skip_run')

    # --- Path 2: One-Time Initial Training Flow ---
    # This path runs only once to create the first model. It does not run inference.
    run_initial_training_flow = DummyOperator(task_id='run_initial_training_flow')
    
    # Use existing training scripts instead of duplicating logic
    train_lightgbm_initial = BashOperator(
        task_id='train_lightgbm_initial',
        bash_command='cd /opt/airflow/utils && python LightGBM_training_run.py',
        do_xcom_push=True,
    )
    
    train_catboost_initial = BashOperator(
        task_id='train_catboost_initial',
        bash_command='cd /opt/airflow/utils && python CatBoost_training_run.py',
        do_xcom_push=True,
    )
    
    # Combined approach: Extract results from both training tasks and select best model
    select_best_model_initial_task = PythonOperator(
        task_id='select_best_model_initial',
        python_callable=select_best_model_initial,
    )
    
    # PythonOperator that takes the winning run_id from XComs.
    # It uses the MLflow client to register a new model version and transition it to "Production".
    register_model_initial_task = PythonOperator(
        task_id='register_model_initial',
        python_callable=register_model_initial,
    )

    # --- Path 3: Standard Weekly Lifecycle Flow ---
    # This path evaluates the prod model and decides to retrain or just run inference.
    run_weekly_lifecycle_flow = DummyOperator(task_id='run_weekly_lifecycle_flow')
    
    # PythonOperator that evaluates the current "Production" model on the new week's data.
    # Uses logic from utils/weekly_evaluation.py.
    # It must save metrics to a database and push key metrics (e.g., oot_auc) to XComs.
    evaluate_production_model_task = PythonOperator(
        task_id='evaluate_production_model',
        python_callable=evaluate_production_model,
    )
    
    check_retraining_trigger_task = BranchPythonOperator(
        task_id='check_retraining_trigger',
        python_callable=check_retraining_trigger,
    )
    # This dummy marks the path where retraining is not needed.
    skip_retraining = DummyOperator(task_id='skip_retraining')
    
    # This dummy marks the start of the retraining path.
    trigger_retraining = DummyOperator(task_id='trigger_retraining')
    
    # Weekly training tasks using the same scripts as initial training
    train_lightgbm_weekly_task = PythonOperator(
        task_id='train_lightgbm_weekly',
        python_callable=train_lightgbm_weekly,
    )
    
    train_catboost_weekly_task = PythonOperator(
        task_id='train_catboost_weekly',
        python_callable=train_catboost_weekly,
    )
    
    # Extract results and select best model (same logic as initial)
    select_best_model_weekly_task = PythonOperator(
        task_id='select_best_model_weekly',
        python_callable=select_best_model_weekly,
    )
    
    # Register the best model and promote to Production (same logic as initial)
    register_model_weekly_task = PythonOperator(
        task_id='register_model_weekly',
        python_callable=register_model_weekly,
    )

    # === End of Pipeline ===
    # This task uses 'one_success' to ensure it runs as soon as any of the
    # terminal branches (skip, initial training, or weekly inference) completes.
    end = DummyOperator(
        task_id='end_pipeline',
        trigger_rule='one_success',
    )

    # --- Final Step for Weekly Lifecycle: Inference ---
    run_model_inference_task = PythonOperator(
        task_id='run_model_inference',
        python_callable=run_model_inference,
        trigger_rule='one_success',  # Run if either skip_retraining or register_model_weekly succeeds
    )

    # --- Define DAG Dependencies ---

    # The sensor is the first task, ensuring strict weekly ordering.
    wait_for_previous_run >> start >> start_preprocessing


    # Preprocessing feeds into the main gate
    start_preprocessing >> dep_check_source_credit_history
    start_preprocessing >> dep_check_source_demographic
    start_preprocessing >> dep_check_source_financial
    start_preprocessing >> dep_check_source_loan_terms
    
    dep_check_source_credit_history >> bronze_table_cred_history
    dep_check_source_demographic >> bronze_table_demographic
    dep_check_source_financial >> bronze_table_financial
    dep_check_source_loan_terms >> bronze_table_loan_term

    bronze_table_cred_history >> silver_table_cred_history
    bronze_table_demographic >> silver_table_demographic
    bronze_table_financial >> silver_table_financial
    bronze_table_loan_term >> silver_table_loan_term

    [silver_table_cred_history, silver_table_demographic, silver_table_financial, silver_table_loan_term] >> gold_feature_store
    silver_table_loan_term >> gold_label_store

    [gold_feature_store, gold_label_store] >> end_preprocessing

    end_preprocessing >> decide_pipeline_path_task

    # Path 1: Skip Logic
    decide_pipeline_path_task >> skip_run >> end

    # Path 2: Initial Training Logic (staggered)
    decide_pipeline_path_task >> run_initial_training_flow
    run_initial_training_flow >> train_lightgbm_initial >> train_catboost_initial >> select_best_model_initial_task >> register_model_initial_task >> end

    # Path 3: Weekly Lifecycle Logic (staggered)
    decide_pipeline_path_task >> run_weekly_lifecycle_flow
    run_weekly_lifecycle_flow >> evaluate_production_model_task >> check_retraining_trigger_task
    
    # Branching after evaluation
    check_retraining_trigger_task >> skip_retraining >> run_model_inference_task
    check_retraining_trigger_task >> trigger_retraining
    
    # Retraining sub-path (staggered)
    trigger_retraining >> train_lightgbm_weekly_task >> train_catboost_weekly_task >> select_best_model_weekly_task >> register_model_weekly_task
    register_model_weekly_task >> run_model_inference_task
    
    # Inference is the final step for the weekly path
    run_model_inference_task >> end