from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from airflow.operators.python import BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import sys
import os

sys.path.append('/opt/airflow/utils')
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

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Helper functions for dependency management
def check_previous_week_monitoring_complete(**context):
    # """
    # Check if the previous week's monitoring has completed before proceeding
    # This ensures we capture retraining decisions before moving forward
    # """
    # execution_date = context['execution_date']
    # previous_week = execution_date - timedelta(days=7)
    
    # # Check if this is the first week (no previous monitoring needed)
    # if execution_date < datetime(2022, 1, 8):  # First week after start date
    #     return 'proceed_with_processing'
    
    # # Check if previous week's monitoring completed
    # monitoring_completion_file = f"/opt/airflow/logs/monitoring_complete_{previous_week.strftime('%Y_%m_%d')}.txt"
    
    # if os.path.exists(monitoring_completion_file):
    #     print(f"Previous week's monitoring completed. Proceeding with processing.")
    #     return 'proceed_with_processing'
    # else:
    #     print(f"Previous week's monitoring not completed. Creating warning task.")
    #     return 'monitoring_warning'
    return 'start_preprocessingW'

def handle_monitoring_warning(**context):
    """
    Handle the case where previous week's monitoring didn't complete
    This task can be manually triggered to proceed or wait
    """
    execution_date = context['execution_date']
    previous_week = execution_date - timedelta(days=7)
    
    warning_message = f"""
    ⚠️  WARNING: Previous week's monitoring ({previous_week.strftime('%Y-%m-%d')}) did not complete!
    
    Options:
    1. Wait for monitoring to complete manually
    2. Force proceed (if monitoring was completed externally)
    3. Skip this week's processing
    
    To force proceed, create the file: /opt/airflow/logs/monitoring_complete_{previous_week.strftime('%Y_%m_%d')}.txt
    """
    
    print(warning_message)
    
    # Check again if the file was created manually
    monitoring_completion_file = f"/opt/airflow/logs/monitoring_complete_{previous_week.strftime('%Y_%m_%d')}.txt"
    
    if os.path.exists(monitoring_completion_file):
        print("Monitoring completion file found! Proceeding with processing.")
        return 'proceed_with_processing'
    else:
        print("Monitoring completion file not found. Skipping this week.")
        return 'skip_this_week'

def check_data_preprocessing_complete(**context):
    """
    Check if data preprocessing has completed for the current execution date
    """
    execution_date = context['execution_date']
    # Check if gold layer files exist for the execution date
    gold_feature_path = f"/opt/airflow/datamart/gold/feature_store/feature_store_week_{execution_date.strftime('%Y_%m_%d')}/"
    gold_label_path = f"/opt/airflow/datamart/gold/label_store/label_store_week_{execution_date.strftime('%Y_%m_%d')}/"
    
    if os.path.exists(gold_feature_path) and os.path.exists(gold_label_path):
        return 'proceed_with_inference'
    else:
        return 'skip_inference'

def check_initial_model_ready(**context):
    """
    Check if we have 50 weeks of processed data before allowing initial model training
    First run starts at Jan 2, 2022, so initial model should be ready around Dec 18, 2022
    """
    execution_date = context['execution_date']
    start_date = datetime(2022, 1, 2)  # First run date
    
    # Calculate weeks since start
    weeks_since_start = (execution_date - start_date).days // 7
    
    # Need 50 weeks of data
    required_weeks = 50
    
    if weeks_since_start >= required_weeks:
        print(f"Week {weeks_since_start}: Sufficient data ({weeks_since_start} weeks) for initial model training.")
        return 'ready_for_training'
    else:
        print(f"Week {weeks_since_start}: Insufficient data ({weeks_since_start}/{required_weeks} weeks) for initial model training.")
        return 'insufficient_data'

def check_retraining_decision(**context):
    """
    Check if retraining is needed based on monitoring decision file
    """
    execution_date = context['execution_date']
    
    # First check if we have enough data for initial model
    data_check = check_initial_model_ready(**context)
    if data_check == 'insufficient_data':
        return 'skip_retraining'
    
    # Check for retraining decision file
    retraining_decision_path = "/opt/airflow/logs/retraining_decision.json"
    
    if not os.path.exists(retraining_decision_path):
        return 'skip_retraining'
    
    try:
        import json
        with open(retraining_decision_path, 'r') as f:
            decision = json.load(f)
        
        # Check if decision is for this week and still valid
        decision_date = datetime.strptime(decision.get('decision_date', ''), '%Y-%m-%d')
        week_start = execution_date - timedelta(days=execution_date.weekday())
        decision_week_start = decision_date - timedelta(days=decision_date.weekday())
        
        if decision_week_start == week_start and decision.get('retrain', False):
            return 'proceed_with_retraining'
        else:
            return 'skip_retraining'
            
    except (json.JSONDecodeError, ValueError, KeyError):
        return 'skip_retraining'

def write_retraining_decision(**context):
    """
    Write retraining decision to file (called by monitoring DAG)
    """
    execution_date = context['execution_date']
    
    # This is where you would implement your monitoring logic
    # For now, let's simulate a decision based on performance metrics
    
    # Simulate performance check (replace with actual monitoring logic)
    import random
    performance_score = random.random()  # Replace with actual performance metric
    
    # Decision logic: retrain if performance is below threshold
    should_retrain = performance_score < 0.7  # 70% threshold
    
    decision = {
        'decision_date': execution_date.strftime('%Y-%m-%d'),
        'performance_score': performance_score,
        'retrain': should_retrain,
        'reason': f"Performance score {performance_score:.3f} {'below' if should_retrain else 'above'} threshold"
    }
    
    # Ensure logs directory exists
    os.makedirs('/opt/airflow/logs', exist_ok=True)
    
    # Write decision to file
    import json
    with open('/opt/airflow/logs/retraining_decision.json', 'w') as f:
        json.dump(decision, f, indent=2)
    
    print(f"Retraining decision written: {decision}")
    return should_retrain

def clear_retraining_decision(**context):
    """
    Clear retraining decision file after processing
    """
    retraining_decision_path = "/opt/airflow/logs/retraining_decision.json"
    
    if os.path.exists(retraining_decision_path):
        os.remove(retraining_decision_path)
        print("Retraining decision file cleared")
    
    return "Decision cleared"

def log_retraining_activity(**context):
    """
    Log that retraining occurred this week
    """
    execution_date = context['execution_date']
    week_start = execution_date - timedelta(days=execution_date.weekday())
    retraining_log_path = f"/opt/airflow/logs/retraining_log_{week_start.strftime('%Y_%m_%d')}.txt"
    
    with open(retraining_log_path, 'w') as f:
        f.write(f"Retraining completed on {execution_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return "Retraining logged"

def mark_monitoring_complete(**context):
    """
    Mark that monitoring has completed for this week
    This creates a file that the next week's processing will check
    """
    execution_date = context['execution_date']
    monitoring_completion_file = f"/opt/airflow/logs/monitoring_complete_{execution_date.strftime('%Y_%m_%d')}.txt"
    
    # Ensure logs directory exists
    os.makedirs('/opt/airflow/logs', exist_ok=True)
    
    with open(monitoring_completion_file, 'w') as f:
        f.write(f"Monitoring completed on {execution_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Monitoring completion marked: {monitoring_completion_file}")
    return "Monitoring completion marked"

# Unified DAG - Weekly P2P Credit Scoring Pipeline
with DAG(
    'P2PCreditScore_WeeklyPipeline',
    default_args=default_args,
    description='Weekly unified pipeline: preprocessing -> training/inference -> monitoring',
    schedule_interval='0 6 * * 0',  # Weekly on Sundays at 6 AM
    start_date=datetime(2022, 1, 16),
    end_date=datetime(2024, 12, 31),
    catchup=True,
    tags=['p2p', 'credit-scoring', 'ml-pipeline']
) as unified_dag:
    
    # === PHASE 1: DEPENDENCY CHECK ===
    # Check if previous week's monitoring completed
    check_previous_monitoring = DummyOperator(task_id='check_previous_monitoring')
    # check_previous_monitoring = BranchPythonOperator(
    #     task_id='check_previous_monitoring',
    #     python_callable=check_previous_week_monitoring_complete,
    #     provide_context=True
    # )
    
    # Handle monitoring warning (manual intervention point)
    monitoring_warning = DummyOperator(task_id='monitoring_warning')
    # monitoring_warning = BranchPythonOperator(
    #     task_id='monitoring_warning',
    #     python_callable=handle_monitoring_warning,
    #     provide_context=True
    # )
    
    # Skip this week if previous monitoring didn't complete
    skip_this_week = DummyOperator(task_id='skip_this_week')
    
    # === PHASE 2: DATA PREPROCESSING ===
    # Start preprocessing
    start_preprocessing = DummyOperator(task_id='start_preprocessing')
    
    # Source data checks
    dep_check_source_credit_history = DummyOperator(task_id='dep_check_source_credit_history')
    # dep_check_source_credit_history = FileSensor(
    #     task_id='dep_check_source_credit_history',
    #     filepath='/opt/airflow/data/features_credit_history.csv',
    #     poke_interval=10,
    #     timeout=600,
    #     mode='poke'
    # )
    
    dep_check_source_demographic = DummyOperator(task_id='dep_check_source_demographic')
    # dep_check_source_demographic = FileSensor(
    #     task_id='dep_check_source_demographic',
    #     filepath='/opt/airflow/data/features_demographic.csv',
    #     poke_interval=10,
    #     timeout=600,
    #     mode='poke'
    # )
    
    dep_check_source_financial = DummyOperator(task_id='dep_check_source_financial')
    # dep_check_source_financial = FileSensor(
    #     task_id='dep_check_source_financial',
    #     filepath='/opt/airflow/data/features_financial.csv',
    #     poke_interval=10,
    #     timeout=600,
    #     mode='poke'
    # )
    
    dep_check_source_loan_terms = DummyOperator(task_id='dep_check_source_loan_terms')
    # dep_check_source_loan_terms = FileSensor(
    #     task_id='dep_check_source_loan_terms',
    #     filepath='/opt/airflow/data/features_loan_terms.csv',
    #     poke_interval=10,
    #     timeout=600,
    #     mode='poke'
    # )

    # Bronze layer processing
    bronze_table_cred_history = DummyOperator(task_id='bronze_table_credit_history')
    # bronze_table_cred_history = PythonOperator(
    #     task_id='bronze_table_cred_history',
    #     python_callable=process_bronze_table,
    #     op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', None, 'credit_history', 'weekly']
    # )
    
    bronze_table_demographic = DummyOperator(task_id='bronze_table_demographic')
    # bronze_table_demographic = PythonOperator(
    #     task_id='bronze_table_demographic',
    #     python_callable=process_bronze_table,
    #     op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', None, 'demographic', 'weekly']
    # )
    
    bronze_table_financial = DummyOperator(task_id='bronze_table_financial')
    # bronze_table_financial = PythonOperator(
    #     task_id='bronze_table_financial',
    #     python_callable=process_bronze_table,
    #     op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', None, 'financial', 'weekly']
    # )
    
    bronze_table_loan_term = DummyOperator(task_id='bronze_table_loan_term')
    # bronze_table_loan_term = PythonOperator(
    #     task_id='bronze_table_loan_term',
    #     python_callable=process_bronze_table,
    #     op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', None, 'loan_terms', 'weekly']
    # )

    # Silver layer processing
    silver_table_cred_history = DummyOperator(task_id='silver_table_cred_history')
    # silver_table_cred_history = PythonOperator(
    #     task_id='silver_table_cred_history',
    #     python_callable=process_silver_table,
    #     op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', '/opt/airflow/datamart/silver/', 'credit_history', None]
    # )
    
    silver_table_demographic = DummyOperator(task_id='silver_table_demographic')
    # silver_table_demographic = PythonOperator(
    #     task_id='silver_table_demographic',
    #     python_callable=process_silver_table,
    #     op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', '/opt/airflow/datamart/silver/', 'demographic', None]
    # )
    
    silver_table_financial = DummyOperator(task_id='silver_table_financial')
    # silver_table_financial = PythonOperator(
    #     task_id='silver_table_financial',
    #     python_callable=process_silver_table,
    #     op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', '/opt/airflow/datamart/silver/', 'financial', None]
    # )
    
    silver_table_loan_term = DummyOperator(task_id='silver_table_loan_term')
    # silver_table_loan_term = PythonOperator(
    #     task_id='silver_table_loan_term',
    #     python_callable=process_silver_table,
    #     op_args=['{{ ds }}', '/opt/airflow/datamart/bronze/', '/opt/airflow/datamart/silver/', 'loan_terms', None]
    # )

    # Gold layer processing - FEATURE AND LABEL STORE (ONLY ACTIVE TASKS)
    gold_feature_store = DummyOperator(task_id='gold_feature_store')
    # gold_feature_store = PythonOperator(
    #     task_id='gold_feature_store',
    #     python_callable=create_feature_store,
    #     op_args=['/opt/airflow/datamart/silver/', '/opt/airflow/datamart/gold/', '{{ ds }}']
    # )
    
    gold_label_store = DummyOperator(task_id='gold_label_store')
    # gold_label_store = PythonOperator(
    #     task_id='gold_label_store',
    #     python_callable=create_gold_label_store,
    #     op_args=['/opt/airflow/datamart/silver/', '/opt/airflow/datamart/gold/', '{{ ds }}']
    # )
    
    # Preprocessing complete
    preprocessing_complete = DummyOperator(task_id='preprocessing_complete')
    
    # === PHASE 3: MODEL TRAINING/INFERENCE ===
    # Check if we have enough data for initial model training
    check_initial_model_ready_task = DummyOperator(task_id='check_initial_model_ready')
    # check_initial_model_ready_task = BranchPythonOperator(
    #     task_id='check_initial_model_ready',
    #     python_callable=check_initial_model_ready,
    #     provide_context=True
    # )
    
    # Skip if insufficient data
    insufficient_data = DummyOperator(task_id='insufficient_data')
    
    # Check if we should proceed with inference
    check_inference_condition = DummyOperator(task_id='check_inference_condition')
    # check_inference_condition = BranchPythonOperator(
    #     task_id='check_inference_condition',
    #     python_callable=check_data_preprocessing_complete,
    #     provide_context=True
    # )
    
    # Skip inference if no data
    skip_inference = DummyOperator(task_id='skip_inference')
    
    # Check if retraining is needed
    check_retraining_needed = DummyOperator(task_id='check_retraining_needed')
    # check_retraining_needed = BranchPythonOperator(
    #     task_id='check_retraining_needed',
    #     python_callable=check_retraining_decision,
    #     provide_context=True
    # )
    
    # Skip retraining if not needed
    skip_retraining = DummyOperator(task_id='skip_retraining')
    
    # Training tasks
    train_model_task = DummyOperator(task_id='train_model_task')
    
    # Get model for inference
    get_model = DummyOperator(task_id='get_model')
    
    # Inference task
    run_inference_task = DummyOperator(task_id='run_inference_task')
    
    # Clear retraining decision after processing
    clear_decision = DummyOperator(task_id='clear_retraining_decision')
    # clear_decision = PythonOperator(
    #     task_id='clear_retraining_decision',
    #     python_callable=clear_retraining_decision,
    #     provide_context=True
    # )
    
    # Log retraining activity if retraining occurred
    log_retraining = DummyOperator(task_id='log_retraining_activity')
    # log_retraining = PythonOperator(
    #     task_id='log_retraining_activity',
    #     python_callable=log_retraining_activity,
    #     provide_context=True,
    #     trigger_rule='one_success'
    # )
    
    # Training/Inference complete
    training_inference_complete = DummyOperator(task_id='training_inference_complete', trigger_rule='one_success')
    
    # === PHASE 4: MODEL MONITORING ===
    # Check if monitoring is needed
    check_monitoring = DummyOperator(task_id='check_monitoring')
    
    # Skip monitoring if not enough data
    skip_monitoring = DummyOperator(task_id='skip_monitoring')
    
    # Get week end date
    get_week_end_date_task = DummyOperator(task_id='get_week_end_date')
    
    # Get model for monitoring
    get_model_monitoring = DummyOperator(task_id='get_model_monitoring')
    
    # Calculate metrics
    calculate_metrics = DummyOperator(task_id='calculate_metrics')
    
    # Skip report generation if no metrics
    skip_report = DummyOperator(task_id='skip_report')
    
    # Generate weekly report
    generate_report = DummyOperator(task_id='generate_weekly_report')
    
    # Write retraining decision
    write_retraining_decision_task = DummyOperator(task_id='write_retraining_decision')
    # write_retraining_decision_task = PythonOperator(
    #     task_id='write_retraining_decision',
    #     python_callable=write_retraining_decision,
    #     provide_context=True
    # )
    
    # Send alerts
    send_alerts = DummyOperator(task_id='send_alerts')
    
    # Mark monitoring as complete (CRITICAL - creates file for next week's dependency check)
    mark_monitoring_complete_task = DummyOperator(task_id='mark_monitoring_complete')
    # mark_monitoring_complete_task = PythonOperator(
    #     task_id='mark_monitoring_complete',
    #     python_callable=mark_monitoring_complete,
    #     provide_context=True
    # )
    
    # End task
    end_task = DummyOperator(task_id='end_task', trigger_rule='one_success')
    
    # === DEFINE TASK DEPENDENCIES ===
    
    # Phase 1: Dependency check
    check_previous_monitoring >> [monitoring_warning, start_preprocessing]
    monitoring_warning >> [skip_this_week, start_preprocessing]
    
    # Phase 2: Data preprocessing
    start_preprocessing >> [dep_check_source_credit_history, dep_check_source_demographic, dep_check_source_financial, dep_check_source_loan_terms]
    
    dep_check_source_credit_history >> bronze_table_cred_history
    dep_check_source_demographic >> bronze_table_demographic
    dep_check_source_financial >> bronze_table_financial
    dep_check_source_loan_terms >> bronze_table_loan_term
    
    # Silver processing after bronze processing
    bronze_table_cred_history >> silver_table_cred_history
    bronze_table_demographic >> silver_table_demographic
    bronze_table_financial >> silver_table_financial
    bronze_table_loan_term >> silver_table_loan_term
    
    # Gold processing after silver processing
    silver_table_cred_history >> gold_feature_store
    silver_table_demographic >> gold_feature_store
    silver_table_financial >> gold_feature_store
    silver_table_loan_term >> gold_feature_store
    silver_table_loan_term >> gold_label_store
    
    [gold_feature_store, gold_label_store] >> preprocessing_complete
    
    # Phase 3: Training/Inference
    preprocessing_complete >> check_initial_model_ready_task
    check_initial_model_ready_task >> [insufficient_data, check_inference_condition]
    
    check_inference_condition >> [skip_inference, check_retraining_needed]
    
    check_retraining_needed >> [skip_retraining, train_model_task]
    skip_retraining >> get_model
    train_model_task >> get_model
    
    get_model >> run_inference_task >> clear_decision >> log_retraining >> training_inference_complete
    skip_inference >> training_inference_complete
    insufficient_data >> training_inference_complete
    
    # Phase 4: Monitoring
    training_inference_complete >> check_monitoring
    check_monitoring >> [skip_monitoring, get_week_end_date_task]
    
    get_week_end_date_task >> get_model_monitoring >> calculate_metrics
    
    calculate_metrics >> [skip_report, generate_report]
    
    generate_report >> write_retraining_decision_task >> send_alerts >> mark_monitoring_complete_task >> end_task
    skip_report >> mark_monitoring_complete_task >> end_task
    skip_monitoring >> mark_monitoring_complete_task >> end_task
    
    # Skip path
    skip_this_week >> end_task