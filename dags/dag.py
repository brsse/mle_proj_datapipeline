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

sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts/utils'))
from data_processing_bronze_table import process_bronze_loan_table, process_bronze_clickstream_table, process_bronze_attributes_table, process_bronze_financials_table
from data_processing_silver_table import process_silver_table

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Helper functions for dependency management
def check_data_preprocessing_complete(**context):
    """
    Check if data preprocessing has completed for the current execution date
    """
    execution_date = context['execution_date']
    # Check if gold layer files exist for the execution date
    gold_feature_path = f"/opt/airflow/scripts/datamart/gold/features/feature_store_{execution_date.strftime('%Y_%m_%d')}.parquet"
    gold_label_path = f"/opt/airflow/scripts/datamart/gold/labels/gold_label_store_{execution_date.strftime('%Y_%m_%d')}.parquet"
    
    if os.path.exists(gold_feature_path) and os.path.exists(gold_label_path):
        return 'proceed_with_inference'
    else:
        return 'skip_inference'

def check_retraining_decision(**context):
    """
    Check if retraining is needed based on monitoring decision file
    """
    execution_date = context['execution_date']
    
    # Check if we have at least 1 year of data
    start_date = datetime(2022, 1, 1)
    if execution_date < start_date + timedelta(days=365):
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

# Data Preprocessing DAG
with DAG(
    'P2PCreditScore_DataPreprocessing',
    default_args=default_args,
    description='Daily data preprocessing pipeline for P2P credit scoring',
    schedule_interval='0 1 * * *',  # Daily at 1 AM
    start_date=datetime(2022, 1, 1),
    catchup=False,
    tags=['data', 'preprocessing', 'etl']
) as data_preprocessing_dag:
    
    # Start task
    start_task = DummyOperator(task_id="start_task")
    
    # Source data checks
    dep_check_source_lms = DummyOperator(task_id="dep_check_source_lms")
    dep_check_source_attributes = DummyOperator(task_id="dep_check_source_attributes")
    dep_check_source_financial = DummyOperator(task_id="dep_check_source_financial")
    dep_check_loan_term = DummyOperator(task_id="dep_check_loan_term")

    # Original source data checks (commented out)
    # dep_check_source_data_bronze_1 = FileSensor(
    #     task_id='dep_check_source_lms',
    #     filepath='/opt/airflow/scripts/data/lms_loan_daily.csv',
    #     poke_interval=10,
    #     timeout=600,
    #     mode='poke'
    # )
    # dep_check_source_data_bronze_2 = FileSensor(
    #     task_id='dep_check_source_attributes',
    #     filepath='/opt/airflow/scripts/data/features_attributes.csv',
    #     poke_interval=10,
    #     timeout=600,
    #     mode='poke'
    # )
    # dep_check_source_data_bronze_3 = FileSensor(
    #     task_id='dep_check_source_financials',
    #     filepath='/opt/airflow/scripts/data/features_financials.csv',
    #     poke_interval=10,
    #     timeout=600,
    #     mode='poke'
    # )
    # dep_check_source_data_bronze_4 = FileSensor(
    # task_id='dep_check_source_clickstream',
    # filepath='/opt/airflow/scripts/data/feature_clickstream.csv',
    # poke_interval=10,
    # timeout=600,
    # mode='poke'
    # )

    # Bronze layer processing
    bronze_table_cred_history = DummyOperator(task_id="bronze_table_cred_history")
    bronze_table_demographic = DummyOperator(task_id="bronze_table_demographic")
    bronze_table_financial = DummyOperator(task_id="bronze_table_financial")
    bronze_table_loan_term = DummyOperator(task_id="bronze_table_loan_term")
    
    # Original bronze table processing (commented out)
    # bronze_table_1 = PythonOperator(
    #     task_id='run_bronze_table_lms',
    #     python_callable=process_bronze_loan_table,
    #     op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/lms/']
    # )
    # bronze_table_2 = PythonOperator(
    #     task_id='run_bronze_table_attributes',
    #     python_callable=process_bronze_attributes_table,
    #     op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/attributes/']
    # )
    # bronze_table_3 = PythonOperator(
    #     task_id='run_bronze_table_financials',
    #     python_callable=process_bronze_financials_table,
    #     op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/financials/']
    # )
    # bronze_table_4 = PythonOperator(
    #     task_id='run_bronze_table_clickstream',
    #     python_callable=process_bronze_clickstream_table,
    #     op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/bronze/clickstream/']
    # )

    # Silver layer processing
    silver_table_cred_history = DummyOperator(task_id="silver_table_cred_history")
    silver_table_demographic = DummyOperator(task_id="silver_table_demographic")
    silver_table_financial = DummyOperator(task_id="silver_table_financial")
    silver_table_loan_term = DummyOperator(task_id="silver_table_loan_term")

    # Original silver table processing (commented out)
    # silver_table_1 = PythonOperator(
    #     task_id = 'run_silver_table_lms',
    #     python_callable = process_silver_table,
    #     op_args = ['lms', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/', '{{ ds }}']
    # )
    # silver_table_2 = PythonOperator(
    #     task_id = 'run_silver_table_attributes',
    #     python_callable = process_silver_table,
    #     op_args = ['attributes', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/', '{{ ds }}']
    # )
    # silver_table_3 = PythonOperator(
    #     task_id = 'run_silver_table_financials',
    #     python_callable = process_silver_table,
    #     op_args = ['financials', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/', '{{ ds }}']
    # )
    # silver_table_4 = PythonOperator(
    #     task_id = 'run_silver_table_clickstream',
    #     python_callable = process_silver_table,
    #     op_args = ['clickstream', '/opt/airflow/scripts/datamart/bronze/', '/opt/airflow/scripts/datamart/silver/', '{{ ds }}']
    # )

    # Gold layer processing
    gold_feature_store = DummyOperator(task_id="gold_feature_store")
    gold_label_store = DummyOperator(task_id="gold_label_store")

    # Original gold layer processing (commented out)
    # gold_feature_store = PythonOperator(
    #     task_id='run_gold_feature_store',
    #     python_callable=process_gold_feature_store,
    #     op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/silver/', '/opt/airflow/scripts/datamart/gold/']
    # )
    # gold_label_store = PythonOperator(
    #     task_id='run_gold_label_store',
    #     python_callable=process_gold_label_store,
    #     op_args=['{{ ds }}', '/opt/airflow/scripts/datamart/silver/', '/opt/airflow/scripts/datamart/gold/']
    # )

    # End task
    end_task = DummyOperator(task_id="end_task")

    # Define task dependencies
    start_task >> [dep_check_source_lms, dep_check_source_attributes, dep_check_source_financial, dep_check_loan_term]
    
    dep_check_source_lms >> bronze_table_cred_history
    dep_check_source_attributes >> bronze_table_demographic
    dep_check_source_financial >> bronze_table_financial
    dep_check_loan_term >> bronze_table_loan_term
    
    [bronze_table_cred_history, bronze_table_demographic, bronze_table_financial, bronze_table_loan_term] >> silver_table_cred_history
    [bronze_table_cred_history, bronze_table_demographic, bronze_table_financial, bronze_table_loan_term] >> silver_table_demographic
    [bronze_table_cred_history, bronze_table_demographic, bronze_table_financial, bronze_table_loan_term] >> silver_table_financial
    [bronze_table_cred_history, bronze_table_demographic, bronze_table_financial, bronze_table_loan_term] >> silver_table_loan_term
    
    [silver_table_cred_history, silver_table_demographic, silver_table_financial, silver_table_loan_term] >> gold_feature_store
    [silver_table_cred_history, silver_table_demographic, silver_table_financial, silver_table_loan_term] >> gold_label_store
    
    [gold_feature_store, gold_label_store] >> end_task

# Model Training and Inference DAG
with DAG(
    'P2PCreditScore_ModelTrainingInference',
    default_args=default_args,
    description='Daily model training and inference pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2022, 1, 1),
    catchup=False,
    tags=['ml', 'training', 'inference']
) as ml_inference_dag:
    
    # Wait for data preprocessing to complete
    wait_for_preprocessing = DummyOperator(task_id='wait_for_preprocessing')
    # wait_for_preprocessing = ExternalTaskSensor(
    #     task_id='wait_for_preprocessing',
    #     external_dag_id='P2PCreditScore_DataPreprocessing',
    #     external_task_id='end_task',
    #     timeout=3600,
    #     mode='reschedule',
    #     poke_interval=300
    # )
    
    # Check if we should proceed with inference
    check_inference_condition = DummyOperator(task_id='check_inference_condition')
    # check_inference_condition = BranchPythonOperator(
    #     task_id='check_inference_condition',
    #     python_callable=check_data_preprocessing_complete,
    #     provide_context=True
    # )
    
    # Skip inference if no data
    skip_inference = DummyOperator(task_id='skip_inference')
    
    # Check if retraining is needed (file-based decision)
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
    # train_model_task = PythonOperator(
    #     task_id='train_model',
    #     python_callable=train_model,
    #     op_kwargs={
    #         'snapshot_date_str': '{{ ds }}',
    #         'model_bank_directory': '/opt/airflow/model_bank/'
    #     }
    # )
    
    # Get model for inference
    get_model = DummyOperator(task_id='get_model')
    # get_model = PythonOperator(
    #     task_id='get_model_for_inference',
    #     python_callable=get_model_for_inference,
    #     provide_context=True
    # )
    
    # Inference task
    run_inference_task = DummyOperator(task_id='run_inference_task')
    # run_inference_task = PythonOperator(
    #     task_id='run_inference',
    #     python_callable=run_inference,
    #     op_kwargs={
    #         'snapshot_date_str': '{{ ds }}',
    #         'model_name': '{{ task_instance.xcom_pull(task_ids="get_model_for_inference", key="model_name") }}',
    #         'model_bank_directory': '/opt/airflow/model_bank/'
    #     }
    # )
    
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
    
    # End task
    end_task = DummyOperator(task_id='end_task', trigger_rule='one_success')
    
    # Define task dependencies
    wait_for_preprocessing >> check_inference_condition
    check_inference_condition >> [skip_inference, check_retraining_needed]
    
    check_retraining_needed >> [skip_retraining, train_model_task]
    skip_retraining >> get_model
    train_model_task >> get_model
    
    get_model >> run_inference_task >> clear_decision >> log_retraining >> end_task
    skip_inference >> end_task

# Model Monitoring DAG
with DAG(
    'P2PCreditScore_ModelMonitoring',
    default_args=default_args,
    description='Weekly model monitoring and performance tracking',
    schedule_interval='0 6 * * 0',  # Weekly on Sundays at 6 AM
    start_date=datetime(2022, 1, 1),
    catchup=False,
    tags=['ml', 'monitoring', 'performance']
) as ml_monitoring_dag:
    
    # Check if monitoring is needed
    check_monitoring = DummyOperator(task_id='check_monitoring')
    # check_monitoring = BranchPythonOperator(
    #     task_id='check_monitoring_needed',
    #     python_callable=check_if_monitoring_needed,
    #     provide_context=True
    # )
    
    # Skip monitoring if not enough data
    skip_monitoring = DummyOperator(task_id='skip_monitoring')
    
    # Get week end date
    get_week_end_date_task = DummyOperator(task_id='get_week_end_date')
    # get_week_end_date_task = PythonOperator(
    #     task_id='get_week_end_date',
    #     python_callable=get_week_end_date,
    #     provide_context=True
    # )
    
    # Get model for monitoring
    get_model_monitoring = DummyOperator(task_id='get_model_monitoring')
    # get_model_monitoring = BranchPythonOperator(
    #     task_id='get_model_for_monitoring',
    #     python_callable=get_model_for_monitoring,
    #     provide_context=True
    # )
    
    # Calculate metrics
    calculate_metrics = DummyOperator(task_id='calculate_metrics')
    # calculate_metrics = BranchPythonOperator(
    #     task_id='calculate_metrics',
    #     python_callable=calculate_metrics_task,
    #     provide_context=True
    # )
    
    # Skip report generation if no metrics
    skip_report = DummyOperator(task_id='skip_report')
    
    # Generate weekly report
    generate_report = DummyOperator(task_id='generate_weekly_report')
    # generate_report = PythonOperator(
    #     task_id='generate_weekly_report',
    #     python_callable=generate_weekly_report,
    #     op_kwargs={
    #         'week_end_date_str': '{{ task_instance.xcom_pull(task_ids="get_week_end_date", key="week_end_date") }}',
    #         'model_name': '{{ task_instance.xcom_pull(task_ids="get_model_for_monitoring", key="model_name") }}',
    #         'output_directory': '/opt/airflow/reports/'
    #     }
    # )
    
    # Write retraining decision (file-based approach)
    write_retraining_decision_task = DummyOperator(task_id='write_retraining_decision')
    # write_retraining_decision_task = PythonOperator(
    #     task_id='write_retraining_decision',
    #     python_callable=write_retraining_decision,
    #     provide_context=True
    # )
    
    # Send alerts (placeholder for email/Slack notifications)
    send_alerts = DummyOperator(task_id='send_alerts')
    # send_alerts = PythonOperator(
    #     task_id='send_alerts',
    #     python_callable=lambda **context: print("Sending alerts..."),
    #     provide_context=True
    # )
    
    # End task
    end_task = DummyOperator(task_id='end_task', trigger_rule='one_success')
    
    # Define task dependencies
    check_monitoring >> [skip_monitoring, get_week_end_date_task]
    
    get_week_end_date_task >> get_model_monitoring >> calculate_metrics
    
    calculate_metrics >> [skip_report, generate_report]
    
    generate_report >> write_retraining_decision_task >> send_alerts >> end_task
    skip_report >> end_task
    skip_monitoring >> end_task