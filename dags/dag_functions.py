"""
DAG Functions for ML Lifecycle Pipeline

This module contains all the Python functions used by the ML lifecycle pipeline DAG.
Separating these functions from the main DAG file improves maintainability and readability.
"""
import sys
import os
import json
import re
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import mlflow
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
sys.path.append('/opt/airflow/utils')
from model_inference_utils import (
    load_model_from_mlflow,
    load_weekly_data,
    prepare_features,
    evaluate_weekly_performance,
    save_metrics_to_postgres
)

PG_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'airflow',
    'user': 'airflow',
    'password': 'airflow'
}

FEATURE_STORE_PATH = "/opt/airflow/datamart/gold/feature_store"
LABEL_STORE_PATH = "/opt/airflow/datamart/gold/label_store"
TARGET_COLUMN = 'grade'
MODEL_TYPE = 'LightGBM'  # or 'CatBoost', parametrize as needed
MODEL_NAME = 'credit_scoring_model'  # or parametrize as needed

def decide_pipeline_path(**context):
    """
    Directs the pipeline based on the execution date.
    - Before 2023: Skip all model-related tasks.
    - On the first run of 2023: Trigger the dedicated initial model training flow.
    - After the first run of 2023: Run the standard weekly lifecycle.
    """
    logical_date = context["logical_date"]
    initial_training_date = datetime(2023, 1, 1)

    if logical_date.replace(tzinfo=None) < initial_training_date:
        return 'skip_run'
    elif logical_date.date() == initial_training_date.date():
        return 'run_initial_training_flow'
    else:
        return 'run_weekly_lifecycle_flow'


def check_retraining_trigger(**context):
    """
    Decide whether to trigger retraining based on macro F1 threshold and time since last retraining.
    Triggers retraining if macro F1 < 0.87 or 90 days have passed since last retraining.
    """
    # Pull metrics from XCom (from evaluate_production_model)
    metrics = context['ti'].xcom_pull(task_ids='evaluate_production_model')
    macro_f1 = None
    if metrics and isinstance(metrics, dict):
        macro_f1 = metrics.get("macro_f1")
    # Time-based check
    retraining_tracker_path = "/opt/airflow/logs/last_retraining_date.json"
    time_trigger = False
    days_since_retraining = None
    if os.path.exists(retraining_tracker_path):
        with open(retraining_tracker_path, 'r') as f:
            tracker_data = json.load(f)
            last_retraining_date = tracker_data.get('last_retraining_date')
            if last_retraining_date:
                from datetime import datetime, timedelta
                last_retraining_datetime = datetime.strptime(last_retraining_date, '%Y-%m-%d')
                logical_date = context['logical_date']
                # Make logical_date naive for subtraction
                logical_date_naive = logical_date.replace(tzinfo=None)
                days_since_retraining = (logical_date_naive - last_retraining_datetime).days
                if days_since_retraining >= 90:
                    time_trigger = True
    # Metric-based check
    metric_trigger = macro_f1 is not None and macro_f1 < 0.87
    # Combined logic
    if metric_trigger:
        print(f"[Trigger] Macro F1 {macro_f1:.4f} below threshold, triggering retraining.")
        return 'trigger_retraining'
    elif time_trigger:
        print(f"[Trigger] More than 90 days since last retraining, triggering retraining.")
        return 'trigger_retraining'
    else:
        print(f"[Trigger] No retraining needed. Macro F1: {macro_f1}, Days since retraining: {days_since_retraining if days_since_retraining is not None else 'N/A'}")
        return 'skip_retraining'


def extract_mlflow_run_id_from_logs(task_id: str = None, **context):
    """
    Extract MLflow run ID from the training script logs.
    This function parses the logs to find the MLflow run ID.
    
    Args:
        task_id: Specific task ID to extract from (if None, uses current task)
        context: Airflow context
    """
    # Get the task instance to access logs
    task_instance = context['task_instance']
    
    if task_id:
        # Get logs from a specific task
        log_content = task_instance.xcom_pull(task_ids=task_id, key='return_value')
        if not log_content:
            # Try to get logs directly
            try:
                log_content = task_instance.log.read(task_id=task_id)
            except:
                log_content = ""
    else:
        # Get the log content for the current task
        log_content = task_instance.log.read()
    
    # Look for MLflow run ID in the logs
    # The pattern might be something like "run_id: abc123" or similar
    run_id_pattern = r'run_id[:\s]+([a-f0-9]+)'
    match = re.search(run_id_pattern, log_content, re.IGNORECASE)
    
    if match:
        run_id = match.group(1)
        print(f"Extracted MLflow run ID from {task_id or 'current task'}: {run_id}")
        return run_id
    else:
        # If we can't find the run ID in logs, we'll need to query MLflow directly
        print(f"Could not extract run ID from {task_id or 'current task'} logs, will query MLflow directly")
        return None


def extract_metrics_from_logs(task_id: str = None, **context):
    """
    Extract performance metrics from the training script logs.
    
    Args:
        task_id: Specific task ID to extract from (if None, uses current task)
        context: Airflow context
    """
    task_instance = context['task_instance']
    
    if task_id:
        # Get logs from a specific task
        log_content = task_instance.xcom_pull(task_ids=task_id, key='return_value')
        if not log_content:
            # Try to get logs directly
            try:
                log_content = task_instance.log.read(task_id=task_id)
            except:
                log_content = ""
    else:
        log_content = task_instance.log.read()
    
    # Look for Macro F1 score in logs
    f1_pattern = r'Macro F1 Score[:\s]+([0-9.]+)'
    match = re.search(f1_pattern, log_content, re.IGNORECASE)
    
    if match:
        macro_f1 = float(match.group(1))
        print(f"Extracted Macro F1 Score from {task_id or 'current task'}: {macro_f1}")
        return macro_f1
    else:
        print(f"Could not extract Macro F1 score from {task_id or 'current task'} logs")
        return None


def query_mlflow_for_run_info(model_type: str = None, **context) -> Tuple[Optional[str], Optional[float]]:
    """
    Query MLflow to get run information and metrics.
    This is a fallback if we can't extract from logs.
    
    Args:
        model_type: Optional model type to filter results
        context: Airflow context
    """
    # Set up MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Get the current execution date
    logical_date = context['logical_date']
    
    # Query recent runs from the experiment
    experiment_name = "test"  # This should match what's used in the training scripts
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return None, None
    
    # Get runs from the last hour (should include our recent training run)
    start_time = int((logical_date - timedelta(hours=1)).timestamp() * 1000)  # ms since epoch
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"start_time >= {start_time}",
        max_results=10,  # Increased to catch both models
        order_by=["start_time DESC"]
    )
    
    if runs.empty:
        print("No recent runs found")
        return None, None
    
    # If model_type is specified, look for runs with that model type
    if model_type:
        # Look for runs with model type in the run name or parameters
        for _, run in runs.iterrows():
            run_name = run.get('tags.mlflow.runName', '')
            if model_type.lower() in run_name.lower():
                run_id = run['run_id']
                macro_f1 = run.get('metrics.macro_f1_score', None)
                print(f"Found {model_type} run: {run_id}, Macro F1: {macro_f1}")
                return run_id, macro_f1
    
    # If no specific model type or not found, return the most recent run
    latest_run = runs.iloc[0]
    run_id = latest_run['run_id']
    macro_f1 = latest_run.get('metrics.macro_f1_score', None)
    
    print(f"Found recent run: {run_id}, Macro F1: {macro_f1}")
    return run_id, macro_f1


def select_best_model_initial(**context):
    """
    Extract results from both training tasks and select the best model.
    This combines result extraction and model selection for efficiency.
    """
    print("Extracting results from both training tasks and selecting best model...")
    
    # Extract LightGBM results
    print("Extracting LightGBM results...")
    lightgbm_run_id = extract_mlflow_run_id_from_logs(task_id='train_lightgbm_initial', **context)
    lightgbm_f1 = extract_metrics_from_logs(task_id='train_lightgbm_initial', **context)
    
    # If extraction failed, query MLflow
    if lightgbm_run_id is None or lightgbm_f1 is None:
        print("LightGBM log extraction failed, querying MLflow directly...")
        lightgbm_run_id, lightgbm_f1 = query_mlflow_for_run_info(model_type='lightgbm', **context)
    
    # Extract CatBoost results
    print("Extracting CatBoost results...")
    catboost_run_id = extract_mlflow_run_id_from_logs(task_id='train_catboost_initial', **context)
    catboost_f1 = extract_metrics_from_logs(task_id='train_catboost_initial', **context)
    
    # If extraction failed, query MLflow
    if catboost_run_id is None or catboost_f1 is None:
        print("CatBoost log extraction failed, querying MLflow directly...")
        catboost_run_id, catboost_f1 = query_mlflow_for_run_info(model_type='catboost', **context)
    
    # Handle missing results
    if lightgbm_run_id is None:
        print("Warning: Could not get LightGBM run ID")
        lightgbm_run_id = "unknown"
        lightgbm_f1 = 0.0
    
    if catboost_run_id is None:
        print("Warning: Could not get CatBoost run ID")
        catboost_run_id = "unknown"
        catboost_f1 = 0.0
    
    if lightgbm_f1 is None:
        print("Warning: Could not get LightGBM Macro F1 score")
        lightgbm_f1 = 0.0
    
    if catboost_f1 is None:
        print("Warning: Could not get CatBoost Macro F1 score")
        catboost_f1 = 0.0
    
    # Push individual results to XComs for debugging/auditing
    context['task_instance'].xcom_push(key='lightgbm_run_id', value=lightgbm_run_id)
    context['task_instance'].xcom_push(key='catboost_run_id', value=catboost_run_id)
    context['task_instance'].xcom_push(key='lightgbm_macro_f1', value=lightgbm_f1)
    context['task_instance'].xcom_push(key='catboost_macro_f1', value=catboost_f1)
    
    print(f"LightGBM Run ID: {lightgbm_run_id}, Macro F1: {lightgbm_f1:.4f}")
    print(f"CatBoost Run ID: {catboost_run_id}, Macro F1: {catboost_f1:.4f}")
    
    # Select best model
    if lightgbm_f1 > catboost_f1:
        best_run_id = lightgbm_run_id
        best_model_type = "LightGBM"
        best_f1 = lightgbm_f1
    else:
        best_run_id = catboost_run_id
        best_model_type = "CatBoost"
        best_f1 = catboost_f1
    
    print(f"Selected {best_model_type} as best model with Macro F1: {best_f1:.4f}")
    
    # Push best model info to XComs
    context['task_instance'].xcom_push(key='best_run_id', value=best_run_id)
    context['task_instance'].xcom_push(key='best_model_type', value=best_model_type)
    context['task_instance'].xcom_push(key='best_macro_f1', value=best_f1)
    
    return f"Best model selected: {best_model_type} (Run ID: {best_run_id}, Macro F1: {best_f1:.4f})"


def register_model_initial(**context):
    """
    Register the best model to MLflow Model Registry and promote to Production.
    """
    # Get best model info from XComs
    best_run_id = context['task_instance'].xcom_pull(key='best_run_id')
    best_model_type = context['task_instance'].xcom_pull(key='best_model_type')
    best_f1 = context['task_instance'].xcom_pull(key='best_macro_f1')
    
    print(f"Registering {best_model_type} model (Run ID: {best_run_id}) to MLflow Registry")
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Get the model URI from the run
    model_uri = f"runs:/{best_run_id}/model"
    
    # Register the model
    model_name = "credit_scoring_model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    # Transition to Production
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    
    print(f"Model registered successfully: {model_name} v{model_version.version}")
    print(f"Model promoted to Production stage")
    
    # Update retraining tracker
    retraining_tracker_path = "/opt/airflow/logs/last_retraining_date.json"
    tracker_data = {
        'last_retraining_date': context['logical_date'].strftime('%Y-%m-%d'),
        'model_name': model_name,
        'model_version': model_version.version,
        'model_type': best_model_type,
        'macro_f1_score': best_f1,
        'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    os.makedirs(os.path.dirname(retraining_tracker_path), exist_ok=True)
    with open(retraining_tracker_path, 'w') as f:
        json.dump(tracker_data, f, indent=2)
    
    print(f"Retraining tracker updated: {retraining_tracker_path}")
    
    return f"Model registered and promoted to Production: {model_name} v{model_version.version}"


def evaluate_production_model(**context):
    """
    Pull the latest metric from Postgres for the previous week to evaluate model performance.
    If it's the week after ANY retraining event (initial or subsequent), just succeed.
    Returns a standard Python dict with metrics for XCom passing.
    """
    logical_date = context['logical_date']
    current_week_date = logical_date.strftime('%Y_%m_%d')
    # Get previous week's date for evaluation
    prev_week_date = (logical_date - timedelta(weeks=1)).strftime('%Y_%m_%d')
    print(f"[Eval] Evaluating production model performance for previous week: {prev_week_date}")

    # Check if this is the week after ANY retraining event
    retraining_tracker_path = "/opt/airflow/logs/last_retraining_date.json"
    if os.path.exists(retraining_tracker_path):
        with open(retraining_tracker_path, 'r') as f:
            tracker_data = json.load(f)
            last_retraining_date = tracker_data.get('last_retraining_date')
            if last_retraining_date:
                # Calculate the week after the last retraining
                last_retraining_datetime = datetime.strptime(last_retraining_date, '%Y-%m-%d')
                week_after_retraining = (last_retraining_datetime + timedelta(weeks=1)).strftime('%Y_%m_%d')
                if current_week_date == week_after_retraining:
                    print(f"[Eval] This is the week after the last retraining event ({last_retraining_date}). Skipping evaluation.")
                    return {"macro_f1": None, "message": "Week after retraining event, skipping evaluation."}

    # Query previous week's metrics from Postgres
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        query = """
            SELECT * FROM model_performance_metrics
            WHERE week_date = %s
            ORDER BY evaluation_date DESC
            LIMIT 1
        """
        cursor.execute(query, (prev_week_date,))
        result = cursor.fetchone()
        if not result:
            print(f"[Eval] No metrics found for previous week {prev_week_date}")
            return {"macro_f1": None, "message": f"No metrics for previous week {prev_week_date}"}
        print(f"[Eval] Found metrics for previous week {prev_week_date}:")
        print(f"  - Model: {result['model_name']}")
        print(f"  - Macro F1: {result['macro_f1']:.4f}")
        print(f"  - Accuracy: {result['accuracy']:.4f}")
        print(f"  - Total Samples: {result['total_samples']}")
        # Return as standard dict for XCom
        return {
            "macro_f1": result["macro_f1"],
            "model_name": result["model_name"],
            "accuracy": result["accuracy"],
            "total_samples": result["total_samples"],
            "message": "success"
        }
    except Exception as e:
        print(f"[Eval] Error querying Postgres: {e}")
        return {"macro_f1": None, "message": f"Error querying Postgres: {e}"}
    finally:
        if conn:
            conn.close()


def prepare_training_data_weekly(**context):
    """
    Load the rolling 50-week training data for weekly retraining.
    Should use the same reusable function as the initial data prep task, but with different date parameters.
    """
    # TODO: Implement weekly training data preparation
    # This would be similar to initial training but with rolling window
    print("Weekly training data preparation - to be implemented")
    return "Weekly training data prepared"


def check_data_availability(**context):
    """
    Check if there is at least 50 weeks of data.
    Logic can be adapted from utils/weekly_evaluation.py -> get_available_weeks().
    If data is insufficient, it should raise an AirflowSkipException.
    """
    # TODO: Implement data availability check
    print("Data availability check - to be implemented")
    return "Data availability check completed"


def run_model_inference(**context):
    """
    Run model inference for the current Airflow week using shared utils for full consistency with batch evaluation.
    """
    # The Airflow execution date is the start of the week (Sunday)
    # We want to run inference for this week
    week_date = context['logical_date'].strftime('%Y_%m_%d')
    model_name = context.get('params', {}).get('model_name', MODEL_NAME)
    model_type = context.get('params', {}).get('model_type', MODEL_TYPE)
    run_id = context.get('params', {}).get('run_id')

    print(f"[Inference] Airflow execution date: {context['logical_date']}")
    print(f"[Inference] Running inference for week: {week_date}")

    # 1. Get production model run_id from MLflow if not provided
    if not run_id:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not prod_versions:
            raise Exception("No Production model found in MLflow Model Registry!")
        run_id = prod_versions[0].run_id
        print(f"[Inference] Using Production model: {model_name}, run_id: {run_id}")

    # 2. Load model, mapping, features
    model, grade_mapping, feature_names = load_model_from_mlflow(run_id, model_type)
    if model is None:
        print(f"[Inference] Could not load model for run_id {run_id}")
        return

    # 3. Load data for this week
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("AirflowInference").getOrCreate()
    weekly_data = load_weekly_data(spark, week_date, FEATURE_STORE_PATH, LABEL_STORE_PATH)
    if weekly_data is None:
        print(f"[Inference] No data for week {week_date}")
        return

    # 4. Prepare features (but don't use the returned y for metrics)
    X, y_raw, _ = prepare_features(weekly_data, feature_names, TARGET_COLUMN)
    
    # 5. Encode y using grade_mapping from MLflow (not a new LabelEncoder)
    print(f"[DEBUG] y_raw data type: {y_raw.dtype}")
    print(f"[DEBUG] y_raw unique values: {y_raw.unique()}")
    print(f"[DEBUG] y_raw sample values: {y_raw.head(10).tolist()}")
    
    # Check if labels are already encoded (integers) or need encoding (strings)
    if y_raw.dtype in ['int64', 'int32', 'int16', 'int8'] or all(isinstance(x, (int, np.integer)) for x in y_raw.unique()):
        print(f"[DEBUG] Labels are already encoded as integers, using directly")
        y_encoded = y_raw.astype(int)
    else:
        print(f"[DEBUG] Labels are strings, encoding using grade_mapping")
        y_encoded = y_raw.map(grade_mapping)
        if y_encoded.isnull().any():
            print(f"[WARNING] Some labels in y could not be mapped using the grade mapping! Unmapped labels: {y_raw[y_encoded.isnull()].unique()}")
            y_encoded = y_encoded.fillna(-1).astype(int)
        else:
            y_encoded = y_encoded.astype(int)
    
    print(f"[DEBUG] Used grade_mapping: {grade_mapping}")
    print(f"[DEBUG] y_encoded value counts: {pd.Series(y_encoded).value_counts().to_dict()}")

    # 6. Evaluate using encoded labels
    metrics, y_pred = evaluate_weekly_performance(model, X, y_encoded, grade_mapping)

    # 7. Save metrics
    save_metrics_to_postgres(metrics, week_date, run_id, model_name)

    print(f"[Inference] Completed for week {week_date}")
    return f"Inference completed for week {week_date}"


def verify_grade_mapping_in_mlflow(run_id, client=None):
    """
    Utility to print and verify the grade mapping stored in MLflow for a given run_id.
    """
    import mlflow
    if client is None:
        client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
        grade_mapping_param = run.data.params.get('grade_mapping', '')
        if grade_mapping_param:
            grade_mapping = json.loads(grade_mapping_param)
            print(f"[VERIFY] grade_mapping param in MLflow for run {run_id}: {grade_mapping}")
        else:
            print(f"[VERIFY] No grade_mapping param found in MLflow for run {run_id}")
        # Optionally, check the artifact as well
        try:
            grade_mapping_artifact = mlflow.artifacts.load_dict(f"runs:/{run_id}/grade_mapping.json")
            print(f"[VERIFY] grade_mapping artifact in MLflow for run {run_id}: {grade_mapping_artifact}")
        except Exception as e:
            print(f"[VERIFY] Could not load grade_mapping artifact: {e}")
    except Exception as e:
        print(f"[VERIFY] Error accessing MLflow run {run_id}: {e}")


def decode_predictions(y_pred, grade_mapping):
    """
    Given a list/array of predicted indices and a grade_mapping (grade->idx),
    return the decoded grade labels.
    """
    reverse_grade_mapping = {v: k for k, v in grade_mapping.items()}
    return [reverse_grade_mapping.get(idx, "Unknown") for idx in y_pred]


def get_training_data_window(current_week_date):
    """
    Calculate the 50-week training data window ending on the previous week.
    This ensures we don't leak current week's data into training.
    """
    from datetime import datetime, timedelta
    
    # Parse current week date
    current_date = datetime.strptime(current_week_date, '%Y_%m_%d')
    
    # Training window ends on previous week
    training_end_date = current_date - timedelta(weeks=1)
    
    # Training window starts 50 weeks before the end date
    training_start_date = training_end_date - timedelta(weeks=49)  # 50 weeks total
    
    # Generate list of all week dates in the training window
    training_weeks = []
    current_week = training_start_date
    while current_week <= training_end_date:
        training_weeks.append(current_week.strftime('%Y_%m_%d'))
        current_week += timedelta(weeks=1)
    
    print(f"[Training] Data window: {len(training_weeks)} weeks from {training_start_date.strftime('%Y_%m_%d')} to {training_end_date.strftime('%Y_%m_%d')}")
    return training_weeks


def train_lightgbm_weekly(**context):
    """
    Train LightGBM model for weekly retraining using 50-week data window.
    """
    week_date = context['logical_date'].strftime('%Y_%m_%d')
    print(f"[Weekly Training] Starting LightGBM training for week: {week_date}")
    
    # Get training data window
    training_weeks = get_training_data_window(week_date)
    
    # Calculate the end date for training (previous week)
    from datetime import datetime, timedelta
    current_date = datetime.strptime(week_date, '%Y_%m_%d')
    training_end_date = current_date - timedelta(weeks=1)
    
    print(f"[Weekly Training] Training window: end_date={training_end_date.strftime('%Y-%m-%d')}, weeks=50")
    
    # Create a temporary wrapper script that modifies the training script's date
    import tempfile
    import shutil
    import os
    
    # Read the original training script
    original_script_path = "/opt/airflow/utils/LightGBM_training_run.py"
    with open(original_script_path, 'r') as f:
        script_content = f.read()
    
    # Replace the hardcoded date with our dynamic date
    old_date_line = "SNAPSHOT_DATE = datetime(2023, 1, 1)"
    new_date_line = f"SNAPSHOT_DATE = datetime({training_end_date.year}, {training_end_date.month}, {training_end_date.day})"
    
    modified_script_content = script_content.replace(old_date_line, new_date_line)
    
    # Create temporary script in the utils directory so it can find model_operations
    utils_dir = "/opt/airflow/utils"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=utils_dir, delete=False) as f:
        f.write(modified_script_content)
        temp_script_path = f.name
    
    try:
        # Call the modified training script from the utils directory
        import subprocess
        import sys
        
        process = subprocess.run(
            [sys.executable, os.path.basename(temp_script_path)],
            cwd=utils_dir,  # Run from utils directory
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"[Weekly Training] LightGBM training completed successfully")
        print(f"[Weekly Training] Output: {process.stdout}")
        
        return "LightGBM weekly training completed"
        
    except subprocess.CalledProcessError as e:
        print(f"[Weekly Training] LightGBM training failed: {e}")
        print(f"[Weekly Training] Error output: {e.stderr}")
        raise
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_script_path)
        except:
            pass


def train_catboost_weekly(**context):
    """
    Train CatBoost model for weekly retraining using 50-week data window.
    """
    week_date = context['logical_date'].strftime('%Y_%m_%d')
    print(f"[Weekly Training] Starting CatBoost training for week: {week_date}")
    
    # Get training data window
    training_weeks = get_training_data_window(week_date)
    
    # Calculate the end date for training (previous week)
    from datetime import datetime, timedelta
    current_date = datetime.strptime(week_date, '%Y_%m_%d')
    training_end_date = current_date - timedelta(weeks=1)
    
    print(f"[Weekly Training] Training window: end_date={training_end_date.strftime('%Y-%m-%d')}, weeks=50")
    
    # Create a temporary wrapper script that modifies the training script's date
    import tempfile
    import shutil
    import os
    
    # Read the original training script
    original_script_path = "/opt/airflow/utils/CatBoost_training_run.py"
    with open(original_script_path, 'r') as f:
        script_content = f.read()
    
    # Replace the hardcoded date with our dynamic date
    old_date_line = "SNAPSHOT_DATE = datetime(2023, 1, 1)"
    new_date_line = f"SNAPSHOT_DATE = datetime({training_end_date.year}, {training_end_date.month}, {training_end_date.day})"
    
    modified_script_content = script_content.replace(old_date_line, new_date_line)
    
    # Create temporary script in the utils directory so it can find model_operations
    utils_dir = "/opt/airflow/utils"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', dir=utils_dir, delete=False) as f:
        f.write(modified_script_content)
        temp_script_path = f.name
    
    try:
        # Call the modified training script from the utils directory
        import subprocess
        import sys
        
        process = subprocess.run(
            [sys.executable, os.path.basename(temp_script_path)],
            cwd=utils_dir,  # Run from utils directory
            capture_output=True,
            text=True,
            check=True
        )
        
        print(f"[Weekly Training] CatBoost training completed successfully")
        print(f"[Weekly Training] Output: {process.stdout}")
        
        return "CatBoost weekly training completed"
        
    except subprocess.CalledProcessError as e:
        print(f"[Weekly Training] CatBoost training failed: {e}")
        print(f"[Weekly Training] Error output: {e.stderr}")
        raise
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_script_path)
        except:
            pass


def select_best_model_weekly(**context):
    """
    Extract results from both weekly training tasks and select the best model.
    Same logic as initial training.
    """
    print("Extracting results from both weekly training tasks and selecting best model...")
    
    # Extract LightGBM results
    print("Extracting LightGBM results...")
    lightgbm_run_id = extract_mlflow_run_id_from_logs(task_id='train_lightgbm_weekly', **context)
    lightgbm_f1 = extract_metrics_from_logs(task_id='train_lightgbm_weekly', **context)
    
    # If extraction failed, query MLflow
    if lightgbm_run_id is None or lightgbm_f1 is None:
        print("LightGBM log extraction failed, querying MLflow directly...")
        lightgbm_run_id, lightgbm_f1 = query_mlflow_for_run_info(model_type='lightgbm', **context)
    
    # Extract CatBoost results
    print("Extracting CatBoost results...")
    catboost_run_id = extract_mlflow_run_id_from_logs(task_id='train_catboost_weekly', **context)
    catboost_f1 = extract_metrics_from_logs(task_id='train_catboost_weekly', **context)
    
    # If extraction failed, query MLflow
    if catboost_run_id is None or catboost_f1 is None:
        print("CatBoost log extraction failed, querying MLflow directly...")
        catboost_run_id, catboost_f1 = query_mlflow_for_run_info(model_type='catboost', **context)
    
    # Handle missing results
    if lightgbm_run_id is None:
        print("Warning: Could not get LightGBM run ID")
        lightgbm_run_id = "unknown"
        lightgbm_f1 = 0.0
    
    if catboost_run_id is None:
        print("Warning: Could not get CatBoost run ID")
        catboost_run_id = "unknown"
        catboost_f1 = 0.0
    
    if lightgbm_f1 is None:
        print("Warning: Could not get LightGBM Macro F1 score")
        lightgbm_f1 = 0.0
    
    if catboost_f1 is None:
        print("Warning: Could not get CatBoost Macro F1 score")
        catboost_f1 = 0.0
    
    # Push individual results to XComs for debugging/auditing
    context['task_instance'].xcom_push(key='lightgbm_run_id_weekly', value=lightgbm_run_id)
    context['task_instance'].xcom_push(key='catboost_run_id_weekly', value=catboost_run_id)
    context['task_instance'].xcom_push(key='lightgbm_macro_f1_weekly', value=lightgbm_f1)
    context['task_instance'].xcom_push(key='catboost_macro_f1_weekly', value=catboost_f1)
    
    print(f"LightGBM Run ID: {lightgbm_run_id}, Macro F1: {lightgbm_f1:.4f}")
    print(f"CatBoost Run ID: {catboost_run_id}, Macro F1: {catboost_f1:.4f}")
    
    # Select best model
    if lightgbm_f1 > catboost_f1:
        best_run_id = lightgbm_run_id
        best_model_type = "LightGBM"
        best_f1 = lightgbm_f1
    else:
        best_run_id = catboost_run_id
        best_model_type = "CatBoost"
        best_f1 = catboost_f1
    
    print(f"Selected {best_model_type} as best weekly model with Macro F1: {best_f1:.4f}")
    
    # Push best model info to XComs
    context['task_instance'].xcom_push(key='best_run_id_weekly', value=best_run_id)
    context['task_instance'].xcom_push(key='best_model_type_weekly', value=best_model_type)
    context['task_instance'].xcom_push(key='best_macro_f1_weekly', value=best_f1)
    
    return f"Best weekly model selected: {best_model_type} (Run ID: {best_run_id}, Macro F1: {best_f1:.4f})"


def register_model_weekly(**context):
    """
    Register the best weekly model to MLflow Model Registry and promote to Production.
    Same logic as initial registration but updates retraining tracker.
    """
    # Get best model info from XComs
    best_run_id = context['task_instance'].xcom_pull(key='best_run_id_weekly')
    best_model_type = context['task_instance'].xcom_pull(key='best_model_type_weekly')
    best_f1 = context['task_instance'].xcom_pull(key='best_macro_f1_weekly')
    
    print(f"Registering {best_model_type} weekly model (Run ID: {best_run_id}) to MLflow Registry")
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # Get the model URI from the run
    model_uri = f"runs:/{best_run_id}/model"
    
    # Register the model (creates new version)
    model_name = "credit_scoring_model"
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    # Transition to Production
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    
    print(f"Weekly model registered successfully: {model_name} v{model_version.version}")
    print(f"Model promoted to Production stage")
    
    # Update retraining tracker with current date (resets 90-day timer)
    retraining_tracker_path = "/opt/airflow/logs/last_retraining_date.json"
    tracker_data = {
        'last_retraining_date': context['logical_date'].strftime('%Y-%m-%d'),
        'model_name': model_name,
        'model_version': model_version.version,
        'model_type': best_model_type,
        'macro_f1_score': best_f1,
        'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'retraining_type': 'weekly'
    }
    
    os.makedirs(os.path.dirname(retraining_tracker_path), exist_ok=True)
    with open(retraining_tracker_path, 'w') as f:
        json.dump(tracker_data, f, indent=2)
    
    print(f"Retraining tracker updated: {retraining_tracker_path}")
    
    return f"Weekly model registered and promoted to Production: {model_name} v{model_version.version}"


# Future functions for weekly lifecycle (to be implemented) 