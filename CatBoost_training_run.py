import pandas as pd
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import os
from typing import List
from pyspark.sql import SparkSession
import mlflow
import mlflow.catboost
import json

# It's good practice to import your project's modules with a sys.path append
# to ensure they can be found.
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from model_operations import load_data_for_training, train_and_tune_model

# --- Configuration ---
SNAPSHOT_DATE = datetime(2023, 1, 1)
TRAINING_WEEKS = 50
# Use absolute paths to ensure the script runs from any directory
FEATURE_STORE_PATH = "/opt/airflow/datamart/gold/feature_store"
LABEL_STORE_PATH = "/opt/airflow/datamart/gold/label_store"
TARGET_COLUMN = 'grade'
UNIQUE_ID_COLUMN = 'id'

# MLflow configuration
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "credit_scoring_pipeline_catboost"

def get_date_range_for_training(end_date: datetime, num_weeks: int) -> List[str]:
    """
    Calculates the list of weekly partition strings for data loading.
    It assumes partitions are named directly with the execution date (e.g., a Sunday).
    """
    weeks = []
    # Loop backwards from the end_date for the specified number of weeks.
    for i in range(num_weeks):
        # Each step back is exactly 7 days.
        partition_date = end_date - timedelta(weeks=i)
        weeks.append(partition_date.strftime('%Y_%m_%d'))
        
    # The list is already chronological, but sorting ensures it.
    return sorted(weeks)

def save_model_with_mlflow(model, X_train, y_train, grade_mapping, metrics, model_name="baseline_model"):
    """
    Save the trained model and its metrics using MLflow.
    """
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Set or create experiment
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log model parameters
            mlflow.log_params({
                "model_type": "CatBoost",
                "objective": "MultiClass",
                "num_classes": len(grade_mapping),
                "training_samples": len(X_train),
                "features_count": X_train.shape[1],
                "snapshot_date": SNAPSHOT_DATE.strftime('%Y-%m-%d'),
                "training_weeks": TRAINING_WEEKS
            })
            
            # Log the grade mapping and feature names
            mlflow.log_param("grade_mapping", json.dumps(grade_mapping))
            mlflow.log_param("feature_names", json.dumps(list(X_train.columns)))

            # Log metrics
            flat_metrics = {
                'accuracy': metrics['accuracy'],
                'macro_f1': metrics['macro_f1'],
                'weighted_f1': metrics['weighted_f1'],
            }
            for grade, f1 in metrics['f1_by_grade'].items():
                flat_metrics[f'f1_grade_{grade}'] = f1
            
            mlflow.log_metrics(flat_metrics)
            
            # Log the model
            mlflow.catboost.log_model(model, "model")
            
            print(f"Model and metrics saved to MLflow with run_id: {mlflow.active_run().info.run_id}")
            return mlflow.active_run().info.run_id
            
    except Exception as e:
        print(f"Error saving model to MLflow: {e}")
        return None

def evaluate_model_performance(model, X_test, y_test, grade_mapping):
    """
    Evaluate model performance and return metrics including Macro F1 Score.
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Get per-class F1 scores
    f1_per_class = f1_score(y_test, y_pred, average=None)
    f1_by_grade = {grade_mapping[i]: score for i, score in enumerate(f1_per_class)}
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'f1_by_grade': f1_by_grade
    }
    
    return metrics, y_pred

def main():
    """
    Main function to execute a standalone model training and tuning run using CatBoost.
    """
    print("--- Starting Standalone Training Run with CatBoost ---")
    
    # 1. Initialize Spark with more memory for the driver
    spark = SparkSession.builder \
        .appName("StandaloneTrainingCatBoost") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # 2. Get the last 50 weeks of data
    training_weeks = get_date_range_for_training(SNAPSHOT_DATE, TRAINING_WEEKS)
    print(f"Loading data for {len(training_weeks)} weeks, from {training_weeks[0]} to {training_weeks[-1]}")
    
    try:
        full_df = load_data_for_training(spark, FEATURE_STORE_PATH, LABEL_STORE_PATH, training_weeks)
        print(f"Successfully loaded {full_df.shape[0]} records.")
    except Exception as e:
        print(f"ERROR: Could not load data. Please check paths and data availability. Details: {e}")
        spark.stop()
        return

    # --- Multiclass Target Preparation ---
    label_encoder = LabelEncoder()
    full_df['grade_encoded'] = label_encoder.fit_transform(full_df[TARGET_COLUMN])
    
    # Store the mapping from number back to letter grade for later interpretation
    grade_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    print(f"Grade mapping created: {grade_mapping}")

    # Define features (X) and target (y)
    features_to_drop = [
        UNIQUE_ID_COLUMN, 
        TARGET_COLUMN, 
        'snapshot_date', 
        'earliest_cr_date',
        'snapshot_month',
        'earliest_cr_month',
        'months_since_earliest_cr_line'
    ]
    X = full_df.drop(columns=features_to_drop + ['grade_encoded'], errors='ignore')
    y = full_df['grade_encoded']

    # 3. Do train-test split (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into {len(X_train)} training records and {len(X_test)} testing records.")

    # 4. Create a baseline CatBoost multiclass model
    print("\n--- Training Baseline CatBoost Multiclass Model ---")
    baseline_model = cb.CatBoostClassifier(
        objective='MultiClass',
        classes_count=len(grade_mapping),
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=42,
        verbose=False,
        iterations=100,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        colsample_bylevel=0.8,
        task_type='CPU'
    )
    baseline_model.fit(X_train, y_train)
    
    # 5. Evaluate model performance
    print("\n--- Evaluating Model Performance ---")
    metrics, y_pred_baseline = evaluate_model_performance(baseline_model, X_test, y_test, grade_mapping)
    
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}")
    
    print("\nF1 Score by Grade:")
    for grade, f1 in metrics['f1_by_grade'].items():
        print(f"  Grade {grade}: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_baseline))
    
    print("\nClassification Report:")
    target_names = [grade_mapping[i] for i in sorted(grade_mapping.keys())]
    print(classification_report(y_test, y_pred_baseline, target_names=target_names))

    # 6. Save model with MLflow
    print("\n--- Saving Model to MLflow ---")
    run_id = save_model_with_mlflow(baseline_model, X_train, y_train, grade_mapping, metrics, "baseline_credit_scoring_catboost")
    
    if run_id:
        print(f"Model successfully saved to MLflow with run_id: {run_id}")
        print(f"Key metric - Macro F1 Score: {metrics['macro_f1']:.4f}")
        print(f"Access MLflow UI at: http://localhost:5000")
    else:
        print("Failed to save model to MLflow")

    # 7. Save metrics for future weekly comparisons
    metrics_data = {
        'run_id': run_id,
        'model_name': 'baseline_credit_scoring_catboost',
        'training_date': datetime.now().isoformat(),
        'snapshot_date': SNAPSHOT_DATE.isoformat(),
        'training_weeks': TRAINING_WEEKS,
        'metrics': metrics,
        'grade_mapping': grade_mapping,
        'feature_names': list(X_train.columns)
    }
    
    # Save metrics to a JSON file for now (later we'll use PostgreSQL)
    metrics_file = f"/opt/airflow/model_bank/catboost_baseline_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("/opt/airflow/model_bank", exist_ok=True)
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"Metrics saved to: {metrics_file}")
    
    print("\n--- Next Steps for Weekly Evaluation ---")
    print("1. Baseline model saved to MLflow")
    print("2. Macro F1 Score baseline: {:.4f}".format(metrics['macro_f1']))
    print("3. For weekly evaluation:")
    print("   - Load the saved model from MLflow")
    print("   - Evaluate on new week's data")
    print("   - Compare Macro F1 Score with baseline")
    print("   - Store results in PostgreSQL for trend analysis")
    
    spark.stop()

if __name__ == "__main__":
    main() 