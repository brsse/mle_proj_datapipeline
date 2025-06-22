import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any, Tuple
from pyspark.sql import SparkSession
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import json
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuration
FEATURE_STORE_PATH = "/opt/airflow/datamart/gold/feature_store"
LABEL_STORE_PATH = "/opt/airflow/datamart/gold/label_store"
TARGET_COLUMN = 'grade'
UNIQUE_ID_COLUMN = 'id'
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "test"

# PostgreSQL configuration (for future metric storage)
PG_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'airflow',
    'user': 'airflow',
    'password': 'airflow'
}

def load_model_from_mlflow(run_id: str, model_type: str):
    """
    Load a model from MLflow using the run_id and model type.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Load the model - all models are logged as sklearn models
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        
        # Load the run to get parameters
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        # Debug: Print all available parameters
        print(f"  Debug - All MLflow parameters for run {run_id}:")
        for key, value in run.data.params.items():
            print(f"    {key}: {value}")
        
        # Debug: Print all available tags
        print(f"  Debug - All MLflow tags for run {run_id}:")
        for key, value in run.data.tags.items():
            print(f"    {key}: {value}")
        
        # Extract parameters
        feature_names = json.loads(run.data.params.get('feature_names', '[]'))
        
        # Try to load grade mapping as parameter first, then fall back to JSON file
        grade_mapping = {}
        try:
            # First try to get it as a parameter (newer models)
            grade_mapping_param = run.data.params.get('grade_mapping', '')
            if grade_mapping_param:
                grade_mapping = json.loads(grade_mapping_param)
                print(f"  Debug - Loaded grade_mapping from parameter: {grade_mapping}")
            else:
                # Fall back to JSON file (older models)
                grade_mapping_uri = f"runs:/{run_id}/grade_mapping.json"
                grade_mapping = mlflow.artifacts.load_dict(grade_mapping_uri)
                print(f"  Debug - Loaded grade_mapping from JSON file: {grade_mapping}")
        except Exception as e:
            print(f"  Warning: Could not load grade_mapping: {e}")
            grade_mapping = {}
        
        # If feature_names is empty (old models), we'll handle this in prepare_features
        if not feature_names:
            print(f"⚠️  No feature_names found in MLflow run {run_id}. Will use all available features.")
        
        print(f"✅ Successfully loaded {model_type} model from MLflow run: {run_id}")
        return model, grade_mapping, feature_names
        
    except Exception as e:
        print(f"❌ Error loading {model_type} model from MLflow: {e}")
        return None, None, None

def find_latest_run_id(model_name_prefix: str) -> str:
    """
    Find the run_id of the most recent successful run for a given model prefix.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            filter_string=f"tags.mlflow.runName LIKE '{model_name_prefix}%' AND status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1
        )
        if not runs.empty:
            run_id = runs.iloc[0]['run_id']
            print(f"🔍 Found latest run for '{model_name_prefix}': {run_id}")
            return run_id
        else:
            print(f"⚠️ No finished runs found for model prefix: '{model_name_prefix}'")
            return None
    except Exception as e:
        print(f"❌ Error searching for runs for '{model_name_prefix}': {e}")
        return None

def load_all_baseline_models():
    """
    Load all available baseline models from MLflow by finding the latest run for each.
    Returns a dictionary with model info for each successfully loaded model.
    """
    models = {}
    
    model_configs = {
        # "XGBoost": {"model_name_prefix": "xgboost_baseline", "type": "XGBoost"},  # Disabled
        "LightGBM": {"model_name_prefix": "lightgbm_baseline", "type": "LightGBM"},
        "CatBoost": {"model_name_prefix": "catboost_baseline", "type": "CatBoost"}
    }
    
    print(f"📊 Loading available baseline models from MLflow...")
    
    for model_key, config in model_configs.items():
        print(f"\n--- Searching for {model_key} Model ---")
        
        run_id = find_latest_run_id(config["model_name_prefix"])
        
        if run_id:
            model, grade_mapping, feature_names = load_model_from_mlflow(run_id, config["type"])
            if model:
                models[model_key] = {
                    "model": model,
                    "grade_mapping": grade_mapping,
                    "feature_names": feature_names,
                    "run_id": run_id,
                    "model_name": config["model_name_prefix"],
                    "type": config["type"]
                }
                print(f"✅ {model_key} model loaded successfully from run {run_id}")
        else:
            print(f"⚠️  Could not find a run for {model_key}, skipping.")

    if not models:
        print("\n❌ No models could be loaded from MLflow.")
        print("💡 Please ensure training scripts have been run successfully.")
        print("   python /opt/airflow/utils/train_all_models.py")
        return None
    
    print(f"\n✅ Successfully loaded {len(models)} models: {', '.join(models.keys())}")
    return models

def load_weekly_data(spark: SparkSession, week_date: str) -> pd.DataFrame:
    """
    Load data for a specific week using the correct directory structure.
    """
    try:
        # Use the correct directory structure
        feature_path = f"/opt/airflow/datamart/gold/feature_store/feature_store_week_{week_date}"
        label_path = f"/opt/airflow/datamart/gold/label_store/label_store_week_{week_date}"
        
        # Check if directories exist
        if not os.path.exists(feature_path):
            print(f"❌ Feature directory not found: {feature_path}")
            return None
        if not os.path.exists(label_path):
            print(f"❌ Label directory not found: {label_path}")
            return None
        
        # Load feature and label data for the specific week
        feature_df = spark.read.parquet(feature_path)
        label_df = spark.read.parquet(label_path)
        
        # Join features and labels
        full_df = feature_df.join(label_df, on='id', how='inner')
        
        # Convert to pandas
        pandas_df = full_df.toPandas()
        
        print(f"✅ Loaded {len(pandas_df)} records for week {week_date}")
        return pandas_df
        
    except Exception as e:
        print(f"❌ Error loading data for week {week_date}: {e}")
        return None

def prepare_features(df: pd.DataFrame, feature_names: List[str], target_column: str = 'grade'):
    """
    Prepare features for model evaluation using the same logic as training.
    """
    # Create a copy to avoid PerformanceWarning on fragmented DataFrame
    df = df.copy()

    # Encode the target variable
    label_encoder = LabelEncoder()
    df['grade_encoded'] = label_encoder.fit_transform(df[target_column])
    
    # Use the same features_to_drop logic as in training
    features_to_drop = [
        'id',  # UNIQUE_ID_COLUMN
        target_column,  # TARGET_COLUMN
        'snapshot_date', 
        'earliest_cr_date',
        'snapshot_month',
        'earliest_cr_month',
        'months_since_earliest_cr_line'
    ]
    
    # Drop the same columns as in training
    df_cleaned = df.drop(columns=features_to_drop + ['grade_encoded'], errors='ignore')
    
    # If feature_names is empty (old models), use all available features
    if not feature_names:
        print(f"⚠️  Using all available features: {len(df_cleaned.columns)} features")
        feature_names = list(df_cleaned.columns)
    else:
        # Ensure we have all required features
        missing_features = set(feature_names) - set(df_cleaned.columns)
        if missing_features:
            print(f"⚠️  Missing features: {missing_features}")
            # Add missing features with default values more efficiently
            missing_cols_dict = {feature: 0 for feature in missing_features}
            df_cleaned = df_cleaned.assign(**missing_cols_dict)
    
    # Select only the features used in training
    X = df_cleaned[feature_names]
    y = df['grade_encoded']
    
    print(f"✅ Prepared {X.shape[1]} features for evaluation")
    return X, y, label_encoder

def evaluate_weekly_performance(model, X, y, grade_mapping):
    """
    Evaluate model performance on weekly data.
    """
    y_pred = model.predict(X)

    # Handle CatBoost's 2D output format, which causes errors with pandas/sklearn
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')
    weighted_f1 = f1_score(y, y_pred, average='weighted')
    
    # Get per-class F1 scores
    f1_per_class = f1_score(y, y_pred, average=None)
    
    # Debug: Print what we have
    print(f"  Debug - grade_mapping: {grade_mapping}")
    print(f"  Debug - f1_per_class length: {len(f1_per_class)}")
    print(f"  Debug - unique predictions: {sorted(set(y_pred))}")
    print(f"  Debug - unique actual: {sorted(set(y))}")
    
    # Create reverse mapping: index -> grade
    # grade_mapping is {grade: index}, so we need {index: grade}
    reverse_grade_mapping = {idx: grade for grade, idx in grade_mapping.items()}
    print(f"  Debug - reverse_grade_mapping: {reverse_grade_mapping}")
    
    # Handle potential missing keys by using available indices
    f1_by_grade = {}
    for i, score in enumerate(f1_per_class):
        if i in reverse_grade_mapping:
            f1_by_grade[reverse_grade_mapping[i]] = score
        else:
            print(f"  Warning: Index {i} not found in reverse_grade_mapping, using 'Unknown_{i}'")
            f1_by_grade[f'Unknown_{i}'] = score
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'f1_by_grade': f1_by_grade,
        'total_samples': len(y),
        'predictions_distribution': pd.Series(y_pred).value_counts().to_dict()
    }
    
    return metrics, y_pred

def save_metrics_to_postgres(metrics: Dict[str, Any], week_date: str, run_id: str, model_name: str):
    """
    Save weekly evaluation metrics to PostgreSQL for a specific model.
    """
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Create table if it doesn't exist
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS model_performance_metrics (
            id SERIAL PRIMARY KEY,
            evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            week_date VARCHAR(10),
            mlflow_run_id VARCHAR(50),
            model_name VARCHAR(100),
            accuracy DECIMAL(5,4),
            macro_f1 DECIMAL(5,4),
            weighted_f1 DECIMAL(5,4),
            total_samples INTEGER,
            f1_by_grade JSONB,
            predictions_distribution JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(create_table_sql)
        
        # Insert metrics
        insert_sql = """
        INSERT INTO model_performance_metrics 
        (week_date, mlflow_run_id, model_name, accuracy, macro_f1, weighted_f1, 
         total_samples, f1_by_grade, predictions_distribution)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_sql, (
            week_date,
            run_id,
            model_name,
            metrics['accuracy'],
            metrics['macro_f1'],
            metrics['weighted_f1'],
            metrics['total_samples'],
            json.dumps(metrics['f1_by_grade']),
            json.dumps(metrics['predictions_distribution'])
        ))
        
        conn.commit()
        print(f"✅ Metrics saved to PostgreSQL for {model_name} - week {week_date}")
        
    except Exception as e:
        print(f"❌ Error saving to PostgreSQL: {e}")
    finally:
        if conn:
            conn.close()

def get_available_weeks(start_date: str = "2023_01_08") -> List[str]:
    """
    Get all available weeks from the feature store directory names starting from start_date.
    """
    try:
        # List all directories in the feature store
        feature_store_dir = "/opt/airflow/datamart/gold/feature_store"
        if not os.path.exists(feature_store_dir):
            print(f"❌ Feature store directory not found: {feature_store_dir}")
            return []
        
        # Get all week directories
        week_dirs = [d for d in os.listdir(feature_store_dir) if d.startswith('feature_store_week_')]
        
        # Extract week dates from directory names
        available_weeks = []
        for week_dir in week_dirs:
            # Extract date from "feature_store_week_YYYY_MM_DD"
            week_date = week_dir.replace('feature_store_week_', '')
            if week_date >= start_date:
                available_weeks.append(week_date)
        
        available_weeks.sort()
        
        print(f"📅 Found {len(available_weeks)} weeks available for evaluation:")
        for week in available_weeks:
            print(f"   - {week}")
        
        return available_weeks
        
    except Exception as e:
        print(f"❌ Error getting available weeks: {e}")
        return []

def main():
    """
    Main function for weekly model evaluation - evaluates all three models on each week.
    """
    print("--- Multi-Model Weekly Evaluation (All Weeks) ---")
    
    # 1. Load all baseline models directly from MLflow
    models = load_all_baseline_models()
    if not models:
        print("❌ Cannot proceed without loading the models.")
        return
    
    # 2. Initialize Spark
    spark = SparkSession.builder \
        .appName("MultiModelWeeklyEvaluation") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    # 3. Get all available weeks to evaluate
    start_date = "2023_01_08"  # First week AFTER baseline training data
    available_weeks = get_available_weeks(start_date)
    
    if not available_weeks:
        print("❌ No weeks available for evaluation")
        spark.stop()
        return
    
    # 4. Evaluate all models on each week
    print(f"\n--- Evaluating {len(models)} Models on {len(available_weeks)} Weeks ---")
    
    all_weekly_results = {model_name: [] for model_name in models.keys()}
    
    for i, week_to_evaluate in enumerate(available_weeks, 1):
        print(f"\n{'='*80}")
        print(f"Week {i}/{len(available_weeks)}: {week_to_evaluate}")
        print(f"{'='*80}")
        
        # Load weekly data
        weekly_data = load_weekly_data(spark, week_to_evaluate)
        if weekly_data is None:
            print(f"⚠️  Skipping week {week_to_evaluate} - no data available")
            continue
        
        # Evaluate each model on this week's data
        week_results = {}
        
        for model_key, model_info in models.items():
            print(f"\n--- Evaluating {model_key} Model ---")
            
            # Prepare features using this model's feature names
            X, y, label_encoder = prepare_features(weekly_data, model_info["feature_names"], TARGET_COLUMN)
            
            # Evaluate model performance
            metrics, y_pred = evaluate_weekly_performance(model_info["model"], X, y, model_info["grade_mapping"])
            
            # Display results
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Macro F1 Score: {metrics['macro_f1']:.4f}")
            print(f"  Weighted F1 Score: {metrics['weighted_f1']:.4f}")
            print(f"  Total Samples: {metrics['total_samples']}")
            
            # Save metrics to PostgreSQL
            save_metrics_to_postgres(metrics, week_to_evaluate, model_info["run_id"], model_info["model_name"])
            
            # Store results for summary
            week_results[model_key] = {
                'week_date': week_to_evaluate,
                'metrics': metrics,
                'model_name': model_info["model_name"],
                'total_samples': metrics['total_samples']
            }
            
            all_weekly_results[model_key].append(week_results[model_key])
        
        # Compare models for this week
        print(f"\n--- Model Comparison for Week {week_to_evaluate} ---")
        for model_key, result in week_results.items():
            print(f"  {model_key}: Macro F1 = {result['metrics']['macro_f1']:.4f}")
        
        # Find best model for this week
        best_model = max(week_results.items(), key=lambda x: x[1]['metrics']['macro_f1'])
        print(f"  🏆 Best Model: {best_model[0]} (F1 = {best_model[1]['metrics']['macro_f1']:.4f})")
        
        print(f"✅ Week {week_to_evaluate} evaluation complete for all models")
    
    # 5. Generate comprehensive summary report
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE SUMMARY REPORT - {len(available_weeks)} Weeks Evaluated")
    print(f"{'='*80}")
    
    # Calculate summary statistics for each model
    model_summaries = {}
    
    for model_key, weekly_results in all_weekly_results.items():
        if not weekly_results:
            continue
            
        macro_f1_scores = [result['metrics']['macro_f1'] for result in weekly_results]
        accuracy_scores = [result['metrics']['accuracy'] for result in weekly_results]
        total_samples = sum([result['total_samples'] for result in weekly_results])
        
        model_summaries[model_key] = {
            'avg_macro_f1': sum(macro_f1_scores) / len(macro_f1_scores),
            'min_macro_f1': min(macro_f1_scores),
            'max_macro_f1': max(macro_f1_scores),
            'std_macro_f1': pd.Series(macro_f1_scores).std(),
            'avg_accuracy': sum(accuracy_scores) / len(accuracy_scores),
            'total_samples': total_samples,
            'weeks_evaluated': len(weekly_results)
        }
        
        print(f"\n📊 {model_key} Model Summary:")
        print(f"  Average Macro F1: {model_summaries[model_key]['avg_macro_f1']:.4f}")
        print(f"  Min Macro F1: {model_summaries[model_key]['min_macro_f1']:.4f}")
        print(f"  Max Macro F1: {model_summaries[model_key]['max_macro_f1']:.4f}")
        print(f"  Std Dev: {model_summaries[model_key]['std_macro_f1']:.4f}")
        print(f"  Average Accuracy: {model_summaries[model_key]['avg_accuracy']:.4f}")
        print(f"  Total Samples: {model_summaries[model_key]['total_samples']:,}")
    
    # Find the best overall model
    if model_summaries:
        best_overall_model = max(model_summaries.items(), key=lambda x: x[1]['avg_macro_f1'])
        print(f"\n🏆 BEST OVERALL MODEL: {best_overall_model[0]}")
        print(f"   Average Macro F1: {best_overall_model[1]['avg_macro_f1']:.4f}")
        
        # Model ranking
        print(f"\n📈 Model Ranking (by Average Macro F1):")
        sorted_models = sorted(model_summaries.items(), key=lambda x: x[1]['avg_macro_f1'], reverse=True)
        for i, (model_name, stats) in enumerate(sorted_models, 1):
            print(f"  {i}. {model_name}: {stats['avg_macro_f1']:.4f}")
    
    # Save comprehensive results
    summary_file = f"/opt/airflow/model_bank/multi_model_evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    summary_data = {
        'evaluation_date': datetime.now().isoformat(),
        'models_evaluated': list(models.keys()),
        'weeks_evaluated': len(available_weeks),
        'start_date': start_date,
        'end_date': available_weeks[-1] if available_weeks else None,
        'model_summaries': model_summaries,
        'all_weekly_results': all_weekly_results,
        'best_overall_model': best_overall_model[0] if model_summaries else None
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n📁 Comprehensive results saved to: {summary_file}")
    
    print(f"\n--- Multi-Model Weekly Evaluation Complete ---")
    print(f"🔗 Access MLflow UI at: http://localhost:5000")
    print(f"📊 View metrics in PostgreSQL or Grafana")
    print(f"📈 {len(available_weeks)} weeks evaluated with {len(models)} models")
    
    spark.stop()

if __name__ == "__main__":
    main() 