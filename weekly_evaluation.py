import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import os
from typing import List, Dict, Any
from pyspark.sql import SparkSession
import mlflow
import mlflow.xgboost
import json
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuration
FEATURE_STORE_PATH = "/opt/airflow/datamart/gold/feature_store"
LABEL_STORE_PATH = "/opt/airflow/datamart/gold/label_store"
TARGET_COLUMN = 'grade'
UNIQUE_ID_COLUMN = 'id'
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "credit_scoring_pipeline"

# PostgreSQL configuration (for future metric storage)
PG_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'airflow',
    'user': 'airflow',
    'password': 'airflow'
}

def load_baseline_model_from_mlflow(run_id: str):
    """
    Load the baseline model from MLflow using the run_id.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Load the model
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.xgboost.load_model(model_uri)
        
        # Load the run to get parameters
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        # Extract parameters
        grade_mapping = json.loads(run.data.params.get('grade_mapping', '{}'))
        feature_names = json.loads(run.data.params.get('feature_names', '[]'))
        
        print(f"✅ Successfully loaded model from MLflow run: {run_id}")
        return model, grade_mapping, feature_names
        
    except Exception as e:
        print(f"❌ Error loading model from MLflow: {e}")
        return None, None, None

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
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')
    weighted_f1 = f1_score(y, y_pred, average='weighted')
    
    # Get per-class F1 scores
    f1_per_class = f1_score(y, y_pred, average=None)
    f1_by_grade = {grade_mapping[str(i)]: score for i, score in enumerate(f1_per_class)}
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'f1_by_grade': f1_by_grade,
        'total_samples': len(y),
        'predictions_distribution': pd.Series(y_pred).value_counts().to_dict()
    }
    
    return metrics, y_pred

def save_metrics_to_postgres(metrics: Dict[str, Any], week_date: str, run_id: str):
    """
    Save weekly evaluation metrics to PostgreSQL.
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
            'baseline_credit_scoring',
            metrics['accuracy'],
            metrics['macro_f1'],
            metrics['weighted_f1'],
            metrics['total_samples'],
            json.dumps(metrics['f1_by_grade']),
            json.dumps(metrics['predictions_distribution'])
        ))
        
        conn.commit()
        print(f"✅ Metrics saved to PostgreSQL for week {week_date}")
        
    except Exception as e:
        print(f"❌ Error saving to PostgreSQL: {e}")
    finally:
        if conn:
            conn.close()

def get_baseline_metrics():
    """
    Get the baseline metrics from the saved JSON file.
    """
    try:
        # Find the most recent baseline metrics file
        model_bank_dir = "/opt/airflow/model_bank"
        baseline_files = [f for f in os.listdir(model_bank_dir) if f.startswith('baseline_metrics_')]
        
        if not baseline_files:
            print("❌ No baseline metrics file found")
            return None
        
        latest_file = max(baseline_files)
        file_path = os.path.join(model_bank_dir, latest_file)
        
        with open(file_path, 'r') as f:
            baseline_data = json.load(f)
        
        print(f"✅ Loaded baseline metrics from {latest_file}")
        return baseline_data
        
    except Exception as e:
        print(f"❌ Error loading baseline metrics: {e}")
        return None

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
    Main function for weekly model evaluation - iterates through all available weeks.
    """
    print("--- Weekly Model Evaluation (All Weeks) ---")
    
    # 1. Load baseline metrics
    baseline_data = get_baseline_metrics()
    if not baseline_data:
        print("❌ Cannot proceed without baseline metrics")
        return
    
    baseline_macro_f1 = baseline_data['metrics']['macro_f1']
    baseline_run_id = baseline_data['run_id']
    baseline_snapshot_date = baseline_data['snapshot_date']
    
    print(f"📊 Baseline Macro F1 Score: {baseline_macro_f1:.4f}")
    print(f"📅 Baseline model trained on data up to: {baseline_snapshot_date}")
    
    # 2. Load the baseline model from MLflow
    model, grade_mapping, feature_names = load_baseline_model_from_mlflow(baseline_run_id)
    if model is None:
        print("❌ Cannot proceed without loading the model")
        return
    
    # 3. Initialize Spark
    spark = SparkSession.builder \
        .appName("WeeklyEvaluation") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    # 4. Get all available weeks to evaluate
    start_date = "2023_01_08"  # First week AFTER baseline training data
    available_weeks = get_available_weeks(start_date)
    
    if not available_weeks:
        print("❌ No weeks available for evaluation")
        spark.stop()
        return
    
    # 5. Evaluate model on each week
    print(f"\n--- Evaluating Model on {len(available_weeks)} Weeks ---")
    
    weekly_results = []
    
    for i, week_to_evaluate in enumerate(available_weeks, 1):
        print(f"\n{'='*60}")
        print(f"Week {i}/{len(available_weeks)}: {week_to_evaluate}")
        print(f"{'='*60}")
        
        # Load weekly data
        weekly_data = load_weekly_data(spark, week_to_evaluate)
        if weekly_data is None:
            print(f"⚠️  Skipping week {week_to_evaluate} - no data available")
            continue
        
        # Prepare features
        X, y, label_encoder = prepare_features(weekly_data, feature_names, TARGET_COLUMN)
        
        # Evaluate model performance
        metrics, y_pred = evaluate_weekly_performance(model, X, y, grade_mapping)
        
        # Display results
        print(f"Weekly Performance Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1 Score: {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1 Score: {metrics['weighted_f1']:.4f}")
        print(f"  Total Samples: {metrics['total_samples']}")
        
        # Compare with baseline
        f1_change = metrics['macro_f1'] - baseline_macro_f1
        print(f"\nPerformance vs Baseline:")
        print(f"  Baseline Macro F1: {baseline_macro_f1:.4f}")
        print(f"  Weekly Macro F1:   {metrics['macro_f1']:.4f}")
        print(f"  Change:            {f1_change:+.4f}")
        
        if f1_change < -0.05:
            print("⚠️  WARNING: Significant performance degradation detected!")
        elif f1_change > 0.05:
            print("🎉 Performance improvement detected!")
        else:
            print("✅ Performance is stable")
        
        # Save metrics to PostgreSQL
        save_metrics_to_postgres(metrics, week_to_evaluate, baseline_run_id)
        
        # Store results for summary
        weekly_results.append({
            'week_date': week_to_evaluate,
            'metrics': metrics,
            'f1_change': f1_change,
            'total_samples': metrics['total_samples']
        })
        
        print(f"✅ Week {week_to_evaluate} evaluation complete")
    
    # 6. Generate summary report
    print(f"\n{'='*60}")
    print(f"SUMMARY REPORT - {len(weekly_results)} Weeks Evaluated")
    print(f"{'='*60}")
    
    if weekly_results:
        # Calculate summary statistics
        macro_f1_scores = [result['metrics']['macro_f1'] for result in weekly_results]
        f1_changes = [result['f1_change'] for result in weekly_results]
        total_samples = sum([result['total_samples'] for result in weekly_results])
        
        print(f"📊 Macro F1 Score Statistics:")
        print(f"  Average: {sum(macro_f1_scores)/len(macro_f1_scores):.4f}")
        print(f"  Min: {min(macro_f1_scores):.4f}")
        print(f"  Max: {max(macro_f1_scores):.4f}")
        print(f"  Std Dev: {pd.Series(macro_f1_scores).std():.4f}")
        
        print(f"\n📈 Performance Change vs Baseline:")
        print(f"  Average Change: {sum(f1_changes)/len(f1_changes):+.4f}")
        print(f"  Worst Week: {min(f1_changes):+.4f}")
        print(f"  Best Week: {max(f1_changes):+.4f}")
        
        print(f"\n📋 Total Samples Evaluated: {total_samples:,}")
        
        # Identify problematic weeks
        degraded_weeks = [r for r in weekly_results if r['f1_change'] < -0.05]
        if degraded_weeks:
            print(f"\n⚠️  Weeks with Significant Degradation:")
            for week in degraded_weeks:
                print(f"  - {week['week_date']}: {week['f1_change']:+.4f}")
        
        # Save comprehensive results
        summary_file = f"/opt/airflow/model_bank/weekly_evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary_data = {
            'evaluation_date': datetime.now().isoformat(),
            'baseline_run_id': baseline_run_id,
            'baseline_macro_f1': baseline_macro_f1,
            'baseline_snapshot_date': baseline_snapshot_date,
            'weeks_evaluated': len(weekly_results),
            'start_date': start_date,
            'end_date': available_weeks[-1] if available_weeks else None,
            'summary_stats': {
                'avg_macro_f1': sum(macro_f1_scores)/len(macro_f1_scores),
                'min_macro_f1': min(macro_f1_scores),
                'max_macro_f1': max(macro_f1_scores),
                'std_macro_f1': pd.Series(macro_f1_scores).std(),
                'avg_f1_change': sum(f1_changes)/len(f1_changes),
                'total_samples': total_samples
            },
            'weekly_results': weekly_results,
            'degraded_weeks': [w['week_date'] for w in degraded_weeks]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n📁 Comprehensive results saved to: {summary_file}")
    
    print(f"\n--- Weekly Evaluation Complete ---")
    print(f"🔗 Access MLflow UI at: http://localhost:5000")
    print(f"📊 View metrics in PostgreSQL or Grafana")
    print(f"📈 {len(weekly_results)} weeks evaluated from {start_date} onwards")
    
    spark.stop()

if __name__ == "__main__":
    main() 