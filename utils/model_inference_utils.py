import pandas as pd
import mlflow
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession
import os
from typing import List, Dict, Any

# Configuration (can be overridden by caller)
PG_CONFIG = {
    'host': 'postgres',
    'port': 5432,
    'database': 'airflow',
    'user': 'airflow',
    'password': 'airflow'
}

MLFLOW_TRACKING_URI = "http://mlflow:5000"

# --- Model Loading ---
def load_model_from_mlflow(run_id: str, model_type: str):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    feature_names = json.loads(run.data.params.get('feature_names', '[]'))
    grade_mapping = {}
    grade_mapping_param = run.data.params.get('grade_mapping', '')
    if grade_mapping_param:
        grade_mapping = json.loads(grade_mapping_param)
    return model, grade_mapping, feature_names

# --- Data Loading ---
def load_weekly_data(spark: SparkSession, week_date: str, feature_store_path: str, label_store_path: str) -> pd.DataFrame:
    feature_path = os.path.join(feature_store_path, f"feature_store_week_{week_date}")
    label_path = os.path.join(label_store_path, f"label_store_week_{week_date}")
    if not os.path.exists(feature_path) or not os.path.exists(label_path):
        print(f"Feature or label directory not found for week {week_date}")
        return None
    feature_df = spark.read.parquet(feature_path)
    label_df = spark.read.parquet(label_path)
    full_df = feature_df.join(label_df, on='id', how='inner')
    return full_df.toPandas()

# --- Feature Preparation ---
def prepare_features(df: pd.DataFrame, feature_names: List[str], target_column: str = 'grade'):
    df = df.copy()
    label_encoder = LabelEncoder()
    df['grade_encoded'] = label_encoder.fit_transform(df[target_column])
    features_to_drop = [
        'id', target_column, 'snapshot_date', 'earliest_cr_date',
        'snapshot_month', 'earliest_cr_month', 'months_since_earliest_cr_line'
    ]
    df_cleaned = df.drop(columns=features_to_drop + ['grade_encoded'], errors='ignore')
    if not feature_names:
        feature_names = list(df_cleaned.columns)
    else:
        missing_features = set(feature_names) - set(df_cleaned.columns)
        if missing_features:
            missing_cols_dict = {feature: 0 for feature in missing_features}
            df_cleaned = df_cleaned.assign(**missing_cols_dict)
    X = df_cleaned[feature_names]
    y = df['grade_encoded']
    return X, y, label_encoder

# --- Evaluation ---
def evaluate_weekly_performance(model, X, y, grade_mapping):
    y_pred = model.predict(X)
    if hasattr(y_pred, 'ndim') and y_pred.ndim > 1:
        y_pred = y_pred.flatten()
    accuracy = accuracy_score(y, y_pred)
    macro_f1 = f1_score(y, y_pred, average='macro')
    weighted_f1 = f1_score(y, y_pred, average='weighted')
    f1_per_class = f1_score(y, y_pred, average=None)
    reverse_grade_mapping = {idx: grade for grade, idx in grade_mapping.items()}
    f1_by_grade = {}
    for i, score in enumerate(f1_per_class):
        if i in reverse_grade_mapping:
            f1_by_grade[reverse_grade_mapping[i]] = score
        else:
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

# --- Metrics Saving ---
def save_metrics_to_postgres(metrics: Dict[str, Any], week_date: str, run_id: str, model_name: str, pg_config: Dict[str, Any] = None):
    if pg_config is None:
        pg_config = PG_CONFIG
    try:
        conn = psycopg2.connect(**pg_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
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
        if 'conn' in locals():
            conn.close() 