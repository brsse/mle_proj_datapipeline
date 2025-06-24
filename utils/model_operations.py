import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
from typing import Dict, Any, List
import os
from pyspark.sql import SparkSession
import numpy as np
import psutil

def load_data_for_training(spark: SparkSession, feature_store_path: str, label_store_path: str, weeks: List[str]) -> pd.DataFrame:
    """Loads feature and label data for a given list of week partitions and joins them."""
    
    print(f"Loading data for {len(weeks)} weeks...")
    
    # Load data in chunks to avoid memory issues
    chunk_size = 12  # Load 12 weeks at a time (2 chunks for 24 weeks)
    all_data = []
    
    for i in range(0, len(weeks), chunk_size):
        chunk_weeks = weeks[i:i+chunk_size]
        print(f"Processing weeks {i+1}-{min(i+chunk_size, len(weeks))} ({len(chunk_weeks)} weeks)...")
        
        feature_paths = [os.path.join(feature_store_path, f"feature_store_week_{week}") for week in chunk_weeks]
        label_paths = [os.path.join(label_store_path, f"label_store_week_{week}") for week in chunk_weeks]
        
        print(f"Reading {len(feature_paths)} feature files...")
        features_df = spark.read.parquet(*feature_paths)
        print(f"Reading {len(label_paths)} label files...")
        labels_df = spark.read.parquet(*label_paths)
        
        print(f"Joining features and labels...")
        # The 'id' column is the key to join features and labels
        chunk_df = features_df.join(labels_df, "id")
        
        # Get the count before converting to pandas
        count = chunk_df.count()
        print(f"Chunk records after join: {count}")
        
        print("Converting chunk to pandas...")
        chunk_pandas = chunk_df.toPandas()
        all_data.append(chunk_pandas)
        
        print(f"Chunk {i//chunk_size + 1} completed. Memory usage: {chunk_pandas.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        print(f"System memory: {psutil.virtual_memory().percent:.1f}% used")
    
    print("Combining all chunks...")
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"Final dataset shape: {final_df.shape}")
    
    return final_df

def load_data_for_training_spark(spark: SparkSession, feature_store_path: str, label_store_path: str, weeks: List[str]):
    """Loads feature and label data and keeps it in Spark format for efficient processing."""
    
    print(f"Loading data for {len(weeks)} weeks in Spark format...")
    
    feature_paths = [os.path.join(feature_store_path, f"feature_store_week_{week}") for week in weeks]
    label_paths = [os.path.join(label_store_path, f"label_store_week_{week}") for week in weeks]
    
    print(f"Reading {len(feature_paths)} feature files...")
    features_df = spark.read.parquet(*feature_paths)
    print(f"Reading {len(label_paths)} label files...")
    labels_df = spark.read.parquet(*label_paths)
    
    print(f"Joining features and labels...")
    final_df = features_df.join(labels_df, "id")
    
    count = final_df.count()
    print(f"Total records: {count}")
    
    return final_df

def sample_spark_data_for_training(spark_df, sample_fraction=0.1, seed=42):
    """Sample Spark DataFrame for faster training while keeping it in Spark format."""
    print(f"Sampling {sample_fraction*100}% of data for training...")
    sampled_df = spark_df.sample(fraction=sample_fraction, seed=seed)
    count = sampled_df.count()
    print(f"Sampled to {count} records")
    return sampled_df

def train_and_tune_model(training_df: pd.DataFrame, model_type: str = 'xgboost'):
    """
    Trains and tunes a model (XGBoost or LightGBM) using Hyperopt.
    Logs all parameters, metrics, and the model artifact to MLflow.
    """
    
    # Separate features (X) and target (y)
    # The 'errors="ignore"' flag prevents a crash if the columns were already dropped.
    X = training_df.drop(columns=['id', 'grade', 'snapshot_date'], errors='ignore')
    y = training_df['grade'].apply(lambda x: 1 if x in ['D', 'E', 'F', 'G'] else 0) # Example binary target
    
    # Split data into training and validation for hyperparameter tuning
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def objective(params):
        with mlflow.start_run(nested=True):
            if model_type == 'xgboost':
                model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=False)
            elif model_type == 'lightgbm':
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30, verbose=-1)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            
            mlflow.log_params(params)
            mlflow.log_metric("validation_auc", auc)
            
            # Hyperopt minimizes the loss, so we return 1 - AUC
            return {'loss': 1 - auc, 'status': STATUS_OK}

    if model_type == 'xgboost':
        search_space = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'subsample': hp.uniform('subsample', 0.7, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1.0),
        }
    elif model_type == 'lightgbm':
         search_space = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
            'learning_rate': hp.loguniform('learning_rate', -3, 0),
            'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
        }

    with mlflow.start_run(run_name=f"Tune_{model_type}"):
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=20, # Increase for more thorough search
            trials=trials
        )
        
        # Log the best parameters found
        mlflow.log_params(best_params)
        
        # Train final model on full data with best parameters
        if model_type == 'xgboost':
            final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
            mlflow.xgboost.log_model(final_model, f"{model_type}-model")
        elif model_type == 'lightgbm':
            final_model = lgb.LGBMClassifier(**best_params)
            mlflow.lightgbm.log_model(final_model, f"{model_type}-model")
            
        final_model.fit(X, y) # Fit on all data
        
        # You can log other artifacts like feature importance plots here
        print(f"Finished training for {model_type}. Best validation AUC: {1 - trials.best_trial['result']['loss']:.4f}")

def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Calculate the Population Stability Index (PSI) for a single variable."""
    
    # Create bins based on the 'expected' distribution
    breakpoints = pd.unique(np.percentile(expected, [i * 100 / buckets for i in range(buckets + 1)]))
    
    expected_percents = pd.cut(expected, bins=breakpoints, retbins=False, labels=False).value_counts(normalize=True)
    actual_percents = pd.cut(actual, bins=breakpoints, retbins=False, labels=False).value_counts(normalize=True)

    # Align the series to ensure we have the same bins
    expected_percents, actual_percents = expected_percents.align(actual_percents, join='outer', fill_value=0)
    
    # Add a small epsilon to avoid division by zero
    expected_percents = expected_percents.replace(0, 0.0001)
    actual_percents = actual_percents.replace(0, 0.0001)

    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    
    return np.sum(psi_values)

def run_oot_monitoring(oot_df: pd.DataFrame, training_df_for_psi: pd.DataFrame, model_uri: str):
    """
    Performs OOT validation on a model from the MLflow Registry.
    Calculates AUC, KS, and PSI.
    """
    # Load the production model
    production_model = mlflow.pyfunc.load_model(model_uri)
    
    # Prepare OOT data
    X_oot = oot_df.drop(columns=['id', 'grade', 'snapshot_date'])
    y_oot = oot_df['grade'].apply(lambda x: 1 if x in ['D', 'E', 'F', 'G'] else 0)
    
    # Make predictions
    y_pred_proba = production_model.predict(X_oot)
    
    # Calculate metrics
    oot_auc = roc_auc_score(y_oot, y_pred_proba)
    
    # Calculate PSI on the model score
    training_scores = production_model.predict(training_df_for_psi.drop(columns=['id', 'grade', 'snapshot_date']))
    score_psi = calculate_psi(pd.Series(training_scores), pd.Series(y_pred_proba))

    with mlflow.start_run(run_name="OOT_Validation"):
        mlflow.log_metric("oot_auc", oot_auc)
        mlflow.log_metric("oot_score_psi", score_psi)
        print(f"OOT AUC: {oot_auc:.4f}, Score PSI: {score_psi:.4f}")

    return {"oot_auc": oot_auc, "oot_score_psi": score_psi} 