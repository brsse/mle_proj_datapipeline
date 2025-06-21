# ML Pipeline DAG Architecture

## Overview
This project implements a robust ML pipeline using Apache Airflow with three main DAGs that communicate through file-based decisions rather than direct DAG-to-DAG triggers.

## DAG Structure

### 1. Data Preprocessing DAG (`P2PCreditScore_DataPreprocessing`)
- **Schedule**: Daily at 1 AM
- **Purpose**: Process raw data into bronze, silver, and gold layers
- **Output**: Feature store and label store files in gold layer

### 2. Model Training/Inference DAG (`P2PCreditScore_ModelTrainingInference`)
- **Schedule**: Daily at 2 AM
- **Purpose**: Train models and run inference
- **Dependencies**: 
  - Waits for data preprocessing completion
  - Reads retraining decisions from monitoring DAG

### 3. Model Monitoring DAG (`P2PCreditScore_ModelMonitoring`)
- **Schedule**: Weekly on Sundays at 6 AM
- **Purpose**: Monitor model performance and make retraining decisions
- **Output**: Writes retraining decisions to files

## Robust Communication Architecture

### File-Based Decision System

Instead of using DAG-to-DAG triggers, the system uses a file-based communication approach:

```
Weekly Monitoring DAG
    ↓ (analyzes performance)
    ↓ (writes decision to /opt/airflow/logs/retraining_decision.json)
    ↓ (completes)

Daily Training/Inference DAG
    ↓ (reads decision file)
    ↓ (executes retraining if needed)
    ↓ (clears decision file)
    ↓ (logs activity)
```

### Decision File Format
```json
{
  "decision_date": "2024-01-15",
  "performance_score": 0.65,
  "retrain": true,
  "reason": "Performance score 0.650 below threshold"
}
```

## Why This Approach is More Robust

### ✅ Advantages of File-Based Communication

1. **Decoupled DAGs**: No direct dependencies between DAGs
2. **Persistent State**: Decisions survive DAG failures and restarts
3. **Easy Debugging**: Can inspect decision files manually
4. **Flexible Timing**: Training DAG can check decisions at any time
5. **Failure Recovery**: If monitoring fails, training can proceed with last known decision
6. **No Race Conditions**: No timing issues between DAG executions
7. **Audit Trail**: Decision files provide clear audit trail

### ❌ Problems with DAG-to-DAG Triggers

1. **Tight Coupling**: DAGs become dependent on each other
2. **Race Conditions**: Timing issues between DAG executions
3. **Failure Propagation**: If monitoring fails, retraining won't happen
4. **Harder Debugging**: Issues span across multiple DAGs
5. **Complex Error Handling**: Need to handle cross-DAG failures

## Task Dependencies

### Data Preprocessing DAG
```
start → source_data_checks → bronze_processing → silver_processing → gold_processing → end
```

### Model Training/Inference DAG
```
wait_for_preprocessing → check_inference_condition → [skip_inference, check_retraining_needed]
check_retraining_needed → [skip_retraining, train_model_task]
[skip_retraining, train_model_task] → get_model → run_inference → clear_decision → log_retraining → end
```

### Model Monitoring DAG
```
check_monitoring → [skip_monitoring, get_week_end_date]
get_week_end_date → get_model → calculate_metrics → [skip_report, generate_report]
generate_report → write_retraining_decision → send_alerts → end
```

## Key Features

### 1. Conditional Execution
- Training only happens when retraining decision is positive
- Inference only happens when data preprocessing is complete
- Monitoring only happens when sufficient data is available

### 2. Weekly Retraining Limits
- Retraining is limited to once per week using log files
- Prevents excessive retraining and resource waste

### 3. Error Handling
- Graceful handling of missing files
- Fallback mechanisms for decision failures
- Clear logging of all activities

### 4. Scalability
- Easy to add new monitoring metrics
- Simple to modify retraining logic
- Can add new DAGs without affecting existing ones

## File Structure

```
/opt/airflow/
├── logs/
│   ├── retraining_decision.json    # Current retraining decision
│   └── retraining_log_YYYY_MM_DD.txt  # Weekly retraining logs
├── scripts/
│   └── datamart/
│       ├── bronze/                 # Raw processed data
│       ├── silver/                 # Cleaned data
│       └── gold/                   # Feature store
└── model_bank/                     # Trained models
```

## Best Practices Implemented

1. **Separation of Concerns**: Each DAG has a single responsibility
2. **Idempotency**: Tasks can be safely retried
3. **Observability**: Clear logging and decision tracking
4. **Fault Tolerance**: Graceful handling of failures
5. **Maintainability**: Simple, clear code structure
6. **Scalability**: Easy to extend and modify

## Monitoring and Alerting

- Performance metrics are calculated weekly
- Retraining decisions are logged with reasons
- Alerts can be sent for critical issues
- All activities are tracked in log files

This architecture provides a robust, maintainable, and scalable ML pipeline that can handle failures gracefully while maintaining clear separation between different pipeline stages. 