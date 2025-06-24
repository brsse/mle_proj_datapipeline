# Parked Discussion Items & Future Enhancements

This document serves as a parking lot for important MLOps concepts and potential future enhancements to our pipeline that we have discussed but decided to implement later.

---

## 1. Advanced Model Retraining Strategy

**Discussion Date:** 2025-06-22

### Summary:
We discussed moving beyond a simple time-based retraining schedule (e.g., "retrain every 3 months") to a more dynamic and robust strategy. The best practice is a hybrid approach that combines performance monitoring with a scheduled safety net.

### Key Concepts:

#### a. Performance-Based Triggers
The primary trigger for retraining should be a **sustained drop in model performance**.
- **Monitoring Tool**: The "Weekly Model Performance" dashboard in Grafana.
- **The Signal**: A clear, downward trend in the key business metric (e.g., Macro F1 score) over several consecutive weeks (e.g., 3-4 weeks). A drop of 5-10% from the model's peak performance is a common threshold.
- **Interpretation**: This indicates that the patterns in new, incoming data have diverged significantly from the data the model was trained on (Concept Drift).

#### b. Data Drift Monitoring (Leading Indicator)
This is a more proactive trigger. Sometimes, the underlying data distributions change *before* the model's performance metrics degrade.
- **Monitoring Tool**: A new panel in Grafana to track the **Population Stability Index (PSI)**.
- **The Signal**: A significant PSI value for key predictive features (e.g., `annual_inc`, `dti`).
- **Interpretation**: This indicates that the population of customers has changed. For example, a sudden shift in the average income or debt-to-income ratio of applicants. This is a strong leading indicator that model performance will likely degrade soon.

#### c. Scheduled Retraining (Safety Net)
This is a fallback mechanism to ensure the model stays fresh and doesn't become stale over time, even if performance degradation is very slow and gradual.
- **Implementation Tool**: A scheduled Airflow DAG.
- **Suggested Schedule**: A quarterly (every 3 months) or semi-annual (every 6 months) schedule is a reasonable and common practice for this safety net.

### Proposed Hybrid Strategy:
1.  **Active Monitoring**: Use Grafana dashboards to watch for performance degradation (Macro F1) and data drift (PSI).
2.  **Automated Alerting**: Configure alerts in Grafana to automatically send a notification when a metric crosses a predefined threshold for a sustained period.
3.  **Scheduled Fallback**: Implement a quarterly Airflow DAG that automatically retrains the model on the latest data, ensuring it never becomes too outdated.

This approach ensures that retraining is done when it's *necessary*, not just when the calendar says so, while the schedule provides a guarantee against model staleness.

## Parked Discussion Items for Future Implementation

This document tracks key architectural and feature decisions to be implemented in the future.

### Next Steps & Long-Term Vision:

1.  **Script Organization:** Relocate all primary execution scripts (`standalone_training_run.py`, `weekly_evaluation.py`, etc.) into the `utils` folder to create a cleaner and more modular file hierarchy.

2.  **Airflow DAG Logic - Data Availability:**
    *   The DAG should have a condition at the start to check if at least 6 months (24 weeks) of data is available from the start date (16 Jan 2022).
    *   If less than 24 weeks of data exists, all downstream tasks, including model training and evaluation, should be skipped.

3.  **Retraining Strategy:**
    *   **Trigger Conditions:** Define a dual-trigger system for model retraining:
        *   **Performance-based:** Retrain if the `macro_f1_score` from weekly monitoring drops below a predefined threshold.
        *   **Time-based:** Force a retraining every 3 months, regardless of performance, to prevent model staleness.
    *   **Training Data Window:** When retraining, the model should only use the most recent 6 months of data leading up to the trigger date. It must not include data from the week that is currently being processed for inference.

4.  **DAG Dependencies and Backfill Safety:**
    *   Structure the Airflow DAG to ensure that the model monitoring task for a given week (Week N) must complete successfully *before* the DAG for the next week (Week N+1) decides whether to retrain its model.
    *   This will require careful dependency management (e.g., using `ExternalTaskSensor`) to handle Airflow's backfill behavior correctly and prevent race conditions.

5.  **Model Selection Simplification:**
    *   During each retraining cycle, perform a model selection step.
    *   Consider simplifying the model suite to only include **LightGBM** and **CatBoost**, as they consistently outperform XGBoost in initial tests.

6.  **Model Registry Implementation:**
    *   Fully leverage the MLflow Model Registry.
    *   After each retraining and model selection cycle, the winning model (e.g., the best-performing between LightGBM and CatBoost) should be registered and promoted to a "Production" stage.
    *   Inference and monitoring tasks must always pull the model version currently tagged as "Production".

7.  **Source of Performance Metrics:**
    *   Clarify that the primary metrics used for ongoing monitoring and triggering retraining should be generated exclusively from the weekly **inference** (out-of-time validation) results, not from hold-out sets during training.

8.  **Parallel Training in Airflow:**
    *   When the pipeline is migrated to Airflow, the training for different models (e.g., LightGBM and CatBoost) should be executed in parallel tasks.
    *   This will improve efficiency, as the total training time will be determined by the longest-running model, not the sum of all.
    *   A downstream task will then be responsible for selecting the best model out of the completed runs before registration.

9.  **GPU-Accelerated Tuning (Future Optimization):**
    *   Investigate the use of GPU acceleration (`tree_method='gpu_hist'` for XGBoost, `device='gpu'` for LightGBM, `task_type='GPU'` for CatBoost) for the offline hyperparameter tuning process.
    *   This could dramatically reduce the time required for experimental, exhaustive tuning jobs, allowing for more thorough searches when needed. 