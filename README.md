# mle_proj_datapipeline

## Git Link
https://github.com/brsse/mle_proj_datapipeline

## Overview
This project contains a comprehensive Machine Learning (ML) lifecycle pipeline for credit scoring, implemented using Apache Airflow. The pipeline processes data through multiple layers, trains ML models, and manages the complete model lifecycle with automated retraining and evaluation.

## DAG Pipeline Summary (`dags/dag.py`)

### **Pipeline Architecture**
The main DAG (`data_ml_pipeline`) implements a **weekly ML lifecycle** that runs every Sunday at 6 AM, processing credit scoring data through a three-layer data architecture:

#### **Data Flow: Bronze → Silver → Gold**
1. **Bronze Layer**: Raw data ingestion and standardization
   - Processes credit history, demographic, financial, and loan terms data
   - Applies basic data validation and formatting

2. **Silver Layer**: Data quality and transformation
   - Cleans and validates data from bronze layer
   - Applies business rules and data transformations

3. **Gold Layer**: Feature and label stores
   - Creates unified feature store combining all data types
   - Generates label store with target variables for model training

#### **ML Lifecycle Management**
The pipeline implements three distinct execution paths:

1. **Skip Path**: For historical data (pre-2023), skips ML processing
2. **Initial Training**: One-time setup to create the first production model
   - Trains both LightGBM and CatBoost models
   - Selects best performing model and registers as "Production"
3. **Weekly Lifecycle**: Ongoing operations with automated decision making
   - Evaluates current production model on new data
   - Automatically decides whether retraining is needed
   - If retraining required: trains new models, selects best, updates production
   - Runs inference with current production model

### **Key Features**
- **Weekly Dependencies**: Ensures each week's run waits for previous completion
- **Automated Model Selection**: Compares LightGBM vs CatBoost performance
- **MLflow Integration**: Model versioning and lifecycle management
- **Fault Tolerance**: Retry mechanisms and error handling
- **Data Quality Monitoring**: File sensors ensure data availability
- **Production-Ready**: Designed for continuous operation with minimal intervention

### **Pipeline Components**
- **Data Processing**: Bronze, silver, and gold layer transformations
- **Model Training**: Automated training of multiple algorithms
- **Model Evaluation**: Performance monitoring and comparison
- **Model Deployment**: Automated promotion to production
- **Inference**: Prediction generation for new data

### **Technology Stack**
- **Orchestration**: Apache Airflow
- **ML Framework**: LightGBM, CatBoost
- **Model Management**: MLflow
- **Data Processing**: PySpark, Pandas
- **Monitoring**: File sensors, external task sensors

---