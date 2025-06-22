#!/usr/bin/env python3
"""
Script to train all specified models in sequence.
This ensures models are available for evaluation.
"""

import subprocess
import sys
from datetime import datetime

def run_training_script(script_name: str, model_name: str):
    """
    Run a training script and stream its output in real-time.
    """
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING {model_name.upper()} MODEL")
    print(f"{'='*80}")
    
    try:
        script_path = f"/opt/airflow/utils/{script_name}"
        
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd="/opt/airflow",
            bufsize=1,
            universal_newlines=True
        )

        for line in process.stdout:
            print(line, end='')

        process.wait()
        
        if process.returncode == 0:
            print(f"\n✅ {model_name} training completed successfully!")
            return True
        else:
            print(f"\n❌ {model_name} training failed with return code {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """
    Main function to orchestrate the training of selected models.
    """
    print("🎯 MULTI-MODEL TRAINING PIPELINE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define the training scripts to run. XGBoost is currently disabled.
    training_scripts = [
        # ("standalone_training_run.py", "XGBoost"), # Disabled
        ("LightGBM_training_run.py", "LightGBM"),
        ("CatBoost_training_run.py", "CatBoost")
    ]
    
    results = {}
    
    for script_name, model_name in training_scripts:
        success = run_training_script(script_name, model_name)
        results[model_name] = success
        if not success:
            print(f"⚠️  {model_name} training failed. Continuing with other models...")
    
    print(f"\n{'='*80}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*80}")
    
    successful_models = [name for name, success in results.items() if success]
    failed_models = [name for name, success in results.items() if not success]
    
    print(f"✅ Successfully trained: {', '.join(successful_models) if successful_models else 'None'}")
    print(f"❌ Failed to train: {', '.join(failed_models) if failed_models else 'None'}")
    
    if successful_models:
        print(f"\n🎉 Ready for multi-model evaluation!")
    else:
        print(f"\n💥 No models were trained successfully. Please check the errors above.")
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 