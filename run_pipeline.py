import os
import subprocess
import time

def run_script(script_name):
    print(f"\n" + "="*50)
    print(f"EXECUTING: {script_name}")
    print("="*50)
    
    start_time = time.time()
    try:
        # Run the script and wait for it to complete. check=True throws an error if the script fails.
        subprocess.run(["python", script_name], check=True)
        
        elapsed = time.time() - start_time
        print(f"\nSUCCESS: {script_name} completed in {elapsed:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"\nFATAL ERROR: {script_name} failed with exit code {e.returncode}.")
        print("Halting the pipeline to prevent downstream data corruption.")
        exit(1)
    except FileNotFoundError:
        print(f"\nERROR: Could not find {script_name} in the current directory.")
        exit(1)

def main():
    print("==================================================")
    print("  NIFTY-50 VOLATILITY PREDICTION PIPELINE START   ")
    print("==================================================")
    
    # The exact chronological sequence required to reproduce the project from scratch
    pipeline_scripts = [
        # 1. Data Extraction & Engineering
        "fetch_data.py",
        "preprocess_data.py",
        "split_and_scale.py",
        
        # 2. Baseline Predictions
        "naive_persistence_predictions.py",
        "linear_regression_prediction.py",
        
        # 3. Static ML Predictions (2008-2019 Train -> 2020-2026 Test)
        "random_forest_prediction.py",
        "xgboost_prediction.py",
        
        # 4. Walk-Forward Rolling ML Predictions (6-Month Adapting Windows)
        "random_forst_rolling_prediction.py",
        "xgboost_rolling_prediction.py",
        
        # 5. Final Evaluation & High-Res Visual Generation
        "evaluate_models.py"
    ]
    
    total_start = time.time()
    
    for script in pipeline_scripts:
        run_script(script)
        
    total_elapsed = time.time() - total_start
    print("\n" + "="*50)
    print(f"PIPELINE COMPLETE! All models trained, evaluated, and visuals generated.")
    print(f"Total Execution Time: {total_elapsed / 60:.2f} minutes.")
    print("Check the 'Outputs' folder for the final report visuals.")
    print("="*50)

if __name__ == "__main__":
    main()