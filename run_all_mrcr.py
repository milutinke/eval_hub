import os
import subprocess
import glob
import pandas as pd
from datetime import datetime

# Configuration - will prioritize .env if present
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://192.168.40.13:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = os.environ.get("OPENAI_MODEL_NAME", "qwen36-27b-nvfp4-fp8kv-262k")

DATA_DIR = "mrcr_v2"
RESULTS_DIR = "results_mrcr"

def run_all_evaluations():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Error: Data directory '{DATA_DIR}' not found. Run download.sh first.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {DATA_DIR}")
        return

    print(f"🚀 Starting batch evaluation for {len(csv_files)} datasets...")
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: {BASE_URL}")
    print("-" * 50)

    summary_data = []

    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULTS_DIR, result_filename)
        
        print(f"\nEvaluating: {filename}...")
        
        cmd = [
            "uv", "run", "eval_hub/mrcr_v2/run_evaluation.py",
            f"--input_path={csv_file}",
            f"--output_path={result_path}",
            f"--model_name={MODEL_NAME}",
            f"--openai_api_key={API_KEY}",
            f"--openai_base_url={BASE_URL}"
        ]
        
        try:
            # Run the evaluation script
            subprocess.run(cmd, check=True)
            
            # Read the results to get the average score
            df = pd.read_csv(result_path)
            avg_score = df['score'].mean()
            
            summary_data.append({
                "Dataset": filename,
                "Avg Score": avg_score,
                "Samples": len(df)
            })
            
            print(f"✅ Finished {filename}. Avg Score: {avg_score:.4f}")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error evaluating {filename}: {e}")
            summary_data.append({
                "Dataset": filename,
                "Avg Score": "FAILED",
                "Samples": 0
            })

    # Save and Print Summary
    print("\n" + "="*50)
    print("BATCH EVALUATION SUMMARY")
    print("="*50)
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    summary_path = os.path.join(RESULTS_DIR, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*50)
    print(f"Full results saved in: {RESULTS_DIR}")
    print(f"Summary report saved to: {summary_path}")

if __name__ == "__main__":
    run_all_evaluations()
