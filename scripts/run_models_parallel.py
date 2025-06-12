import os
import subprocess
import sys
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_model(model_name, subset=None):
    """
    Run a single model process
    
    Args:
        model_name (str): Name of the model to run
        subset (int, optional): Number of samples to process (for testing)
    """
    print(f"Starting process for model: {model_name}")
    
    # Create command
    cmd = ["python", "model_runner.py", "--models", model_name]
    if subset:
        cmd.extend(["--subset", str(subset)])
    
    # Create log file
    log_file = f"log_{model_name.replace(':', '_').replace('/', '_').replace('.', '_')}.txt"
    
    # Run the process and redirect output to log file
    with open(log_file, 'w') as f:
        try:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Return process and log file for monitoring
            return {"model": model_name, "process": process, "log_file": log_file}
        except Exception as e:
            print(f"Error starting process for {model_name}: {e}")
            return None

def monitor_process(process_info):
    """
    Monitor a running process and return when complete
    
    Args:
        process_info (dict): Process information
    
    Returns:
        dict: Process results
    """
    model = process_info["model"]
    process = process_info["process"]
    log_file = process_info["log_file"]
    
    print(f"Monitoring process for {model}...")
    
    start_time = time.time()
    return_code = None
    
    while return_code is None:
        return_code = process.poll()
        elapsed = time.time() - start_time
        
        # Print status every 30 seconds
        if int(elapsed) % 30 == 0:
            print(f"Model {model} running for {int(elapsed)}s...")
        
        time.sleep(1)
    
    elapsed = time.time() - start_time
    
    if return_code == 0:
        print(f"Model {model} completed successfully in {elapsed:.1f}s")
        return {"model": model, "success": True, "elapsed": elapsed}
    else:
        print(f"Model {model} failed with code {return_code}")
        print(f"Check the log file: {log_file}")
        return {"model": model, "success": False, "elapsed": elapsed}

def main():
    parser = argparse.ArgumentParser(description="Run multiple models in parallel")
    parser.add_argument("--models", nargs="+", default=["llama3.2:1b", "deepseek-r1", "mistral"],
                        help="Model names to process (default: llama3.2:1b, deepseek-r1, mistral)")
    parser.add_argument("--subset", type=int, default=None, 
                        help="Process only a subset of rows (for testing)")
    
    args = parser.parse_args()
    
    print(f"Starting parallel processing for models: {', '.join(args.models)}")
    
    # Start all processes
    processes = []
    for model in args.models:
        process_info = run_model(model, args.subset)
        if process_info:
            processes.append(process_info)
    
    # Monitor all processes in parallel
    results = []
    with ThreadPoolExecutor(max_workers=len(processes)) as executor:
        future_to_process = {executor.submit(monitor_process, p): p for p in processes}
        for future in as_completed(future_to_process):
            process_info = future_to_process[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error monitoring process for {process_info['model']}: {e}")
    
    # Print summary
    print("\n--- RESULTS SUMMARY ---")
    success_count = sum(1 for r in results if r["success"])
    print(f"Completed: {success_count}/{len(args.models)} models")
    
    for result in sorted(results, key=lambda x: x["elapsed"]):
        status = "✓" if result["success"] else "✗"
        print(f"{status} {result['model']}: {result['elapsed']:.1f}s")
    
    # Run evaluation if all models were successful
    if success_count == len(args.models):
        print("\nAll models finished successfully. Running evaluation...")
        subprocess.run(["python", "evaluate_results.py"])
    else:
        print("\nSome models failed. Fix issues before running evaluation.")
    
    print("\nParallel processing complete!")

if __name__ == "__main__":
    main()