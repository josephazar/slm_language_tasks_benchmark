import os
import pandas as pd

print("=== File Diagnostics ===")

# Check the current working directory
print(f"Current working directory: {os.getcwd()}")

# List all files in the directory
print("\nFiles in current directory:")
for file in os.listdir():
    print(f" - {file}")

# Check specifically for the model files
model_files = ["data.csv", "results/data_deepseek.csv", "results/data_gemma.csv", "results/data_mistral.csv"]
print("\nChecking for specific model files:")
for file in model_files:
    if os.path.exists(file):
        # Check file size
        size = os.path.getsize(file)
        print(f" - {file}: EXISTS, size: {size} bytes")
        
        # Check if file can be opened
        try:
            df = pd.read_csv(file)
            print(f"   * Successfully read {len(df)} rows and {len(df.columns)} columns")
            print(f"   * Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"   * ERROR reading file: {str(e)}")
    else:
        print(f" - {file}: NOT FOUND")

# Check the results directory
results_dir = "results"
if os.path.exists(results_dir) and os.path.isdir(results_dir):
    print(f"\nContents of {results_dir} directory:")
    for file in os.listdir(results_dir):
        print(f" - {file}")
else:
    print(f"\n{results_dir} directory not found or is not a directory")

print("\n=== End of Diagnostics ===")