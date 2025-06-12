import json
import pandas as pd
from config import get_data_path, DEFAULT_DATA_FILE, DEFAULT_JSONL_FILE

# List to store our data
data = []

# Open and read the JSON Lines file
input_file = get_data_path(DEFAULT_JSONL_FILE, 'raw')
output_file = get_data_path(DEFAULT_DATA_FILE, 'raw')

with open(input_file, "r", encoding="utf-8") as file:
    for i, line in enumerate(file):
        if i >= 100:  # Only process the first 100 entries
            break
        record = json.loads(line)
        # Extract text and summary, with a default of empty string if missing
        data.append({
            "text": record.get("text", ""),
            "summary": record.get("summary", "")
        })

# Create a DataFrame and save it as CSV
df = pd.DataFrame(data)
df.to_csv(output_file, index=False)

print(f"CSV file '{output_file}' has been created with the top 100 entries.")
