import json
import pandas as pd

# List to store our data
data = []

# Open and read the JSON Lines file
with open("data.jsonl", "r", encoding="utf-8") as file:
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
df.to_csv("data.csv", index=False)

print("CSV file 'data.csv' has been created with the top 100 entries.")
