
import pandas as pd
import re

# Load your DataFrame
df = pd.read_csv("data_deepseek.csv")

# Function to remove <think>...</think> tags and their contents
def remove_thinking_tags(text):
    if isinstance(text, str):
        # This pattern matches <think> tag, everything inside it (including newlines), and the closing </think> tag
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    return text

# Apply the function to all string columns in the DataFrame
for column in df.columns:
    if df[column].dtype == 'object':  # Only process string columns
        df[column] = df[column].apply(remove_thinking_tags)

# Show a sample of the cleaned data
print(df.head())

# Save the cleaned DataFrame
df.to_csv("data_deepseek.csv", index=False)

print("All <think>...</think> tags have been removed from the DataFrame.")