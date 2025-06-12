import pandas as pd
import os


# Read the CSV files
# Note: The files have an empty first column, so we'll use index_col=0 to ignore it
df_deepseek = pd.read_csv('data_deepseek-r1_32b_new.csv', index_col=0)
df_gemma = pd.read_csv('data_gemma3_27b_new.csv', index_col=0)
df_mistral = pd.read_csv('data_mistral-small3.1_new.csv', index_col=0)
df_gpt4 = pd.read_csv('data_gpt-4o-mini.csv', index_col=0)


# Create model files with simplified names to avoid issues with special characters
# For deepseek
df_deepseek_renamed = df_deepseek.copy()
df_deepseek_renamed.rename(columns={
    'translate_deepseek-r1:32b': 'translate_llm',
    'summary_deepseek-r1:32b': 'summary_llm',
    'answer_deepseek-r1:32b': 'answer_llm'
}, inplace=True)
df_deepseek_renamed.to_csv('data_deepseek.csv', index=False)

# For gemma
df_gemma_renamed = df_gemma.copy()
df_gemma_renamed.rename(columns={
    'translate_gemma3:27b': 'translate_llm',
    'summary_gemma3:27b': 'summary_llm',
    'answer_gemma3:27b': 'answer_llm'
}, inplace=True)
df_gemma_renamed.to_csv('data_gemma.csv', index=False)

# For mistral
df_mistral_renamed = df_mistral.copy()
df_mistral_renamed.rename(columns={
    'translate_mistral-small3.1': 'translate_llm',
    'summary_mistral-small3.1': 'summary_llm',
    'answer_mistral-small3.1': 'answer_llm'
}, inplace=True)
df_mistral_renamed.to_csv('data_mistral.csv', index=False)


# For gpt4
df_gpt4_renamed = df_gpt4.copy()
df_gpt4_renamed.rename(columns={
    'translate_gpt-4o-mini': 'translate_llm',
    'summary_gpt-4o-mini': 'summary_llm',
    'answer_gpt-4o-mini': 'answer_llm'
}, inplace=True)
df_gpt4_renamed.to_csv('data_gpt4.csv', index=False)

print("Data files created successfully:")
print("- data_deepseek.csv")
print("- data_gemma.csv")
print("- data_mistral.csv")
print("- data_gpt4.csv")