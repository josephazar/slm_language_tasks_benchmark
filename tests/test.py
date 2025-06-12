import pandas as pd
import os


baseline_df = pd.read_csv('data.csv')

gpt4_df = pd.read_csv('results/data_gpt4.csv')
deepseek_df = pd.read_csv('results/data_deepseek.csv')
gemma_df = pd.read_csv('results/data_gemma.csv')
mistral_df = pd.read_csv('results/data_mistral.csv')


for i in range(len(baseline_df)):
    # BASELINE ABSTRACTIVE SUMMARY 
    baseline_abstractive_summary = baseline_df['summary_en'][i]
    print("Baseline abstractive summary for record", i, ":", baseline_abstractive_summary)
    # DEEPSEEK ABSTRACTIVE SUMMARY
    deepseek_abstractive_summary = deepseek_df['summary_llm'][i]
    print("Deepseek abstractive summary for record", i, ":", deepseek_abstractive_summary)
    # GEMMA ABSTRACTIVE SUMMARY
    gemma_abstractive_summary = gemma_df['summary_llm'][i]
    print("Gemma abstractive summary for record", i, ":", gemma_abstractive_summary)
    # MISTRAL ABSTRACTIVE SUMMARY
    mistral_abstractive_summary = mistral_df['summary_llm'][i]
    print("Mistral abstractive summary for record", i, ":", mistral_abstractive_summary)
    # GPT4 ABSTRACTIVE SUMMARY
    gpt4_abstractive_summary = gpt4_df['summary_llm'][i]
    print("GPT4 abstractive summary for record", i, ":", gpt4_abstractive_summary)

    print(5*"*"*50)