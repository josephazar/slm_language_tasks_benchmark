import pandas as pd  
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time
import argparse
import os
import sys

def process_with_model(model_name):
    """
    Process all tasks with a specific model
    
    Args:
        model_name (str): Name of the Ollama model to use
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING WITH MODEL: {model_name}")
    print(f"{'='*50}\n")
    
    # Load data
    print("Loading data.csv...")
    df = pd.read_csv("data.csv")
    print(f"Loaded {len(df)} rows")
    
    # Initialize model
    print(f"Initializing model {model_name}...")
    chat = OllamaLLM(model=model_name)
    
    # Create safe model name for column headers
    safe_model_name = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
    
    # Create output filename
    filename = f"data_{safe_model_name}.csv"
    
    # Define prompt templates
    template_string1 = """Translate in English the text in that is delimited by triple backticks 
text: ```{text}```  (please to not add any note at the end of the translation)
"""

    template_string2 = """Make a journal title in English of the text delimited by triple backticks text: ```{text}``` (I want only one title)  
"""

    template_string3 = """Answer the question delimited by triple backticks 
text: ```{question}``` in 3 sentences (max) in English according to the text delimited by triple backticks 
text: ```{text}```  (please to not add any note at the end of the translation)
"""

    # TRANSLATION
    print("\nStarting translation task...")
    translations = []
    translation_prompt = ChatPromptTemplate.from_template(template_string1)

    for i, text in enumerate(df["text"]):
        if not isinstance(text, str) or len(text.strip()) == 0:
            print(f"Skipping empty text at row {i}")
            translations.append("")
            continue
            
        print(f"Processing translation {i+1}/{len(df)}")
        try:
            # Truncate very long texts
            if len(text) > 8000:
                text = text[:8000] + "..."
                
            customer_messages = translation_prompt.format_messages(text=text)
            result = chat.invoke(customer_messages)
            translations.append(result)
        except Exception as e:
            print(f"Error: {e}")
            translations.append(f"Error: {str(e)[:100]}")
    
    # Add to dataframe and save intermediate results
    df[f"translate_{safe_model_name}"] = translations
    df.to_csv(filename, index=False)
    print(f"Saved translations to {filename}")

    # SUMMARIZATION
    print("\nStarting summarization task...")
    summaries = []
    summary_prompt = ChatPromptTemplate.from_template(template_string2)

    for i, text in enumerate(df["text"]):
        if not isinstance(text, str) or len(text.strip()) == 0:
            print(f"Skipping empty text at row {i}")
            summaries.append("")
            continue
            
        print(f"Processing summary {i+1}/{len(df)}")
        try:
            # Truncate very long texts
            if len(text) > 8000:
                text = text[:8000] + "..."
                
            customer_messages = summary_prompt.format_messages(text=text)
            result = chat.invoke(customer_messages)
            summaries.append(result)
        except Exception as e:
            print(f"Error: {e}")
            summaries.append(f"Error: {str(e)[:100]}")
    
    # Add to dataframe and save intermediate results
    df[f"summary_{safe_model_name}"] = summaries
    df.to_csv(filename, index=False)
    print(f"Saved summaries to {filename}")

    # QUESTION ANSWERING
    print("\nStarting question answering task...")
    answers = []
    qa_prompt = ChatPromptTemplate.from_template(template_string3)

    for i, (text, question) in enumerate(zip(df["text"], df["question"])):
        if not isinstance(question, str) or len(question.strip()) == 0:
            print(f"Skipping empty question at row {i}")
            answers.append("")
            continue
            
        if not isinstance(text, str) or len(text.strip()) == 0:
            print(f"Skipping empty text at row {i}")
            answers.append("")
            continue
            
        print(f"Processing QA {i+1}/{len(df)}")
        try:
            # Truncate very long texts
            if len(text) > 8000:
                text = text[:8000] + "..."
                
            customer_messages = qa_prompt.format_messages(question=question, text=text)
            result = chat.invoke(customer_messages)
            answers.append(result)
        except Exception as e:
            print(f"Error: {e}")
            answers.append(f"Error: {str(e)[:100]}")
    
    # Add to dataframe and save final results
    df[f"answer_{safe_model_name}"] = answers
    df.to_csv(filename, index=False)
    print(f"Saved answers to {filename}")
    
    print(f"\nAll tasks completed for model {model_name}")
    print(f"Results saved to {filename}")
    
    return filename

def main():
    parser = argparse.ArgumentParser(description="Process data with multiple Ollama models")
    parser.add_argument("--models", nargs="+", default=["llama3.2:1b", "deepseek-r1", "mistral"],
                        help="Model names to process (default: llama3.2:1b, deepseek-r1, mistral)")
    args = parser.parse_args()
    
    result_files = []
    
    for model in args.models:
        try:
            output_file = process_with_model(model)
            result_files.append(output_file)
        except Exception as e:
            print(f"Error processing model {model}: {e}")
    
    print("\nAll models processed!")
    print(f"Generated files: {', '.join(result_files)}")
    print("Run evaluate_results.py to compare model performances")

if __name__ == "__main__":
    main()