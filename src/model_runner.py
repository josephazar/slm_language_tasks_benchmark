import pandas as pd  
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time
import argparse
import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pydantic import BaseModel, Field
import re

# Define Pydantic models for structured outputs
class Translation(BaseModel):
    translation: str = Field(..., description="The English translation of the Albanian text")

class Summary(BaseModel):
    summary: str = Field(..., description="A comprehensive abstractive summary of the text")

class Answer(BaseModel):
    answer: str = Field(..., description="The answer to the question based on the provided text")

def clean_response(response):
    """
    Clean the response to remove any prefixes, introductions, or code block markers.
    
    Args:
        response (str): The raw response
        
    Returns:
        str: Cleaned response
    """
    if not response:
        return ""
        
    # Remove any prefixes like "Here is the translation:" or "Translation:"
    response = re.sub(r'^(here is|here\'s|the|ok|i\'ll|translation:|translated version:|here is the translation of.*?:|english translation:|in english:|summary:|answer:|this is|abstractive summary:)', '', response, flags=re.IGNORECASE)
    
    # Remove backticks and markdown code blocks
    response = re.sub(r'```[a-z]*\n', '', response)
    response = re.sub(r'```', '', response)
    response = re.sub(r'`', '', response)
    
    # Remove notes at the end
    response = re.sub(r'\(note:.*?\)', '', response, flags=re.IGNORECASE)
    response = re.sub(r'\n\s*note:.*$', '', response, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove any mentions of Albanian, English, etc.
    response = re.sub(r'albanian text translated to english:', '', response, flags=re.IGNORECASE)
    response = re.sub(r'from albanian to english:', '', response, flags=re.IGNORECASE)
    
    # Cleanup extra whitespace
    response = response.strip()
    
    return response

def process_with_model(model_name, subset_size=None):
    """
    Process all tasks with a specific model
    
    Args:
        model_name (str): Name of the Ollama model to use
        subset_size (int, optional): Number of rows to process for testing
        
    Returns:
        str: Path to the output file
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING WITH MODEL: {model_name}")
    print(f"{'='*50}\n")
    
    # Load data
    print("Loading data.csv...")
    df = pd.read_csv("data.csv")
    print(f"Loaded {len(df)} rows")
    
    # Use subset for testing if specified
    if subset_size and subset_size > 0:
        df = df.head(subset_size)
        print(f"Using subset of {subset_size} rows for testing")
    
    # Initialize model
    print(f"Initializing model {model_name}...")
    chat = OllamaLLM(model=model_name)
    
    # Create safe model name for column headers
    safe_model_name = model_name.replace(":", "_").replace("/", "_").replace(".", "_")
    
    # Create output filename
    filename = f"data_{safe_model_name}.csv"
    
    # Define prompt templates with structured output format
    translation_template = """
    You are a professional translator specializing in Albanian to English translation.
    
    Translate the Albanian text delimited by triple backticks into English.
    
    Albanian text: ```{text}```
    
    Provide ONLY the translation with no introductions, notes, or explanations.
    Format your response as a valid JSON object with a single field "translation" containing the English translation.
    """

    summary_template = """
    You are a professional text summarizer.
    
    Create a comprehensive abstractive summary (4-6 sentences) of the text delimited by triple backticks.
    An abstractive summary uses your own words rather than extracting sentences directly.
    
    Text: ```{text}```
    
    Provide ONLY the summary with no introductions, notes, or explanations.
    Format your response as a valid JSON object with a single field "summary" containing the abstractive summary.
    """

    qa_template = """
    Answer the question based on the provided context text.
    
    Context text: ```{text}```
    Question: ```{question}```
    
    Provide a comprehensive answer in 3-5 sentences.
    Format your response as a valid JSON object with a single field "answer" containing your answer.
    Provide ONLY the answer with no introductions, notes, or explanations.
    """

    # TRANSLATION
    print("\nStarting translation task...")
    translations = []
    translation_prompt = ChatPromptTemplate.from_template(translation_template)

    for i, text in enumerate(df["text"]):
        if not isinstance(text, str) or len(text.strip()) == 0:
            print(f"Skipping empty text at row {i}")
            translations.append("")
            continue
            
        print(f"Processing translation {i+1}/{len(df)}")
        try:
            # Skip very long texts
            if len(text) > 8000:
                print(f"Skipping row {i} - text too long ({len(text)} chars)")
                translations.append("SKIPPED - TEXT TOO LONG")
                continue
                
            customer_messages = translation_prompt.format_messages(text=text)
            raw_result = chat.invoke(customer_messages)
            
            # Try to parse JSON
            try:
                # Find JSON in the response
                json_match = re.search(r'({.*})', raw_result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result_obj = json.loads(json_str)
                    if isinstance(result_obj, dict) and "translation" in result_obj:
                        result = result_obj["translation"]
                    else:
                        result = clean_response(raw_result)
                else:
                    result = clean_response(raw_result)
            except json.JSONDecodeError:
                result = clean_response(raw_result)
                
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
    summary_prompt = ChatPromptTemplate.from_template(summary_template)

    for i, text in enumerate(df["text"]):
        if not isinstance(text, str) or len(text.strip()) == 0:
            print(f"Skipping empty text at row {i}")
            summaries.append("")
            continue
            
        print(f"Processing summary {i+1}/{len(df)}")
        try:
            # Skip very long texts
            if len(text) > 8000:
                print(f"Skipping row {i} - text too long ({len(text)} chars)")
                summaries.append("SKIPPED - TEXT TOO LONG")
                continue
                
            customer_messages = summary_prompt.format_messages(text=text)
            raw_result = chat.invoke(customer_messages)
            
            # Try to parse JSON
            try:
                # Find JSON in the response
                json_match = re.search(r'({.*})', raw_result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result_obj = json.loads(json_str)
                    if isinstance(result_obj, dict) and "summary" in result_obj:
                        result = result_obj["summary"]
                    else:
                        result = clean_response(raw_result)
                else:
                    result = clean_response(raw_result)
            except json.JSONDecodeError:
                result = clean_response(raw_result)
                
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
    qa_prompt = ChatPromptTemplate.from_template(qa_template)

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
            # Skip very long texts
            if len(text) > 8000:
                print(f"Skipping row {i} - text too long ({len(text)} chars)")
                answers.append("SKIPPED - TEXT TOO LONG")
                continue
                
            customer_messages = qa_prompt.format_messages(question=question, text=text)
            raw_result = chat.invoke(customer_messages)
            
            # Try to parse JSON
            try:
                # Find JSON in the response
                json_match = re.search(r'({.*})', raw_result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result_obj = json.loads(json_str)
                    if isinstance(result_obj, dict) and "answer" in result_obj:
                        result = result_obj["answer"]
                    else:
                        result = clean_response(raw_result)
                else:
                    result = clean_response(raw_result)
            except json.JSONDecodeError:
                result = clean_response(raw_result)
                
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
    parser.add_argument("--parallel", action="store_true", help="Run models in parallel")
    parser.add_argument("--subset", type=int, default=None, help="Process only a subset of rows (for testing)")
    args = parser.parse_args()
    
    result_files = []
    
    if args.parallel:
        print(f"Running {len(args.models)} models in parallel...")
        with ProcessPoolExecutor(max_workers=len(args.models)) as executor:
            future_to_model = {executor.submit(process_with_model, model, args.subset): model for model in args.models}
            for future in as_completed(future_to_model):
                model = future_to_model[future]
                try:
                    output_file = future.result()
                    result_files.append(output_file)
                except Exception as e:
                    print(f"Error processing model {model}: {e}")
    else:
        print(f"Running {len(args.models)} models sequentially...")
        for model in args.models:
            try:
                output_file = process_with_model(model, args.subset)
                result_files.append(output_file)
            except Exception as e:
                print(f"Error processing model {model}: {e}")
    
    print("\nAll models processed!")
    print(f"Generated files: {', '.join(result_files)}")
    print("Run evaluate_results.py to compare model performances")

if __name__ == "__main__":
    main()