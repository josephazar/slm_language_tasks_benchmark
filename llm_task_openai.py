# Define template strings for the different tasks
template_string1 = """Translate in English the text in that is delimited by triple backticks 
text: ```{text}```  (please to not add any note at the end of the translation. It is mandatory to write only in English.)
"""

template_string2 = """Make a journal title in English of the text delimited by triple backticks 
text: ```{text}``` (I expressly want only one title with one sentence,
 I expressly don't want any comment and I don't want to see the keyword "title" and I don't want to see some "*". 
 It is mandatory to write only 1 title of 1 sentence in English. I expressely don't want to see many empty lines. 
 I don't want any explanation. please answer with 1 title only).
"""

template_string3 = """Answer the question delimited by triple backticks 
text: ```{question}``` in 3 sentences (max) in English according to the text delimited by triple backticks 
text: ```{text}```  (please to not add any note at the end of the translation. I expressly want only 3 sentences 
and I don't want to see some "*". I expressely want complete sentences. It is mandatory to write only in English.)
"""

import pandas as pd  
import os
import time
import sys
from openai import AzureOpenAI
import instructor
from pydantic import BaseModel
import re
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type
import json
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Load the data
print("Starting to load data...")
try:
    df = pd.read_csv("data.csv")
    print(f"Data loaded successfully. Shape: {df.shape}")
    print(df.head())
except Exception as e:
    print(f"ERROR loading CSV: {e}")
    sys.exit(1)

# Define which Azure OpenAI model to use
model_name = "gpt-4o-mini"
model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-4o-mini")
filename = "data_" + model_name + ".csv"

# Check if environment variables are set
print("Checking environment variables...")
required_env_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_API_ENDPOINT"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    print(f"ERROR: Missing required environment variables: {missing_vars}")
    sys.exit(1)
print("All required environment variables found.")

# Initialize the Azure OpenAI client
print("Initializing Azure OpenAI client...")
try:
    # Create the base client first (without instructor)
    base_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        timeout=30.0,  # Add timeout to prevent hanging
    )
    
    # Then create the instructor-enhanced client for structured responses
    client = instructor.from_openai(base_client)
    print("Azure OpenAI client initialized successfully.")
except Exception as e:
    print(f"ERROR initializing Azure OpenAI client: {e}")
    sys.exit(1)

# Define retry decorator for handling TPM errors
@retry(
    retry=retry_if_exception_type((Exception)),  # Retry on any exception since TPM errors could be various types
    wait=wait_fixed(10),  # Wait 10 seconds between retries
    stop=stop_after_attempt(5)  # Stop after 5 attempts
)
def call_api_with_retry(model, response_model, messages, max_tokens):
    print(f"Calling API with model {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            response_model=response_model,
            messages=messages,
            max_tokens=max_tokens,
            timeout=30.0,  # Add timeout to prevent hanging
        )
        print("API call successful")
        return response
    except Exception as e:
        print(f"API call failed: {e}")
        raise  # Re-raise for retry mechanism

# Create response models for each task
class TranslationResponse(BaseModel):
    translation: str

class SummaryResponse(BaseModel):
    title: str

class AnswerResponse(BaseModel):
    answer: str

# Test API connection before starting main processing
print("Testing API connection...")
try:
    # Use the base client (without instructor) for the test
    test_message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello"}
    ]
    test_response = base_client.chat.completions.create(
        model=model_name,
        messages=test_message,
        max_tokens=10,
    )
    print(f"API test successful. Model: {model_name}")
except Exception as e:
    print(f"ERROR testing API connection: {e}")
    print("Please check your Azure OpenAI credentials and model availability.")
    sys.exit(1)


# Main execution block
if __name__ == "__main__":
    print("=== Azure OpenAI Processing Script ===")
    print(f"Running with model: {model_name}")
    
    # You can uncomment this to process a smaller subset for testing
    # print("WARNING: Running in TEST MODE with limited rows")
    # df = df.head(3)
    # Check for texts exceeding the character limit
    print("Checking text lengths...")
    character_limit = 4300
    text_lengths = df["text"].apply(len)
    exceeding_limit = text_lengths > character_limit

    if exceeding_limit.any():
        print(f"WARNING: {exceeding_limit.sum()} texts exceed the {character_limit} character limit and will be skipped.")
        print(f"Text indices exceeding limit: {list(df.index[exceeding_limit])}")    
    try:
        ################## TRANSLATE
        print("\n===== STARTING TRANSLATION PROCESSING =====")
        answer = []
        item_counter = 0

        for index, row in df.iterrows():
            t = row["text"]
            
            # Skip texts exceeding the character limit
            if len(t) > character_limit:
                print(f"Skipping item {index+1}/{len(df)}: Text exceeds {character_limit} character limit ({len(t)} chars)")
                answer.append(f"SKIPPED: Text exceeds {character_limit} character limit")
                continue
                
            print(f"Processing item {index+1}/{len(df)}: Translation")
            
            # Add sleep after every 5 items
            item_counter += 1
            if item_counter % 5 == 0:
                print(f"Processed {item_counter} items. Sleeping for 10 seconds to avoid rate limits...")
                for i in range(10, 0, -1):
                    sys.stdout.write(f"\rResuming in {i} seconds...")
                    sys.stdout.flush()
                    time.sleep(1)
                print("\nResuming processing...")
            
            system_message = {"role": "system", "content": "You are a helpful assistant that translates text to English."}
            user_message = {"role": "user", "content": template_string1.format(text=t)}
            
            try:
                response = call_api_with_retry(
                    model=model_name,
                    response_model=TranslationResponse,
                    messages=[system_message, user_message],
                    max_tokens=1000,
                )
                
                print(f"TRANSLATION RESULT ({index+1}/{len(df)}): {response.translation[:50]}...")
                answer.append(response.translation)
            except Exception as e:
                error_msg = f"Failed after multiple retries: {e}"
                print(f"ERROR on item {index+1}: {error_msg}")
                answer.append(f"ERROR: {str(e)}")
            
            # Save intermediate results every 10 items
            if (index + 1) % 10 == 0:
                print(f"Saving intermediate results after {index+1} items...")
                temp_df = df.copy()
                temp_df["translate_" + model_name] = answer + [""] * (len(df) - len(answer))
                temp_df.to_csv(f"intermediate_{filename}")

        df["translate_" + model_name] = answer

        ################## SUMMARY
        print("\n===== STARTING SUMMARY PROCESSING =====")
        answer = []
        item_counter = 0

        for index, row in df.iterrows():
            t = row["text"]
            
            # Skip texts exceeding the character limit
            if len(t) > character_limit:
                print(f"Skipping item {index+1}/{len(df)}: Text exceeds {character_limit} character limit ({len(t)} chars)")
                answer.append(f"SKIPPED: Text exceeds {character_limit} character limit")
                continue
                
            print(f"Processing item {index+1}/{len(df)}: Summary")
            
            # Add sleep after every 5 items
            item_counter += 1
            if item_counter % 5 == 0:
                print(f"Processed {item_counter} items. Sleeping for 10 seconds to avoid rate limits...")
                for i in range(10, 0, -1):
                    sys.stdout.write(f"\rResuming in {i} seconds...")
                    sys.stdout.flush()
                    time.sleep(1)
                print("\nResuming processing...")
            
            system_message = {"role": "system", "content": "You are a helpful assistant that creates concise journal titles in English."}
            user_message = {"role": "user", "content": template_string2.format(text=t)}
            
            try:
                response = call_api_with_retry(
                    model=model_name,
                    response_model=SummaryResponse,
                    messages=[system_message, user_message],
                    max_tokens=100,
                )
                
                print(f"SUMMARY RESULT ({index+1}/{len(df)}): {response.title}")
                answer.append(response.title)
            except Exception as e:
                error_msg = f"Failed after multiple retries: {e}"
                print(f"ERROR on item {index+1}: {error_msg}")
                answer.append(f"ERROR: {str(e)}")
            
            # Save intermediate results every 10 items
            if (index + 1) % 10 == 0:
                print(f"Saving intermediate results after {index+1} items...")
                temp_df = df.copy()
                if "translate_" + model_name in temp_df.columns:
                    # Keep existing translation results
                    pass
                temp_df["summary_" + model_name] = answer + [""] * (len(df) - len(answer))
                temp_df.to_csv(f"intermediate_{filename}")

        df["summary_" + model_name] = answer

        ################## ANSWER
        print("\n===== STARTING QUESTION ANSWERING PROCESSING =====")
        answer = []
        item_counter = 0

        for index, row in df.iterrows():
            t = row["text"]
            q = row["question"]
            
            # Skip texts exceeding the character limit
            if len(t) > character_limit:
                print(f"Skipping item {index+1}/{len(df)}: Text exceeds {character_limit} character limit ({len(t)} chars)")
                answer.append(f"SKIPPED: Text exceeds {character_limit} character limit")
                continue
                
            print(f"Processing item {index+1}/{len(df)}: Question Answering")
            
            # Add sleep after every 5 items
            item_counter += 1
            if item_counter % 5 == 0:
                print(f"Processed {item_counter} items. Sleeping for 10 seconds to avoid rate limits...")
                for i in range(10, 0, -1):
                    sys.stdout.write(f"\rResuming in {i} seconds...")
                    sys.stdout.flush()
                    time.sleep(1)
                print("\nResuming processing...")
            
            system_message = {"role": "system", "content": "You are a helpful assistant that answers questions based on provided text."}
            user_message = {"role": "user", "content": template_string3.format(question=q, text=t)}
            
            try:
                response = call_api_with_retry(
                    model=model_name,
                    response_model=AnswerResponse,
                    messages=[system_message, user_message],
                    max_tokens=300,
                )
                
                print(f"ANSWER RESULT ({index+1}/{len(df)}): {response.answer[:50]}...")
                answer.append(response.answer)
            except Exception as e:
                error_msg = f"Failed after multiple retries: {e}"
                print(f"ERROR on item {index+1}: {error_msg}")
                answer.append(f"ERROR: {str(e)}")
            
            # Save intermediate results every 10 items
            if (index + 1) % 10 == 0:
                print(f"Saving intermediate results after {index+1} items...")
                temp_df = df.copy()
                # Keep existing results for other columns
                temp_df["answer_" + model_name] = answer + [""] * (len(df) - len(answer))
                temp_df.to_csv(f"intermediate_{filename}")

        df["answer_" + model_name] = answer

        # Save final results
        print("\n===== SAVING FINAL RESULTS =====")
        df.to_csv(filename)
        print(f"Processing complete. Results saved to {filename}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving partial results...")
        # Save any results collected so far
        if "translate_" + model_name in locals() and len(answer) > 0:
            df["translate_" + model_name] = answer + [""] * (len(df) - len(answer))
        df.to_csv("interrupted_" + filename)
        print(f"Partial results saved to interrupted_{filename}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        # Try to save any results collected so far
        try:
            if "answer" in locals() and len(answer) > 0:
                if item_counter <= len(df):
                    df.loc[:item_counter-1, "current_process_" + model_name] = answer
            df.to_csv("error_" + filename)
            print(f"Partial results saved to error_{filename}")
        except:
            print("Could not save partial results")
        raise