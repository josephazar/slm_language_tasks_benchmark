import os
import pandas as pd
from dotenv import load_dotenv
import json
from openai import AzureOpenAI
from pydantic import BaseModel
from azure.ai.language.questionanswering import QuestionAnsweringClient
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel
from azure.ai.translation.text import TextTranslationClient
import re

# Load environment variables from .env file
load_dotenv()

# Initialize the Translator client globally
translator_client = TextTranslationClient(
    endpoint=os.getenv("TRANSLATOR_DOCUMENT_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("TRANSLATOR_KEY"))
)

# Step 1: Define translation function using Azure Translator SDK
def batch_texts(texts, max_chars=10000, max_texts=100):
    """
    Split a list of texts into batches based on character and text count limits.
    
    Args:
        texts (list): List of strings to batch.
        max_chars (int): Maximum total characters per batch (default: 10,000).
        max_texts (int): Maximum number of texts per batch (default: 100).
    
    Returns:
        list of lists: Batches of texts.
    """
    batches = []
    current_batch = []
    current_chars = 0
    
    for text in texts:
        text_len = len(text) if text else 0
        # Start a new batch if adding the next text exceeds limits
        if len(current_batch) >= max_texts or current_chars + text_len > max_chars:
            batches.append(current_batch)
            current_batch = [text]
            current_chars = text_len
        else:
            current_batch.append(text)
            current_chars += text_len
    
    if current_batch:  # Add the last batch if it contains any texts
        batches.append(current_batch)
    
    return batches

def translate_text(texts, from_lang="sq", to_lang="en"):
    """
    Translate a list of texts from one language to another using Azure Translator with batching.
    
    Args:
        texts (list): List of strings to translate.
        from_lang (str): Source language code (default: Albanian "sq").
        to_lang (str): Target language code (default: English "en").
    
    Returns:
        list: Translated texts.
    """
    if not texts:
        return []
    
    # Split texts into batches
    batches = batch_texts(texts)
    all_translations = []
    
    for batch in batches:
        try:
            response = translator_client.translate(
                body=batch,
                from_language=from_lang,
                to_language=[to_lang]
            )
            translations = [item.translations[0].text if item.translations else "" for item in response]
            all_translations.extend(translations)
        except Exception as e:
            print(f"Translation error for batch: {e}")
            all_translations.extend([""] * len(batch))  # Add empty strings if batch fails
    
    return all_translations

    
# Step 2: Define Pydantic model for structured question output
class QuestionResponse(BaseModel):
    question: str

def generate_question(text_en):
    """
    Generate a question based on the English translated text using GPT-4o.
    
    Args:
        text_en (str): English translated text.
    
    Returns:
        str: Generated question or None if generation fails.
    """
    # Skip empty or invalid input
    if not text_en or not text_en.strip():
        print("Skipping empty or invalid text.")
        return None
    
    # Initialize the Azure OpenAI client
    openai_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    )
    
    # Define the prompt
    prompt = (
        "Based on the following text, suggest a question that could be asked about it. "
        "Respond with a JSON object containing a single field 'question'.\n\n"
        f"{text_en}"
    )
    
    try:
        # Make the API call
        response = openai_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=100,
        )
        
        # Check the response content
        content = response.choices[0].message.content.strip()
        if not content:
            print("Empty response from API.")
            return None
        
        # Parse the JSON response
        # Extract JSON from code block
        json_match = re.search(r'```json\n(.*)\n```', content, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)  # Get the content between the markers
            question_json = json.loads(json_content)  # Parse the JSON
            return question_json["question"]  # Return the question string
        else:
            print("No JSON found in response:", content)
            return None
    
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}. Response content: '{content}'")
        return None
    except Exception as e:
        print(f"Error generating question for text: {e}")
        return None

# Step 4: Define function to get answer using Azure Language Service
def get_answer(question, text_en):
    """
    Answer the generated question using the English text as context.
    
    Args:
        question (str): The question to answer.
        text_en (str): English text as context.
    
    Returns:
        tuple: (answer, confidence) or (None, None) if no answer.
    """
    if not question or not text_en:
        return None, None
    
    language_endpoint = os.getenv("LANGUAGE_SERVICE_ENDPOINT")
    language_key = os.getenv("LANGUAGE_SERVICE_KEY")
    credential = AzureKeyCredential(language_key)
    qa_client = QuestionAnsweringClient(language_endpoint, credential)
    
    try:
        response = qa_client.get_answers_from_text(
            question=question,
            text_documents=[{"id": "1", "text": text_en}]
        )
        if response.answers:
            top_answer = max(response.answers, key=lambda x: x.confidence)
            return top_answer.answer, top_answer.confidence
        return None, None
    except Exception as e:
        print(f"Error getting answer for question '{question}': {e}")
        return None, None

# Step 5: Main processing
def main():
    # Load the existing data.csv
    df = pd.read_csv("data.csv")
    print(f"Loaded 'data.csv' with {len(df)} rows.")

    # Translate 'text' column
    print("Translating 'text' to English...")
    df['text_en'] = translate_text(df['text'].tolist(), from_lang="sq", to_lang="en")

    # Translate 'summary' column
    print("Translating 'summary' to English...")
    df['summary_en'] = translate_text(df['summary'].tolist(), from_lang="sq", to_lang="en")

    # Generate questions based on "text_en"
    print("Generating questions...")
    df["question"] = df["text_en"].apply(generate_question)

    # Get answers and confidences for each question
    print("Answering questions...")
    answers = []
    confidences = []
    for index, row in df.iterrows():
        if index % 10 == 0:
            print(f"Processing row {index}...")
        answer, confidence = get_answer(row["question"], row["text_en"])
        answers.append(answer)
        confidences.append(confidence)
    
    df["answer"] = answers
    df["answer_confidence"] = confidences

    # Save the updated DataFrame to data.csv
    df.to_csv("data.csv", index=False)
    print("Updated 'data.csv' with new columns: 'text_en', 'summary_en', 'question', 'answer', 'answer_confidence'.")

if __name__ == "__main__":
    main()