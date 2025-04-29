#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced model evaluation script with semantic metrics for translation, summarization, and QA tasks.
Includes both reference-based and source-based evaluation for summarization tasks.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json
import time
import argparse
import sys
from datetime import datetime
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import ssl
from tqdm import tqdm
import matplotlib.patches as mpatches
import seaborn as sns

# For semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("SentenceTransformer not available. Using fallback similarity methods.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import spaCy for entity extraction
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load English model
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy model loaded successfully for entity extraction")
    except:
        print("Downloading spaCy model for entity extraction...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
except ImportError:
    print("spaCy not available. Entity extraction metrics will be skipped.")
    SPACY_AVAILABLE = False
    nlp = None

# Try to import BERTScore
try:
    import bert_score
    BERT_SCORE_AVAILABLE = True
    print("BERTScore available for evaluation")
except ImportError:
    print("BERTScore not available. BERTScore metrics will be skipped.")
    BERT_SCORE_AVAILABLE = False

# Helper function to convert NumPy types to native Python types for JSON serialization
def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

# Set up SSL context for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
print("Setting up NLTK resources...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("NLTK resources installed successfully")
except Exception as e:
    print(f"Warning: Error downloading NLTK resources: {e}")

# Fallback tokenizer
def simple_tokenize(text):
    """Simple tokenizer fallback that doesn't require NLTK resources"""
    if not isinstance(text, str):
        return []
    return re.findall(r'\b\w+\b', text.lower())

# Safe tokenize function
def safe_tokenize(text):
    """Safely tokenize text, falling back to simple tokenizer if NLTK fails"""
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        return nltk.word_tokenize(text.lower())
    except:
        return simple_tokenize(text)

# Remove thinking tags
def remove_thinking_tags(text):
    """Remove <think>...</think> tags and their content"""
    if not isinstance(text, str):
        return ""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

# Improved text cleaning
def clean_text(text):
    """Clean and normalize text for comparison"""
    if not isinstance(text, str):
        return ""
    # Remove thinking tags first
    text = remove_thinking_tags(text)
    # Convert to lowercase
    text = text.lower()
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove special characters but keep essential punctuation
    text = re.sub(r'[^\w\s.,?!;:-]', '', text)
    return text.strip()

# Entity extraction for information coverage
def extract_entities(text):
    """Extract entities from text using spaCy"""
    if not SPACY_AVAILABLE or not nlp:
        return set()
    if not isinstance(text, str) or not text.strip():
        return set()
    
    try:
        doc = nlp(text)
        # Extract named entities
        entities = set(ent.text.lower() for ent in doc.ents)
        # Extract noun chunks as additional key information
        noun_chunks = set(chunk.text.lower() for chunk in doc.noun_chunks)
        return entities.union(noun_chunks)
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return set()

# Calculate information coverage
def calculate_info_coverage(reference, candidate):
    """Calculate how much information from reference is covered in candidate"""
    if not SPACY_AVAILABLE:
        return 0.0
    
    if not reference or not candidate:
        return 0.0
    
    ref_entities = extract_entities(reference)
    cand_entities = extract_entities(candidate)
    
    if not ref_entities:
        return 0.0
    
    # Calculate coverage
    overlap = ref_entities.intersection(cand_entities)
    return len(overlap) / len(ref_entities)

# Calculate BERTScore
def calculate_bert_score(references, candidates):
    """Calculate BERTScore between references and candidates"""
    if not BERT_SCORE_AVAILABLE:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if not references or not candidates:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    try:
        P, R, F1 = bert_score.score(
            candidates, references, 
            lang="en", 
            verbose=False,
            rescale_with_baseline=True
        )
        
        # Convert torch tensors to Python floats
        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean())
        }
    except Exception as e:
        print(f"Error calculating BERTScore: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

def evaluate_models(baseline_file, model_files, output_dir="results"):
    print("=== Enhanced Model Evaluator with Semantic Metrics ===")
    
    # Create output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")

    samples_dir = os.path.join(output_dir, "samples")
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
        print(f"Created directory: {samples_dir}")
    
    # Initialize results storage
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }
    
    # Initialize scorers
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1
    
    # Initialize sentence transformer for semantic similarity
    semantic_model = None
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            print("Loading sentence transformer model...")
            semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            print("Sentence transformer model loaded successfully.")
        except Exception as e:
            print(f"Error loading sentence transformer: {e}")
            semantic_model = None
    
    # Load baseline data
    print(f"\nLoading baseline data from {baseline_file}...")
    try:
        baseline_df = pd.read_csv(baseline_file)
        print(f"Loaded baseline with {len(baseline_df)} rows and {len(baseline_df.columns)} columns")
    except Exception as e:
        print(f"Error loading baseline data: {e}")
        return
    
    # Process model files
    model_names = []
    model_dfs = {}
    
    for model_path in model_files:
        # Extract model name from file path
        model_file = os.path.basename(model_path)
        model_name = model_file.split('.')[0]  # Remove file extension
        
        print(f"\nLoading model: {model_name} from file {model_path}")
        
        try:
            model_df = pd.read_csv(model_path)
            print(f"Loaded model data with {len(model_df)} rows and {len(model_df.columns)} columns")
            # Sample the first few column names
            sample_cols = model_df.columns.tolist()[:min(5, len(model_df.columns))]
            print(f"Sample columns: {sample_cols}...")
            
            model_names.append(model_name)
            model_dfs[model_name] = model_df
            results["models"][model_name] = {}
        except Exception as e:
            print(f"Error loading model data: {e}")
            continue
    
    # Print some sample text to help debug
    def print_sample_comparisons(task_name, ref_texts, model_texts, n_samples=2):
        print(f"\n=== Sample {task_name} Comparisons ===")
        
        with open(os.path.join(samples_dir, f"{task_name}_samples.txt"), 'w') as f:
            for i in range(min(n_samples, len(ref_texts))):
                f.write(f"Sample {i+1}:\n")
                f.write(f"Reference:\n{ref_texts[i]}\n\n")
                for model_name, texts in model_texts.items():
                    if i < len(texts):
                        f.write(f"{model_name}:\n{texts[i]}\n\n")
                f.write("="*80 + "\n\n")
        
        print(f"Saved {task_name} sample comparisons to {os.path.join(samples_dir, f'{task_name}_samples.txt')}")
    

    # Calculate semantic similarity with better error handling
    def get_semantic_similarity(references, candidates):
        """Calculate semantic similarity using sentence embeddings"""
        if not semantic_model or not references or not candidates:
            return []
        
        similarities = []
        for ref, cand in zip(references, candidates):
            if not ref or not cand:
                similarities.append(0)
                continue
                
            try:
                # Get embeddings
                ref_embedding = semantic_model.encode([ref])[0]
                cand_embedding = semantic_model.encode([cand])[0]
                
                # Ensure embeddings are NumPy arrays
                if not isinstance(ref_embedding, np.ndarray):
                    ref_embedding = np.array(ref_embedding)
                if not isinstance(cand_embedding, np.ndarray):
                    cand_embedding = np.array(cand_embedding)
                
                # Ensure embeddings are 1D
                ref_embedding = ref_embedding.flatten()
                cand_embedding = cand_embedding.flatten()
                
                # Calculate cosine similarity directly
                dot_product = np.dot(ref_embedding, cand_embedding)
                norm_ref = np.linalg.norm(ref_embedding)
                norm_cand = np.linalg.norm(cand_embedding)
                
                if norm_ref == 0 or norm_cand == 0:
                    similarities.append(0)
                    continue
                    
                similarity = dot_product / (norm_ref * norm_cand)
                similarities.append(float(similarity))
            except Exception as e:
                print(f"Error calculating semantic similarity: {e}")
                similarities.append(0)
        
        return similarities

    # Alternative implementation for source-based evaluation
    def get_semantic_similarity_for_source(source, summary):
        """Calculate semantic similarity between source text and summary"""
        if not semantic_model or not source or not summary:
            return 0.0
            
        try:
            # For longer source texts, break into paragraphs
            if len(source) > 1000:
                paragraphs = re.split(r'\n+', source)
                paragraph_scores = []
                
                for paragraph in paragraphs:
                    if len(paragraph.strip()) < 10:  # Skip very short paragraphs
                        continue
                        
                    try:
                        # Get embeddings and convert to numpy arrays
                        para_embedding = np.array(semantic_model.encode([paragraph])[0])
                        summary_embedding = np.array(semantic_model.encode([summary])[0])
                        
                        # Flatten embeddings
                        para_embedding = para_embedding.flatten()
                        summary_embedding = summary_embedding.flatten()
                        
                        # Calculate similarity
                        dot_product = np.dot(para_embedding, summary_embedding)
                        norm_para = np.linalg.norm(para_embedding)
                        norm_summary = np.linalg.norm(summary_embedding)
                        
                        if norm_para == 0 or norm_summary == 0:
                            continue
                            
                        similarity = dot_product / (norm_para * norm_summary)
                        paragraph_scores.append(float(similarity))
                    except Exception as e:
                        print(f"    Paragraph similarity error: {e}")
                
                # Take the average of top 3 paragraph similarities
                if paragraph_scores:
                    paragraph_scores.sort(reverse=True)
                    return np.mean(paragraph_scores[:3])
                else:
                    return 0.0
            else:
                # For shorter sources, compare directly
                source_embedding = np.array(semantic_model.encode([source])[0])
                summary_embedding = np.array(semantic_model.encode([summary])[0])
                
                # Flatten embeddings
                source_embedding = source_embedding.flatten()
                summary_embedding = summary_embedding.flatten()
                
                # Calculate similarity
                dot_product = np.dot(source_embedding, summary_embedding)
                norm_source = np.linalg.norm(source_embedding)
                norm_summary = np.linalg.norm(summary_embedding)
                
                if norm_source == 0 or norm_summary == 0:
                    return 0.0
                    
                return float(dot_product / (norm_source * norm_summary))
        except Exception as e:
            print(f"    Error calculating source similarity: {e}")
            return 0.0
    # Calculate cosine similarity manually if SentenceTransformer is not available
    def cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors"""
        # Reshape vectors to ensure they're 1D if they come as 2D arrays
        if len(a.shape) > 1:
            a = a.reshape(-1)
        if len(b.shape) > 1:
            b = b.reshape(-1)
            
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return np.array([[0]])
            
        similarity = dot_product / (norm_a * norm_b)
        return np.array([[similarity]])
    

    # Calculate containment score
    def calculate_containment(reference, candidate):
        """Calculate how much of reference is contained in candidate"""
        if not reference or not candidate:
            return 0.0
            
        ref_tokens = set(safe_tokenize(reference))
        cand_tokens = set(safe_tokenize(candidate))
        
        if not ref_tokens:
            return 0.0
            
        # Calculate containment as a proportion of reference tokens found in candidate
        common_tokens = ref_tokens.intersection(cand_tokens)
        return len(common_tokens) / len(ref_tokens) if ref_tokens else 0.0
    
    # Function to evaluate each task
    def evaluate_task(task_name, ref_column, model_column_pattern, metrics_fn):
        print(f"\n--- Evaluating {task_name} ---")
        
        all_model_texts = {}
        reference_texts = []
        
        for model_name, model_df in model_dfs.items():
            # Find the relevant column for this model
            model_columns = [col for col in model_df.columns if model_column_pattern in col.lower()]
            
            if not model_columns:
                print(f"  No {task_name} column found for {model_name}")
                results["models"][model_name][task_name.lower()] = {metric: 0 for metric in metrics_fn([], [])}
                continue
                
            model_column = model_columns[0]
            print(f"  Found {task_name} column for {model_name}: {model_column}")
            
            # Collect valid pairs
            valid_refs = []
            valid_cands = []
            all_model_texts[model_name] = []
            
            # Only collect reference texts for the first model
            if not reference_texts:
                for i, ref in enumerate(baseline_df[ref_column]):
                    if i < len(baseline_df) and pd.notna(ref):
                        clean_ref = clean_text(ref)
                        reference_texts.append(ref)  # Keep original for samples
            
            for i, (ref, cand) in enumerate(zip(baseline_df[ref_column], model_df[model_column])):
                if i >= len(model_df) or pd.isna(cand) or (isinstance(cand, str) and "SKIPPED" in cand):
                    continue
                    
                clean_ref = clean_text(ref)
                clean_cand = clean_text(cand)
                
                all_model_texts[model_name].append(cand)  # Keep original for samples
                valid_refs.append(clean_ref)
                valid_cands.append(clean_cand)
            
            if not valid_refs or not valid_cands:
                print(f"  No valid {task_name} pairs found for {model_name}")
                results["models"][model_name][task_name.lower()] = {metric: 0 for metric in metrics_fn([], [])}
                continue
            
            # Calculate metrics
            try:
                task_metrics = metrics_fn(valid_refs, valid_cands)
                results["models"][model_name][task_name.lower()] = task_metrics
                
                # Print results
                for metric, value in task_metrics.items():
                    print(f"  {model_name} {metric}: {value:.4f}")
            except Exception as e:
                print(f"  Error evaluating {task_name} for {model_name}: {e}")
                results["models"][model_name][task_name.lower()] = {metric: 0 for metric in metrics_fn([], [])}
        
        # Print sample comparisons
        print_sample_comparisons(task_name, reference_texts, all_model_texts)
    
    # Function to evaluate summarization against source text
    def evaluate_summarization_with_source(baseline_df, model_dfs, model_names, results):
        """Evaluate summaries against the original text rather than reference summaries"""
        print("\n--- Evaluating Summarization Against Source Text ---")
        
        if "text" not in baseline_df.columns and "text_en" not in baseline_df.columns:
            print("  Source text column not found. Skipping source-based evaluation.")
            return
            
        source_column = "text_en" if "text_en" in baseline_df.columns else "text"
        print(f"  Using source column: {source_column}")
        
        # Find summary columns in each model
        all_model_summaries = {}
        
        for model_name, model_df in model_dfs.items():
            model_columns = [col for col in model_df.columns if "summary_llm" in col.lower()]
            
            if not model_columns:
                print(f"  No summary column found for {model_name}")
                results["models"][model_name]["summarization_source"] = {
                    "source_coverage": 0,
                    "semantic_relevance": 0,
                    "conciseness": 0,
                    "bertscore_f1": 0
                }
                continue
                
            model_column = model_columns[0]
            print(f"  Found summary column for {model_name}: {model_column}")
            
            # Collect valid pairs (source, summary)
            valid_sources = []
            valid_summaries = []
            all_model_summaries[model_name] = []
            
            for i, (source, summary) in enumerate(zip(baseline_df[source_column], model_df[model_column])):
                if i >= len(model_df) or pd.isna(summary) or (isinstance(summary, str) and "SKIPPED" in summary):
                    continue
                    
                clean_source = clean_text(source)
                clean_summary = clean_text(summary)
                
                all_model_summaries[model_name].append(summary)  # Keep original for samples
                valid_sources.append(clean_source)
                valid_summaries.append(clean_summary)
            
            if not valid_sources or not valid_summaries:
                print(f"  No valid source-summary pairs found for {model_name}")
                results["models"][model_name]["summarization_source"] = {
                    "source_coverage": 0,
                    "semantic_relevance": 0,
                    "conciseness": 0,
                    "bertscore_f1": 0
                }
                continue
            
            # Calculate metrics
            metrics = {}
            
            # 1. Information coverage (using entity extraction)
            if SPACY_AVAILABLE:
                coverage_scores = []
                for source, summary in zip(valid_sources, valid_summaries):
                    source_entities = extract_entities(source)
                    summary_entities = extract_entities(summary)
                    
                    if not source_entities:
                        coverage_scores.append(0)
                        continue
                        
                    # Calculate what percentage of important entities from source are in summary
                    overlap = source_entities.intersection(summary_entities)
                    coverage = len(overlap) / len(source_entities)
                    coverage_scores.append(coverage)
                
                metrics["source_coverage"] = np.mean(coverage_scores) if coverage_scores else 0
            else:
                metrics["source_coverage"] = 0
            
            # 2. Semantic relevance (using embeddings)
            if semantic_model:
                relevance_scores = []
                for source, summary in zip(valid_sources, valid_summaries):
                    relevance_scores.append(get_semantic_similarity_for_source(source, summary))
                
                metrics["semantic_relevance"] = np.mean(relevance_scores) if relevance_scores else 0
            else:
                metrics["semantic_relevance"] = 0
                        
            # 3. Conciseness (ratio of summary length to source length)
            conciseness_scores = []
            for source, summary in zip(valid_sources, valid_summaries):
                source_len = len(source.split())
                summary_len = len(summary.split())
                
                if source_len == 0:
                    conciseness_scores.append(0)
                    continue
                
                # Ideal compression ratio might vary, but typically 10-20% is good
                # We'll calculate a score that peaks at around 15% compression
                ratio = summary_len / source_len
                
                # Score peaks at 0.15 (15%) and decreases as it gets further from this ideal
                # This formula gives 1.0 at ratio=0.15, and decreases toward 0 as ratio approaches 0 or 0.3
                conciseness = max(0, 1 - abs(ratio - 0.15) / 0.15)
                conciseness_scores.append(conciseness)
            
            metrics["conciseness"] = np.mean(conciseness_scores) if conciseness_scores else 0
            
            # 4. BERTScore for factual alignment
            if BERT_SCORE_AVAILABLE:
                try:
                    # BERTScore between summaries and source texts
                    # For long sources, we'll score against key sentences
                    modified_sources = []
                    for source in valid_sources:
                        if len(source) > 1000:
                            # Extract first and last few sentences as key content
                            sentences = re.split(r'[.!?]+', source)
                            key_sentences = sentences[:2] + sentences[-2:] if len(sentences) > 4 else sentences
                            modified_sources.append(' '.join(key_sentences))
                        else:
                            modified_sources.append(source)
                    
                    bertscore_results = calculate_bert_score(modified_sources, valid_summaries)
                    metrics["bertscore_f1"] = bertscore_results["f1"]
                except Exception as e:
                    print(f"    Error calculating BERTScore: {e}")
                    metrics["bertscore_f1"] = 0
            else:
                metrics["bertscore_f1"] = 0
            
            # Store metrics
            results["models"][model_name]["summarization_source"] = metrics
            
            # Print results
            for metric, value in metrics.items():
                print(f"  {model_name} {metric}: {value:.4f}")
        
        # Print sample comparisons
        if baseline_df[source_column].iloc[0]:  # Ensure there's at least one source text
            source_samples = []
            for i in range(min(3, len(baseline_df))):
                if pd.notna(baseline_df[source_column].iloc[i]):
                    source_samples.append(baseline_df[source_column].iloc[i])
            
            with open(os.path.join(samples_dir, "summarization_source_samples.txt"), 'w') as f:
                for i, source in enumerate(source_samples):
                    if i >= len(source_samples):
                        break
                        
                    f.write(f"Sample {i+1}:\n")
                    f.write(f"Source text:\n{source[:1000]}...\n\n")
                    
                    for model_name, summaries in all_model_summaries.items():
                        if i < len(summaries):
                            f.write(f"{model_name} summary:\n{summaries[i]}\n\n")
                    
                    f.write("="*80 + "\n\n")
            
            print(f"Saved source-summary samples to {os.path.join(samples_dir, 'summarization_source_samples.txt')}")
    
    # Define metrics functions for each task
    def translation_metrics(references, candidates):
        if not references or not candidates:
            return {"bleu": 0, "semantic_sim": 0}
            
        # Calculate BLEU scores
        bleu_scores = []
        for ref, cand in zip(references, candidates):
            if not ref or not cand:
                bleu_scores.append(0)
                continue
                
            ref_tokens = safe_tokenize(ref)
            cand_tokens = safe_tokenize(cand)
            
            if not ref_tokens or not cand_tokens:
                bleu_scores.append(0)
                continue
                
            try:
                score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
                bleu_scores.append(score)
            except:
                bleu_scores.append(0)
        
        # Calculate semantic similarity
        semantic_scores = get_semantic_similarity(references, candidates) if semantic_model else []
        
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        avg_semantic = np.mean(semantic_scores) if semantic_scores else 0
        
        return {
            "bleu": avg_bleu,
            "semantic_sim": avg_semantic
        }
    
    def summarization_metrics(references, candidates):
        if not references or not candidates:
            return {
                "rouge1": 0, 
                "rouge2": 0, 
                "rougeL": 0, 
                "semantic_sim": 0,
                "info_coverage": 0,
                "bertscore_f1": 0
            }
        
        # Calculate ROUGE scores
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, cand in zip(references, candidates):
            if not ref or not cand:
                rouge1_scores.append(0)
                rouge2_scores.append(0)
                rougeL_scores.append(0)
                continue
                
            try:
                scores = rouge_scorer_instance.score(ref, cand)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except:
                rouge1_scores.append(0)
                rouge2_scores.append(0)
                rougeL_scores.append(0)
        
        # Calculate semantic similarity
        semantic_scores = get_semantic_similarity(references, candidates) if semantic_model else []
        
        # Calculate information coverage
        info_coverage_scores = []
        if SPACY_AVAILABLE:
            for ref, cand in zip(references, candidates):
                if not ref or not cand:
                    info_coverage_scores.append(0)
                    continue
                info_coverage_scores.append(calculate_info_coverage(ref, cand))
        
        # Calculate BERTScore
        bertscore_metrics = {"f1": 0.0}
        if BERT_SCORE_AVAILABLE:
            bertscore_metrics = calculate_bert_score(references, candidates)
        
        avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0
        avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0
        avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0
        avg_semantic = np.mean(semantic_scores) if semantic_scores else 0
        avg_info_coverage = np.mean(info_coverage_scores) if info_coverage_scores else 0
        
        return {
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL,
            "semantic_sim": avg_semantic,
            "info_coverage": avg_info_coverage,
            "bertscore_f1": bertscore_metrics["f1"]
        }
    
    def qa_metrics(references, candidates):
        if not references or not candidates:
            return {"semantic_sim": 0, "containment": 0, "info_coverage": 0}
        
        # Calculate semantic similarity
        semantic_scores = get_semantic_similarity(references, candidates) if semantic_model else []
        
        # Calculate containment scores
        containment_scores = [calculate_containment(ref, cand) for ref, cand in zip(references, candidates)]
        
        # Calculate information coverage
        info_coverage_scores = []
        if SPACY_AVAILABLE:
            for ref, cand in zip(references, candidates):
                if not ref or not cand:
                    info_coverage_scores.append(0)
                    continue
                info_coverage_scores.append(calculate_info_coverage(ref, cand))
        
        avg_semantic = np.mean(semantic_scores) if semantic_scores else 0
        avg_containment = np.mean(containment_scores) if containment_scores else 0
        avg_info_coverage = np.mean(info_coverage_scores) if info_coverage_scores else 0
        
        return {
            "semantic_sim": avg_semantic,
            "containment": avg_containment,
            "info_coverage": avg_info_coverage
        }
    
    # Evaluate translation
    evaluate_task(
        task_name="Translation",
        ref_column="text_en",
        model_column_pattern="translate_llm",
        metrics_fn=translation_metrics
    )
    
    # Evaluate summarization against reference summaries
    if "summary_en" in baseline_df.columns:
        ref_column = "summary_en"
    elif "extractive_summary_en" in baseline_df.columns:
        ref_column = "extractive_summary_en"
    else:
        ref_column = next((col for col in baseline_df.columns if "summary" in col.lower()), None)
    
    if ref_column:
        print(f"Using reference column for summarization: {ref_column}")
        evaluate_task(
            task_name="Summarization",
            ref_column=ref_column,
            model_column_pattern="summary_llm",
            metrics_fn=summarization_metrics
        )
    else:
        print("No suitable reference column found for summarization evaluation")
    
    # Also evaluate summarization against source text (new approach)
    evaluate_summarization_with_source(baseline_df, model_dfs, model_names, results)
    
    # Evaluate QA
    evaluate_task(
        task_name="QA",
        ref_column="answer",
        model_column_pattern="answer_llm",
        metrics_fn=qa_metrics
    )
    
    # Save results
    if model_names:
        # Save JSON results
        results_file = os.path.join(output_dir, f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        # Convert numpy types to native Python types
        serializable_results = convert_to_json_serializable(results)
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nSaved evaluation results to {results_file}")
        
        # Generate plots
        print("\nGenerating comparison plots...")
        
        # Create plots with error bars and more informative styling
        try:
            # Set plot style
            plt.style.use('ggplot')
            
            # 1. Translation Performance
            metrics = ["bleu", "semantic_sim"]
            plot_data = {metric: [] for metric in metrics}
            for model in model_names:
                for metric in metrics:
                    plot_data[metric].append(results["models"][model]["translation"][metric])
            
            fig, ax = plt.subplots(figsize=(12, 7))
            x = np.arange(len(model_names))
            width = 0.35
            
            for i, metric in enumerate(metrics):
                ax.bar(x + (i - 0.5) * width, plot_data[metric], width, label=metric.upper())
            
            ax.set_title("Translation Performance", fontsize=16)
            ax.set_ylabel("Score", fontsize=14)
            ax.set_ylim(0, max([max(plot_data[m]) for m in metrics]) * 1.2 or 1.0)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
            ax.legend(fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "translation_performance.png"))
            plt.close()
            
            # 2. Summarization Performance
            if "summarization" in results["models"][model_names[0]]:
                # Get all available metrics for summarization
                available_metrics = list(results["models"][model_names[0]]["summarization"].keys())
                
                # Group metrics into semantic and lexical for better visualization
                semantic_metrics = [m for m in available_metrics if "semantic" in m or "coverage" in m or "bertscore" in m]
                lexical_metrics = [m for m in available_metrics if "rouge" in m]
                
                # First plot: Semantic metrics
                plot_data = {metric: [] for metric in semantic_metrics}
                for model in model_names:
                    for metric in semantic_metrics:
                        plot_data[metric].append(results["models"][model]["summarization"][metric])
                
                fig, ax = plt.subplots(figsize=(12, 7))
                x = np.arange(len(model_names))
                width = 0.8 / len(semantic_metrics)
                
                for i, metric in enumerate(semantic_metrics):
                    offset = (i - len(semantic_metrics)/2 + 0.5) * width
                    ax.bar(x + offset, plot_data[metric], width, 
                           label=metric.upper().replace("_", " "))
                
                ax.set_title("Summarization Semantic Performance", fontsize=16)
                ax.set_ylabel("Score", fontsize=14)
                max_value = max([max(plot_data[m]) for m in semantic_metrics]) if plot_data else 0
                ax.set_ylim(0, max_value * 1.2 or 1.0)
                ax.set_xticks(x)
                ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
                ax.legend(fontsize=10, loc='upper left')
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "summarization_semantic_performance.png"))
                plt.close()
                
                # Second plot: Lexical metrics (ROUGE)
                plot_data = {metric: [] for metric in lexical_metrics}
                for model in model_names:
                    for metric in lexical_metrics:
                        plot_data[metric].append(results["models"][model]["summarization"][metric])
                
                fig, ax = plt.subplots(figsize=(12, 7))
                x = np.arange(len(model_names))
                width = 0.8 / len(lexical_metrics)
                
                for i, metric in enumerate(lexical_metrics):
                    offset = (i - len(lexical_metrics)/2 + 0.5) * width
                    ax.bar(x + offset, plot_data[metric], width, label=metric.upper())
                
                ax.set_title("Summarization ROUGE Scores", fontsize=16)
                ax.set_ylabel("Score", fontsize=14)
                max_value = max([max(plot_data[m]) for m in lexical_metrics]) if plot_data else 0
                ax.set_ylim(0, max_value * 1.2 or 1.0)
                ax.set_xticks(x)
                ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
                ax.legend(fontsize=12)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "summarization_rouge_performance.png"))
                plt.close()
            
            # Also generate plots for source-based evaluation if available
            if any("summarization_source" in results["models"][model] for model in model_names):
                print("  Generating source-based summarization plots...")
                
                # Get all available metrics for source-based evaluation
                source_metrics = list(next(
                    (results["models"][model]["summarization_source"] 
                     for model in model_names 
                     if "summarization_source" in results["models"][model]),
                    {}
                ).keys())
                
                if source_metrics:
                    plot_data = {metric: [] for metric in source_metrics}
                    for model in model_names:
                        if "summarization_source" in results["models"][model]:
                            for metric in source_metrics:
                                plot_data[metric].append(
                                    results["models"][model]["summarization_source"].get(metric, 0)
                                )
                        else:
                            for metric in source_metrics:
                                plot_data[metric].append(0)
                    
                    fig, ax = plt.subplots(figsize=(12, 7))
                    x = np.arange(len(model_names))
                    width = 0.8 / len(source_metrics)
                    
                    for i, metric in enumerate(source_metrics):
                        offset = (i - len(source_metrics)/2 + 0.5) * width
                        ax.bar(x + offset, plot_data[metric], width, 
                               label=metric.upper().replace("_", " "))
                    
                    ax.set_title("Summarization vs Source Text Performance", fontsize=16)
                    ax.set_ylabel("Score", fontsize=14)
                    max_value = max([max(plot_data[m]) for m in source_metrics]) if any(plot_data.values()) else 0
                    ax.set_ylim(0, max_value * 1.2 or 1.0)
                    ax.set_xticks(x)
                    ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
                    ax.legend(fontsize=10, loc='upper left')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(plots_dir, "summarization_source_performance.png"))
                    plt.close()
            
            # 3. QA Performance
            if "qa" in results["models"][model_names[0]]:
                # Get all available metrics for QA
                available_metrics = list(results["models"][model_names[0]]["qa"].keys())
                
                plot_data = {metric: [] for metric in available_metrics}
                for model in model_names:
                    for metric in available_metrics:
                        plot_data[metric].append(results["models"][model]["qa"][metric])
                
                fig, ax = plt.subplots(figsize=(12, 7))
                x = np.arange(len(model_names))
                width = 0.8 / len(available_metrics)
                
                for i, metric in enumerate(available_metrics):
                    offset = (i - len(available_metrics)/2 + 0.5) * width
                    ax.bar(x + offset, plot_data[metric], width, 
                          label=metric.upper().replace("_", " "))
                
                ax.set_title("Question Answering Performance", fontsize=16)
                ax.set_ylabel("Score", fontsize=14)
                max_value = max([max(plot_data[m]) for m in available_metrics]) if plot_data else 0
                ax.set_ylim(0, max_value * 1.2 or 1.0)
                ax.set_xticks(x)
                ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=12)
                ax.legend(fontsize=10)
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "qa_performance.png"))
                plt.close()
            
            # 4. Overall Comparison: Create heatmap of all metrics
            all_metrics = {}
            task_metrics = {
                "Translation": ["bleu", "semantic_sim"],
                "Summarization": ["semantic_sim", "info_coverage", "bertscore_f1", "rouge1"],
                "Summarization_Source": ["source_coverage", "semantic_relevance", "conciseness", "bertscore_f1"],
                "QA": ["semantic_sim", "containment", "info_coverage"]
            }
            
            # Collect metrics for heatmap
            for task, metrics in task_metrics.items():
                task_key = task.lower()
                for metric in metrics:
                    metric_key = f"{task_key}_{metric}"
                    all_metrics[metric_key] = []
                    
                    for model in model_names:
                        if task_key in results["models"][model] and metric in results["models"][model][task_key]:
                            all_metrics[metric_key].append(results["models"][model][task_key][metric])
                        else:
                            all_metrics[metric_key].append(0)
            
            # Create DataFrame for heatmap
            heatmap_data = pd.DataFrame(all_metrics, index=model_names)
            
            # Create heatmap
            plt.figure(figsize=(14, 8))
            sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=.5)
            plt.title("Overall Model Performance Comparison", fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "overall_performance_heatmap.png"))
            plt.close()
            
            # Create comparison CSV
            comparison_data = []
            for model in model_names:
                model_data = {"Model": model}
                for task in ["translation", "summarization", "summarization_source", "qa"]:
                    if task in results["models"][model]:
                        for metric, value in results["models"][model][task].items():
                            model_data[f"{task}_{metric}"] = value
                comparison_data.append(model_data)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_csv = os.path.join(output_dir, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            comparison_df.to_csv(comparison_csv, index=False)
            
            print(f"Saved comparison CSV to {comparison_csv}")
        except Exception as e:
            print(f"Error generating plots or CSV: {e}")
            import traceback
            traceback.print_exc()
        
        print("\nEvaluation complete!")
        return results
    else:
        print("\nNo models were evaluated.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced evaluation of model outputs with semantic metrics")
    parser.add_argument("--baseline", default="data.csv", help="Path to baseline CSV file (default: data.csv)")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to model CSV files to evaluate")
    parser.add_argument("--output", default="results", help="Directory to save results (default: results)")
    parser.add_argument("--sample_count", type=int, default=5, help="Number of sample comparisons to save (default: 5)")
    
    args = parser.parse_args()
    
    print(f"Evaluating models: {args.models}")
    print("Dependencies status:")
    print(f"- SentenceTransformer: {'Available' if SENTENCE_TRANSFORMERS_AVAILABLE else 'Not Available'}")
    print(f"- SpaCy: {'Available' if SPACY_AVAILABLE else 'Not Available'}")
    print(f"- BERTScore: {'Available' if BERT_SCORE_AVAILABLE else 'Not Available'}")
    
    # Installation instructions if dependencies are missing
    missing_deps = []
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        missing_deps.append("sentence-transformers")
    if not SPACY_AVAILABLE:
        missing_deps.append("spacy")
    if not BERT_SCORE_AVAILABLE:
        missing_deps.append("bert_score")
    
    if missing_deps:
        print("\nMissing dependencies detected. Install with:")
        print(f"pip install {' '.join(missing_deps)}")
        print("\nContinuing with available metrics only...")
    
    evaluate_models(args.baseline, args.models, args.output)