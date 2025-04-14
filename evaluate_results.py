import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import argparse
import json
from datetime import datetime

# Download NLTK data
nltk.download('punkt', quiet=True)

class ModelEvaluator:
    def __init__(self, baseline_file="data.csv", results_dir="results"):
        """
        Initialize model evaluator
        
        Args:
            baseline_file (str): Path to baseline file with ground truth
            results_dir (str): Directory to save evaluation results
        """
        self.baseline_file = baseline_file
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Load baseline data
        self.baseline_df = pd.read_csv(baseline_file)
        
        # Initialize rouge scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Initialize BLEU smoothing function
        self.smoothing = SmoothingFunction().method1
        
        # Find all model result files
        self.model_files = self._find_model_files()
        
        # Results dictionary
        self.results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": {}
        }
    
    def _find_model_files(self):
        """
        Find all model result files (data_*.csv)
        """
        return glob.glob("data_*.csv")
    
    def _clean_text(self, text):
        """
        Clean text for consistent evaluation
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def evaluate_translation(self, model_name, model_df):
        """
        Evaluate translation quality using BLEU score
        """
        print(f"Evaluating translation for {model_name}...")
        
        # Get column names
        translate_col = [col for col in model_df.columns if col.startswith(f"translate_{model_name}")][0] if any(col.startswith(f"translate_{model_name}") for col in model_df.columns) else None
        
        if not translate_col:
            print(f"No translation column found for {model_name}")
            return 0
        
        # Get references and candidates
        references = [nltk.word_tokenize(self._clean_text(text)) for text in self.baseline_df["text_en"]]
        candidates = [nltk.word_tokenize(self._clean_text(text)) for text in model_df[translate_col]]
        
        # Calculate BLEU scores
        bleu_scores = []
        for ref, cand in zip(references, candidates):
            if not ref or not cand:
                bleu_scores.append(0)
                continue
                
            try:
                score = sentence_bleu([ref], cand, smoothing_function=self.smoothing)
                bleu_scores.append(score)
            except Exception as e:
                print(f"Error calculating BLEU: {e}")
                bleu_scores.append(0)
        
        # Calculate average BLEU score
        avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
        
        print(f"Translation BLEU score: {avg_bleu:.4f}")
        return avg_bleu
    
    def evaluate_summarization(self, model_name, model_df):
        """
        Evaluate summarization quality using ROUGE scores
        """
        print(f"Evaluating summarization for {model_name}...")
        
        # Get column names
        summary_col = [col for col in model_df.columns if col.startswith(f"summary_{model_name}")][0] if any(col.startswith(f"summary_{model_name}") for col in model_df.columns) else None
        
        if not summary_col:
            print(f"No summary column found for {model_name}")
            return {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        
        # Get references and candidates
        references = [self._clean_text(text) for text in self.baseline_df["summary_en"]]
        candidates = [self._clean_text(text) for text in model_df[summary_col]]
        
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
                scores = self.rouge_scorer.score(ref, cand)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except Exception as e:
                print(f"Error calculating ROUGE: {e}")
                rouge1_scores.append(0)
                rouge2_scores.append(0)
                rougeL_scores.append(0)
        
        # Calculate average ROUGE scores
        avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0
        avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0
        avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0
        
        print(f"Summarization ROUGE scores:")
        print(f"  ROUGE-1: {avg_rouge1:.4f}")
        print(f"  ROUGE-2: {avg_rouge2:.4f}")
        print(f"  ROUGE-L: {avg_rougeL:.4f}")
        
        return {
            "rouge1": avg_rouge1,
            "rouge2": avg_rouge2,
            "rougeL": avg_rougeL
        }
    
    def evaluate_qa(self, model_name, model_df):
        """
        Evaluate question answering quality using F1 and exact match
        """
        print(f"Evaluating QA for {model_name}...")
        
        # Get column names
        answer_col = [col for col in model_df.columns if col.startswith(f"answer_{model_name}")][0] if any(col.startswith(f"answer_{model_name}") for col in model_df.columns) else None
        
        if not answer_col:
            print(f"No answer column found for {model_name}")
            return {"exact_match": 0, "f1": 0}
        
        # Get references and candidates
        references = [self._clean_text(text) for text in self.baseline_df["answer"]]
        candidates = [self._clean_text(text) for text in model_df[answer_col]]
        
        # Calculate exact match
        exact_matches = [1 if ref == cand else 0 for ref, cand in zip(references, candidates)]
        exact_match_score = np.mean(exact_matches) if exact_matches else 0
        
        # Calculate token-level F1 score
        f1_scores = []
        
        for ref, cand in zip(references, candidates):
            if not ref or not cand:
                f1_scores.append(0)
                continue
                
            # Tokenize
            ref_tokens = set(nltk.word_tokenize(ref))
            cand_tokens = set(nltk.word_tokenize(cand))
            
            # Calculate F1
            if not ref_tokens and not cand_tokens:
                f1_scores.append(1.0)
                continue
                
            if not ref_tokens or not cand_tokens:
                f1_scores.append(0.0)
                continue
                
            common_tokens = ref_tokens.intersection(cand_tokens)
            
            precision = len(common_tokens) / len(cand_tokens) if cand_tokens else 0
            recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        # Calculate average F1
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        
        print(f"QA scores:")
        print(f"  Exact Match: {exact_match_score:.4f}")
        print(f"  F1 Score: {avg_f1:.4f}")
        
        return {
            "exact_match": exact_match_score,
            "f1": avg_f1
        }
    
    def evaluate_model(self, model_file):
        """
        Evaluate a specific model
        
        Args:
            model_file (str): Path to model results file
        """
        print(f"\nEvaluating model from file: {model_file}")
        
        # Extract model name from filename
        model_name = model_file.replace("data_", "").replace(".csv", "")
        
        # Load model data
        model_df = pd.read_csv(model_file)
        
        # Evaluate each task
        translation_score = self.evaluate_translation(model_name, model_df)
        summarization_scores = self.evaluate_summarization(model_name, model_df)
        qa_scores = self.evaluate_qa(model_name, model_df)
        
        # Store results
        self.results["models"][model_name] = {
            "translation": {"bleu": translation_score},
            "summarization": summarization_scores,
            "qa": qa_scores
        }
        
        return model_name
    
    def evaluate_all_models(self):
        """
        Evaluate all model files
        """
        print(f"Found {len(self.model_files)} model files to evaluate")
        
        model_names = []
        
        for model_file in self.model_files:
            model_name = self.evaluate_model(model_file)
            model_names.append(model_name)
        
        # Save results
        results_file = os.path.join(self.results_dir, f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nEvaluation results saved to {results_file}")
        
        # Generate comparison plots
        self.generate_comparison_plots(model_names)
        
        return self.results
    
    def generate_comparison_plots(self, model_names):
        """
        Generate comparison plots
        
        Args:
            model_names (list): List of model names
        """
        plots_dir = os.path.join(self.results_dir, "plots")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        
        # 1. Translation BLEU scores
        plt.figure(figsize=(10, 6))
        bleu_scores = [self.results["models"][model]["translation"]["bleu"] for model in model_names]
        plt.bar(model_names, bleu_scores)
        plt.title("Translation Performance (BLEU Score)")
        plt.ylabel("BLEU Score")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "translation_bleu.png"))
        plt.close()
        
        # 2. Summarization ROUGE scores
        plt.figure(figsize=(12, 6))
        x = np.arange(len(model_names))
        width = 0.25
        
        rouge1_scores = [self.results["models"][model]["summarization"]["rouge1"] for model in model_names]
        rouge2_scores = [self.results["models"][model]["summarization"]["rouge2"] for model in model_names]
        rougeL_scores = [self.results["models"][model]["summarization"]["rougeL"] for model in model_names]
        
        plt.bar(x - width, rouge1_scores, width, label="ROUGE-1")
        plt.bar(x, rouge2_scores, width, label="ROUGE-2")
        plt.bar(x + width, rougeL_scores, width, label="ROUGE-L")
        
        plt.title("Summarization Performance")
        plt.ylabel("ROUGE Score")
        plt.ylim(0, 1.0)
        plt.xticks(x, model_names, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "summarization_rouge.png"))
        plt.close()
        
        # 3. QA scores
        plt.figure(figsize=(10, 6))
        x = np.arange(len(model_names))
        width = 0.35
        
        exact_match_scores = [self.results["models"][model]["qa"]["exact_match"] for model in model_names]
        f1_scores = [self.results["models"][model]["qa"]["f1"] for model in model_names]
        
        plt.bar(x - width/2, exact_match_scores, width, label="Exact Match")
        plt.bar(x + width/2, f1_scores, width, label="F1 Score")
        
        plt.title("Question Answering Performance")
        plt.ylabel("Score")
        plt.ylim(0, 1.0)
        plt.xticks(x, model_names, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "qa_scores.png"))
        plt.close()
        
        # 4. Overall performance radar chart
        for i, model in enumerate(model_names):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, polar=True)
            
            # Compute values (average for summarization)
            summarization_avg = np.mean([
                self.results["models"][model]["summarization"]["rouge1"],
                self.results["models"][model]["summarization"]["rouge2"],
                self.results["models"][model]["summarization"]["rougeL"]
            ])
            
            qa_avg = np.mean([
                self.results["models"][model]["qa"]["exact_match"],
                self.results["models"][model]["qa"]["f1"]
            ])
            
            values = [
                self.results["models"][model]["translation"]["bleu"],
                summarization_avg,
                qa_avg
            ]
            
            labels = ["Translation", "Summarization", "QA"]
            
            # Close the loop
            values.append(values[0])
            labels.append(labels[0])
            
            # Draw the chart
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
            angles.append(angles[0])
            
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
            ax.set_ylim(0, 1)
            plt.title(f"{model} Performance Across Tasks", size=15)
            
            plt.savefig(os.path.join(plots_dir, f"{model}_radar.png"))
            plt.close()
        
        print(f"Comparison plots saved to {plots_dir}")
        
        # 5. Create comparison CSV
        comparison_data = []
        
        for model in model_names:
            model_data = {
                "Model": model,
                "Translation_BLEU": self.results["models"][model]["translation"]["bleu"],
                "Summarization_ROUGE1": self.results["models"][model]["summarization"]["rouge1"],
                "Summarization_ROUGE2": self.results["models"][model]["summarization"]["rouge2"],
                "Summarization_ROUGEL": self.results["models"][model]["summarization"]["rougeL"],
                "QA_ExactMatch": self.results["models"][model]["qa"]["exact_match"],
                "QA_F1": self.results["models"][model]["qa"]["f1"]
            }
            comparison_data.append(model_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_csv = os.path.join(self.results_dir, f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        comparison_df.to_csv(comparison_csv, index=False)
        
        print(f"Comparison CSV saved to {comparison_csv}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model performances")
    parser.add_argument("--baseline", default="data.csv", help="Baseline data file (default: data.csv)")
    parser.add_argument("--results-dir", default="results", help="Directory for evaluation results (default: results)")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(baseline_file=args.baseline, results_dir=args.results_dir)
    evaluator.evaluate_all_models()

if __name__ == "__main__":
    main()