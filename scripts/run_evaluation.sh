#!/bin/bash

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Please install Ollama first."
    echo "Visit https://ollama.com/ for installation instructions."
    exit 1
fi

# Get list of available models
echo "Available Ollama models:"
ollama list

# Run the models
echo "Running models..."
python ../src/model_runner.py --models "llama3.2:1b" "deepseek-r1" 

# Run evaluation
echo "Evaluating results..."
python ../src/evaluate_results.py

echo "Evaluation complete! Check the 'results' directory for comparison reports and visualizations."