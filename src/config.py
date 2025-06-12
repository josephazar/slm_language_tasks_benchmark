"""Configuration file for managing paths and settings."""

import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
SAMPLES_DIR = RESULTS_DIR / "samples"
DOCS_DIR = PROJECT_ROOT / "docs"
PAPER_DIR = DOCS_DIR / "paper"

# Ensure directories exist
for dir_path in [DATA_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, EXPERIMENTS_DIR, 
                  RESULTS_DIR, PLOTS_DIR, SAMPLES_DIR, DOCS_DIR, PAPER_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Helper functions
def get_data_path(filename, data_type='raw'):
    """Get the full path for a data file."""
    if data_type == 'raw':
        return DATA_RAW_DIR / filename
    elif data_type == 'processed':
        return DATA_PROCESSED_DIR / filename
    else:
        return DATA_DIR / filename

def get_results_path(filename):
    """Get the full path for a results file."""
    return RESULTS_DIR / filename

def get_plots_path(filename):
    """Get the full path for a plot file."""
    return PLOTS_DIR / filename

def get_samples_path(filename):
    """Get the full path for a samples file."""
    return SAMPLES_DIR / filename

# Default file names
DEFAULT_DATA_FILE = "data.csv"
DEFAULT_JSONL_FILE = "data.jsonl"