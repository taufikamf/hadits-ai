#!/usr/bin/env python3
"""
Simple runner script for keyword extraction from hadits CSV files.
Usage: python utils/run_keyword_extraction.py [--min-freq 20] [--max-ngram 3]
"""

import argparse
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from advanced_keyword_extractor import main as extract_keywords

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and group keywords from hadits CSV files"
    )
    parser.add_argument(
        "--min-freq", 
        type=int, 
        default=20,
        help="Minimum frequency threshold for terms (default: 20)"
    )
    parser.add_argument(
        "--max-ngram", 
        type=int, 
        default=3,
        help="Maximum n-gram size (default: 3)"
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="data/csv",
        help="Directory containing CSV files (default: data/csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/keywords_map_grouped.json",
        help="Output file path (default: data/processed/keywords_map_grouped.json)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Update configuration in the extractor module
    import advanced_keyword_extractor as extractor
    extractor.MIN_FREQUENCY = args.min_freq
    extractor.MAX_NGRAM = args.max_ngram
    extractor.CSV_DIR = Path(args.csv_dir)
    extractor.OUTPUT_PATH = Path(args.output)
    
    # Run extraction
    extract_keywords() 