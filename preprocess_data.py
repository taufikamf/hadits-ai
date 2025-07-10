#!/usr/bin/env python3
"""
Script to preprocess hadits dataset and save results.
"""
import json
import logging
from pathlib import Path

from data.data_loader import HaditsDataLoader

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main preprocessing function"""
    try:
        # Initialize data loader
        loader = HaditsDataLoader()
        
        # Load and process dataset
        logger.info("Loading and processing dataset...")
        documents = loader.load_default_dataset()
        
        # Get statistics
        stats = loader.get_dataset_stats(documents)
        logger.info(f"Dataset statistics: {stats}")
        
        # Save processed documents
        output_path = Path("data/processed_hadits.json")
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Processed {len(documents)} documents saved to {output_path}")
        
        # Show sample document
        if documents:
            logger.info("Sample document:")
            sample = documents[0]
            print(f"ID: {sample['id']}")
            print(f"Kitab: {sample['kitab']}")
            print(f"Arab (original): {sample['arab_asli'][:100]}...")
            print(f"Arab (clean): {sample['arab_bersih'][:100]}...")
            print(f"Terjemah (clean): {sample['terjemah_bersih'][:100]}...")
            print(f"Content for embedding: {sample['content_for_embedding'][:100]}...")
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()