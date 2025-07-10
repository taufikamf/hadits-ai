#!/usr/bin/env python3
"""
Script to build embedding index from processed hadits data.
"""
import json
import logging
from pathlib import Path

from embedding.embedding_service import EmbeddingService
from retriever.vector_store import VectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to build embedding index"""
    try:
        # Load processed documents
        processed_path = Path("data/processed_hadits.json")
        if not processed_path.exists():
            logger.error("Processed data not found. Run preprocess_data.py first.")
            return
        
        with open(processed_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        logger.info(f"Loaded {len(documents)} processed documents")
        
        # Initialize embedding service
        embedding_service = EmbeddingService()
        
        # Initialize vector store
        vector_store = VectorStore(embedding_service)
        
        # Build index
        logger.info("Building embedding index...")
        vector_store.build_index(documents)
        
        logger.info("Index built successfully!")
        
        # Test retrieval
        logger.info("Testing retrieval...")
        test_query = "apa itu niat"
        results = vector_store.search(test_query, top_k=3)
        
        logger.info(f"Test query: '{test_query}'")
        logger.info(f"Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            logger.info(f"Result {i}:")
            logger.info(f"  Score: {result['score']:.4f}")
            logger.info(f"  Kitab: {result['kitab']}")
            logger.info(f"  Terjemah: {result['terjemah'][:100]}...")
            logger.info("")
        
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise

if __name__ == "__main__":
    main()