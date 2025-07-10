#!/usr/bin/env python3
"""
Script to build embedding index using simple embedding service.
"""
import json
import logging
from pathlib import Path

from embedding.simple_embedding import SimpleEmbeddingService

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
        
        # Initialize simple embedding service
        embedding_service = SimpleEmbeddingService()
        
        # Extract texts and metadata for embedding
        texts = []
        ids = []
        metadata_list = []
        
        for doc in documents:
            texts.append(doc['content_for_embedding'])
            ids.append(str(doc['id']))
            metadata_list.append({
                'kitab': doc['kitab'],
                'arab_asli': doc['arab_asli'],
                'arab_bersih': doc['arab_bersih'],
                'terjemah': doc['terjemah_bersih']
            })
        
        # Build index by adding documents to ChromaDB
        logger.info("Building embedding index...")
        
        # Add documents to collection
        embedding_service.collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadata_list
        )
        
        logger.info("Index built successfully!")
        
        # Test retrieval
        logger.info("Testing retrieval...")
        test_query = "apa itu niat"
        results = embedding_service.search(test_query, top_k=3)
        
        logger.info(f"Test query: '{test_query}'")
        logger.info(f"Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            logger.info(f"Result {i}:")
            logger.info(f"  Distance: {result['distance']:.4f}")
            logger.info(f"  Kitab: {result['metadata']['kitab']}")
            logger.info(f"  Terjemah: {result['metadata']['terjemah'][:100]}...")
            logger.info("")
        
    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise

if __name__ == "__main__":
    main()