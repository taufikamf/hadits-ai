"""
Simple embedding service using ChromaDB's default embedding.
This avoids dependency issues with sentence-transformers.
"""
import logging
import hashlib
import pickle
import os
from typing import List, Optional
import numpy as np

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class SimpleEmbeddingService:
    """
    Simple embedding service using ChromaDB's default embedding.
    This is a fallback when sentence-transformers has dependency issues.
    """
    
    def __init__(self, persist_directory: str = "./chroma_data"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="hadits_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Initialized simple embedding service with ChromaDB")
    
    def embed_documents(self, texts: List[str], ids: List[str] = None) -> List[List[float]]:
        """
        Embed documents using ChromaDB's default embedding.
        Returns normalized embeddings.
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        try:
            # Add documents to collection (this will compute embeddings)
            self.collection.add(
                documents=texts,
                ids=ids
            )
            
            # Get embeddings
            results = self.collection.get(
                ids=ids,
                include=['embeddings']
            )
            
            embeddings = results['embeddings']
            
            # Normalize embeddings
            embeddings = np.array(embeddings)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query using ChromaDB's default embedding.
        Returns normalized embedding.
        """
        try:
            # Create temporary collection for query
            temp_collection = self.client.create_collection(
                name="temp_query",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Add query to collection
            temp_collection.add(
                documents=[text],
                ids=["query"]
            )
            
            # Get embedding
            results = temp_collection.get(
                ids=["query"],
                include=['embeddings']
            )
            
            embedding = np.array(results['embeddings'][0])
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            # Clean up temporary collection
            self.client.delete_collection("temp_query")
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        # ChromaDB's default embedding dimension
        return 384
    
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """
        Search for similar documents.
        Returns list of dicts with 'id', 'document', 'distance', 'metadata'.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'distance': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search: {e}")
            raise


def get_simple_embedding_service() -> SimpleEmbeddingService:
    """Get global simple embedding service instance"""
    return SimpleEmbeddingService()