"""
Embedding service for hadits-ai.
Inspired by Dify's CacheEmbedding with support for local and API models.
"""
import logging
import hashlib
import pickle
import os
from typing import List, Optional, Union
from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer
import openai

from config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """Local embedding using SentenceTransformers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._dimension = None
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            # Get dimension by encoding a dummy text
            dummy_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self._dimension = dummy_embedding.shape[1]
        return self._dimension


class OpenAIEmbedding(BaseEmbeddingModel):
    """OpenAI embedding model"""
    
    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self._dimension = 1536  # Default for text-embedding-3-small
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_name
            )
            embeddings = [item.embedding for item in response.data]
            # Normalize embeddings
            embeddings = np.array(embeddings)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed documents with OpenAI: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model_name
            )
            embedding = np.array(response.data[0].embedding)
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query with OpenAI: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self._dimension


class CachedEmbeddingService:
    """
    Embedding service with local caching.
    Inspired by Dify's CacheEmbedding pattern.
    """
    
    def __init__(self, cache_dir: str = "./embedding_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize embedding model based on config
        if settings.embedding_model == "openai":
            if not settings.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embedding model")
            self.model = OpenAIEmbedding(settings.openai_api_key)
        else:
            # Use local SentenceTransformer model
            self.model = SentenceTransformerEmbedding(settings.embedding_model)
        
        logger.info(f"Initialized embedding service with model: {settings.embedding_model}")
    
    def _get_cache_path(self, text_hash: str) -> str:
        """Get cache file path for a text hash"""
        return os.path.join(self.cache_dir, f"{text_hash}.pkl")
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_from_cache(self, text_hash: str) -> Optional[List[float]]:
        """Load embedding from cache"""
        cache_path = self._get_cache_path(text_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {text_hash}: {e}")
        return None
    
    def _save_to_cache(self, text_hash: str, embedding: List[float]) -> None:
        """Save embedding to cache"""
        cache_path = self._get_cache_path(text_hash)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {text_hash}: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents with caching.
        Similar to Dify's embed_documents pattern.
        """
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = self._hash_text(text)
            cached_embedding = self._load_from_cache(text_hash)
            
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed uncached texts
        if texts_to_embed:
            logger.info(f"Embedding {len(texts_to_embed)} uncached documents")
            new_embeddings = self.model.embed_documents(texts_to_embed)
            
            # Store new embeddings and cache them
            for idx, embedding in zip(indices_to_embed, new_embeddings):
                embeddings[idx] = embedding
                text_hash = self._hash_text(texts[idx])
                self._save_to_cache(text_hash, embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query with caching.
        Similar to Dify's embed_query pattern.
        """
        text_hash = self._hash_text(text)
        
        # Check cache first
        cached_embedding = self._load_from_cache(text_hash)
        if cached_embedding is not None:
            return cached_embedding
        
        # Embed and cache
        embedding = self.model.embed_query(text)
        self._save_to_cache(text_hash, embedding)
        
        return embedding
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_dimension()


# Global embedding service instance
_embedding_service: Optional[CachedEmbeddingService] = None


def get_embedding_service() -> CachedEmbeddingService:
    """Get global embedding service instance"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = CachedEmbeddingService()
    return _embedding_service


# Example usage
if __name__ == "__main__":
    # Test the embedding service
    service = get_embedding_service()
    
    # Test documents
    docs = [
        "Sesungguhnya setiap perbuatan tergantung niatnya",
        "Orang mukmin itu adalah cermin bagi orang mukmin yang lain"
    ]
    
    print(f"Embedding dimension: {service.get_dimension()}")
    
    # Embed documents
    doc_embeddings = service.embed_documents(docs)
    print(f"Document embeddings shape: {len(doc_embeddings)} x {len(doc_embeddings[0])}")
    
    # Embed query
    query = "Apa itu niat dalam Islam?"
    query_embedding = service.embed_query(query)
    print(f"Query embedding shape: {len(query_embedding)}") 