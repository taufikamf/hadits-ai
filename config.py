"""
Configuration settings for hadits-ai application
"""
import os
from typing import Optional, Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Google Gemini API Configuration
    gemini_api_key: str
    
    # Embedding Model Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_api_key: Optional[str] = None
    
    # Vector Database Configuration
    vector_db: Literal["chroma", "faiss"] = "chroma"
    
    # ChromaDB Configuration
    chroma_persist_directory: str = "./chroma_data"
    
    # FAISS Configuration
    faiss_index_path: str = "./faiss_data/hadits.index"
    faiss_metadata_path: str = "./faiss_data/metadata.json"
    
    # Data Configuration
    dataset_path: str = "./data/hadits.csv"
    max_retrieval_results: int = 5
    score_threshold: float = 0.3
    
    # Embedding Cache Configuration
    enable_embedding_cache: bool = True
    embedding_cache_dir: str = "./embedding_cache"
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: str = "logs/hadits-ai.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_vector_store_config(self) -> dict:
        """Get vector store configuration based on selected backend"""
        if self.vector_db == "chroma":
            return {
                "persist_directory": self.chroma_persist_directory,
                "collection_name": "hadits_collection"
            }
        elif self.vector_db == "faiss":
            # Create directory if not exists
            os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
            return {
                "index_path": self.faiss_index_path,
                "metadata_path": self.faiss_metadata_path
            }
        else:
            raise ValueError(f"Unsupported vector database: {self.vector_db}")
    
    def get_embedding_config(self) -> dict:
        """Get embedding configuration"""
        config = {
            "model_name": self.embedding_model,
            "cache_enabled": self.enable_embedding_cache,
            "cache_dir": self.embedding_cache_dir
        }
        
        if self.embedding_model == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            config["api_key"] = self.openai_api_key
            
        return config


# Global settings instance
settings = Settings() 