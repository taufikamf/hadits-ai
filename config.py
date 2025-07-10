"""
Configuration settings for hadits-ai application
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Google Gemini API Configuration
    gemini_api_key: str
    
    # Embedding Model Configuration
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    openai_api_key: Optional[str] = None
    
    # Vector Database Configuration
    vector_db: str = "chroma"
    chroma_persist_directory: str = "./chroma_data"
    
    # Data Configuration
    dataset_path: str = "./data/hadits.csv"
    max_retrieval_results: int = 5
    score_threshold: float = 0.3
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings() 