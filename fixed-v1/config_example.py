"""
Configuration Example - Fixed V1
===============================

Example configuration file for Hadith AI Fixed V1 system.
Copy this file to config.py and customize the settings.

Features:
- Environment variable support
- Multiple LLM providers
- Flexible response modes
- Production-ready settings

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import os
from generation.enhanced_response_generator import GenerationConfig, LLMProvider, ResponseMode
from service.hadith_ai_service import ServiceConfig
from retrieval.enhanced_retrieval_system import RetrievalConfig

# =============================================================================
# API Keys and Credentials
# =============================================================================

# Gemini AI Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# OpenAI Configuration (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# =============================================================================
# Generation Configuration
# =============================================================================

def create_generation_config(provider: str = "auto") -> GenerationConfig:
    """
    Create generation configuration based on available providers.
    
    Args:
        provider (str): LLM provider ("gemini", "openai", "none", "auto")
        
    Returns:
        GenerationConfig: Configured generation settings
    """
    
    # Auto-detect provider based on available API keys
    if provider == "auto":
        if GEMINI_API_KEY:
            provider = "gemini"
        elif OPENAI_API_KEY:
            provider = "openai"
        else:
            provider = "none"
    
    # Provider-specific configurations
    if provider == "gemini":
        return GenerationConfig(
            llm_provider=LLMProvider.GEMINI,
            gemini_api_key=GEMINI_API_KEY,
            gemini_model=GEMINI_MODEL,
            response_mode=ResponseMode.COMPREHENSIVE,
            temperature=0.3,
            max_tokens=2500,
            top_p=0.9,
            top_k=40,
            enable_streaming=True,
            max_hadits_display=5,
            max_arabic_length=300,
            max_translation_length=800,
            max_context_length=8000
        )
    
    elif provider == "openai":
        return GenerationConfig(
            llm_provider=LLMProvider.OPENAI,
            openai_api_key=OPENAI_API_KEY,
            openai_model=OPENAI_MODEL,
            response_mode=ResponseMode.COMPREHENSIVE,
            temperature=0.3,
            max_tokens=2500,
            top_p=0.9,
            enable_streaming=True,
            max_hadits_display=5,
            max_arabic_length=300,
            max_translation_length=800,
            max_context_length=8000
        )
    
    else:  # No LLM provider
        return GenerationConfig(
            llm_provider=LLMProvider.NONE,
            response_mode=ResponseMode.COMPREHENSIVE,
            enable_streaming=False,
            max_hadits_display=5,
            max_arabic_length=300,
            max_translation_length=800
        )

# =============================================================================
# Retrieval Configuration
# =============================================================================

def create_retrieval_config() -> RetrievalConfig:
    """Create retrieval system configuration."""
    return RetrievalConfig(
        # Data paths (relative to fixed-v1 directory)
        embeddings_path="../data/enhanced_index_v1/enhanced_embeddings_v1.pkl",
        keywords_map_path="../data/enhanced_index_v1/enhanced_keywords_map_v1.json",
        faiss_index_path="../data/enhanced_index_v1/enhanced_faiss_index_v1.index",
        metadata_path="../data/enhanced_index_v1/enhanced_metadata_v1.pkl",
        
        # Model settings
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        
        # Retrieval parameters
        top_k=10,
        similarity_threshold=0.1,
        
        # Scoring weights
        semantic_weight=0.7,
        keyword_weight=0.2,
        literal_weight=0.1,
        
        # Filtering settings
        enable_adaptive_filtering=True,
        adaptive_threshold_factor=0.8,
        min_results_threshold=3,
        max_results_threshold=20,
        
        # Performance settings
        enable_gpu=True,
        batch_size=32,
        max_sequence_length=512
    )

# =============================================================================
# Service Configuration
# =============================================================================

def create_service_config(
    environment: str = "development",
    llm_provider: str = "auto"
) -> ServiceConfig:
    """
    Create service configuration for different environments.
    
    Args:
        environment (str): Environment ("development", "production", "testing")
        llm_provider (str): LLM provider to use
        
    Returns:
        ServiceConfig: Configured service settings
    """
    
    # Base configuration
    base_config = {
        "retrieval_config": create_retrieval_config(),
        "generation_config": create_generation_config(llm_provider),
        "enable_llm_generation": True,
        "fallback_to_simple": True,
        "max_results_display": 5,
        "include_scores": False,
        "enable_sessions": True,
        "enable_analytics": True,
        "min_confidence_threshold": 0.1,
        "high_confidence_threshold": 0.6
    }
    
    # Environment-specific settings
    if environment == "development":
        return ServiceConfig(
            **base_config,
            session_timeout_minutes=30,
            max_context_history=10,
            log_queries=True,
            log_results=True,
            analytics_output="logs/hadith_ai_analytics_dev.jsonl"
        )
    
    elif environment == "production":
        return ServiceConfig(
            **base_config,
            session_timeout_minutes=60,
            max_context_history=20,
            log_queries=True,
            log_results=False,  # Reduce logging in production
            analytics_output="logs/hadith_ai_analytics_prod.jsonl",
            result_text_preview_length=300  # Longer previews in production
        )
    
    elif environment == "testing":
        return ServiceConfig(
            **base_config,
            enable_llm_generation=False,  # Disable LLM for faster testing
            enable_sessions=False,
            enable_analytics=False,
            max_results_display=3,
            session_timeout_minutes=5
        )
    
    else:
        raise ValueError(f"Unknown environment: {environment}")

# =============================================================================
# Presets for Common Use Cases
# =============================================================================

# Fast response (no LLM)
FAST_CONFIG = create_service_config("development", "none")

# Comprehensive response (with LLM)
COMPREHENSIVE_CONFIG = create_service_config("development", "auto")

# Production configuration
PRODUCTION_CONFIG = create_service_config("production", "auto")

# Testing configuration
TESTING_CONFIG = create_service_config("testing", "none")

# =============================================================================
# Environment Detection
# =============================================================================

def get_environment_config() -> ServiceConfig:
    """
    Get configuration based on environment variables.
    
    Environment Variables:
        HADITH_AI_ENV: Environment name (development|production|testing)
        HADITH_AI_LLM: LLM provider (gemini|openai|none|auto)
        
    Returns:
        ServiceConfig: Environment-appropriate configuration
    """
    env = os.getenv("HADITH_AI_ENV", "development")
    llm = os.getenv("HADITH_AI_LLM", "auto")
    
    return create_service_config(env, llm)

# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(config: ServiceConfig) -> bool:
    """
    Validate configuration and check for potential issues.
    
    Args:
        config (ServiceConfig): Configuration to validate
        
    Returns:
        bool: True if configuration is valid
    """
    issues = []
    
    # Check LLM configuration
    if config.enable_llm_generation:
        gen_config = config.generation_config
        
        if gen_config.llm_provider == LLMProvider.GEMINI and not gen_config.gemini_api_key:
            issues.append("Gemini API key is required for Gemini provider")
        
        if gen_config.llm_provider == LLMProvider.OPENAI and not gen_config.openai_api_key:
            issues.append("OpenAI API key is required for OpenAI provider")
    
    # Check file paths
    import pathlib
    base_path = pathlib.Path(__file__).parent
    
    required_files = [
        config.retrieval_config.embeddings_path,
        config.retrieval_config.keywords_map_path,
        config.retrieval_config.faiss_index_path,
        config.retrieval_config.metadata_path
    ]
    
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            issues.append(f"Required file not found: {file_path}")
    
    # Print issues
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Configuration is valid")
    return True

# =============================================================================
# Usage Examples
# =============================================================================

if __name__ == "__main__":
    print("Hadith AI Fixed V1 - Configuration Examples")
    print("=" * 50)
    
    # Show available configurations
    configs = {
        "Fast (No LLM)": FAST_CONFIG,
        "Comprehensive (With LLM)": COMPREHENSIVE_CONFIG,
        "Production": PRODUCTION_CONFIG,
        "Testing": TESTING_CONFIG
    }
    
    for name, config in configs.items():
        print(f"\n📋 {name} Configuration:")
        print(f"   - LLM Generation: {config.enable_llm_generation}")
        print(f"   - LLM Provider: {config.generation_config.llm_provider.value}")
        print(f"   - Response Mode: {config.generation_config.response_mode.value}")
        print(f"   - Max Results: {config.max_results_display}")
        print(f"   - Sessions: {config.enable_sessions}")
        print(f"   - Analytics: {config.enable_analytics}")
        
        # Validate configuration
        print(f"   - Valid: ", end="")
        validate_config(config)
    
    print(f"\n💡 To use these configurations:")
    print(f"   from config import COMPREHENSIVE_CONFIG")
    print(f"   from service.hadith_ai_service import HadithAIService")
    print(f"   service = HadithAIService(COMPREHENSIVE_CONFIG)")
