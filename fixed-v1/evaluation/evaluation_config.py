#!/usr/bin/env python3
"""
Evaluation Configuration - Fixed V1
===================================

Configuration specifically for evaluation directory perspective paths.

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import sys
import os
from pathlib import Path

# Add service modules to path
sys.path.append(str(Path(__file__).parent.parent))

from service.hadith_ai_service import ServiceConfig
from generation.enhanced_response_generator import GenerationConfig, LLMProvider, ResponseMode
from retrieval.enhanced_retrieval_system import RetrievalConfig
from preprocessing.query_preprocessor import EnhancedQueryPreprocessor

def create_evaluation_config():
    """Create configuration optimized for evaluation directory."""
    
    # Evaluation-specific paths (from fixed-v1/evaluation perspective)
    data_paths = {
        "hadits_docs": "../../data/processed/hadits_docs.json",
        "keywords_map": "../../data/enhanced_index_v1/enhanced_keywords_map_v1.json",
        "embeddings": "../../data/enhanced_index_v1/enhanced_embeddings_v1.pkl",
        "faiss_index": "../../data/enhanced_index_v1/enhanced_faiss_index_v1.index",
        "metadata": "../../data/enhanced_index_v1/enhanced_metadata_v1.pkl"
    }
    
    # Retrieval configuration for evaluation
    retrieval_config = RetrievalConfig(
        embeddings_path=data_paths["embeddings"],
        keywords_map_path=data_paths["keywords_map"],
        top_k=10,  # Smaller for evaluation
        min_score_threshold=0.1,
        semantic_weight=0.7,
        keyword_weight=0.2,
        literal_overlap_weight=0.1,
        enable_keyword_filtering=True,
        enable_reranking=True,
        max_results=10
    )
    
    # Generation configuration for evaluation
    generation_config = GenerationConfig(
        llm_provider=LLMProvider.NONE,  # Disable LLM for consistent evaluation
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        response_mode=ResponseMode.SUMMARY,  # Shorter responses for evaluation
        max_tokens=500,  # Smaller for evaluation
        temperature=0.3,  # More deterministic for evaluation
        enable_streaming=False,
        max_hadits_display=3  # Fewer hadits for evaluation
    )
    
    # Service configuration for evaluation
    service_config = ServiceConfig(
        retrieval_config=retrieval_config,
        generation_config=generation_config,
        enable_llm_generation=True,  # Enable LLM for complete evaluation
        fallback_to_simple=True,  # Always have fallback
        max_results_display=3,  # Fewer results for evaluation
        enable_sessions=False,  # No sessions during evaluation
        enable_analytics=False,  # No analytics during evaluation
        log_queries=False,  # No logging during evaluation
        log_results=False
    )
    
    return service_config, data_paths

def create_preprocessor_for_evaluation():
    """Create query preprocessor with evaluation-specific config."""
    return EnhancedQueryPreprocessor(
        keywords_map_path="../../data/enhanced_index_v1/enhanced_keywords_map_v1.json"
    )

def validate_evaluation_setup():
    """Validate that all required files exist for evaluation."""
    
    # Get config
    service_config, data_paths = create_evaluation_config()
    
    missing_files = []
    existing_files = []
    
    for name, path in data_paths.items():
        abs_path = Path(__file__).parent / path
        if abs_path.exists():
            size = abs_path.stat().st_size
            existing_files.append(f"✅ {name}: {path} ({size:,} bytes)")
        else:
            missing_files.append(f"❌ {name}: {path}")
    
    print("📁 Evaluation Setup Validation")
    print("=" * 50)
    
    if existing_files:
        print("✅ Available Files:")
        for file_info in existing_files:
            print(f"   {file_info}")
    
    if missing_files:
        print("\n❌ Missing Files:")
        for file_info in missing_files:
            print(f"   {file_info}")
        return False
    
    print(f"\n🎉 All {len(data_paths)} required files available!")
    return True

if __name__ == "__main__":
    validate_evaluation_setup()
