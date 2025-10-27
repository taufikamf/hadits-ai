"""
Enhanced Embedding System - Fixed V1
===================================

Optimized document embedding system combining:
- Enhanced semantic tagging with keyword integration
- Efficient batch processing with progress tracking
- Memory optimization for large datasets
- Comprehensive corpus preparation with Islamic context
- Adaptive semantic enhancement for better retrieval

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import os
import sys
import json
import pickle
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.preprocessing import normalize

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
    logger.info(f"PyTorch available with CUDA: {torch.cuda.is_available()}")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class EmbeddingConfig:
    """Configuration for enhanced embedding system."""
    model_name: str = "intfloat/e5-small-v2"
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    use_gpu: bool = True
    semantic_enhancement: bool = True
    keyword_boost_weight: float = 0.3
    
    # Paths
    data_path: str = "../data/processed/hadits_docs.json"
    keywords_path: str = "../data/enhanced_index_v1/enhanced_keywords_map_v1.json"
    output_path: str = "../data/enhanced_index_v1/enhanced_embeddings_v1.pkl"


class EnhancedEmbeddingSystem:
    """
    Advanced embedding system with semantic enhancement and keyword integration.
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        """
        Initialize the enhanced embedding system.
        
        Args:
            config (EmbeddingConfig): Configuration object
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        self.keywords_map = {}
        self.reverse_keywords_map = {}
        
        # Load components
        self._initialize_model()
        self._load_keywords_map()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            
            # Set device
            device = "cuda" if (self.config.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load model
            self.model = SentenceTransformer(self.config.model_name, device=device)
            
            # Set model to evaluation mode for consistency
            if hasattr(self.model, 'eval'):
                self.model.eval()
                
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _load_keywords_map(self):
        """Load enhanced keywords map for semantic tagging."""
        try:
            if not os.path.exists(self.config.keywords_path):
                logger.warning(f"Keywords map not found at {self.config.keywords_path}")
                return
            
            with open(self.config.keywords_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.keywords_map = data.get('keywords', {})
            
            # Build reverse map for fast lookup
            self.reverse_keywords_map = {}
            for canonical_term, variants in self.keywords_map.items():
                for variant in variants:
                    self.reverse_keywords_map[variant.lower()] = canonical_term
                # Add canonical term itself
                self.reverse_keywords_map[canonical_term.lower()] = canonical_term
            
            logger.info(f"Loaded {len(self.keywords_map)} keyword groups with {len(self.reverse_keywords_map)} total variants")
            
        except Exception as e:
            logger.warning(f"Could not load keywords map: {e}")
            self.keywords_map = {}
            self.reverse_keywords_map = {}
    
    def build_enhanced_semantic_tags(self, document: Dict) -> str:
        """
        Build enhanced semantic tags using keyword matching and context analysis.
        
        Args:
            document (Dict): Document with 'terjemah' field
            
        Returns:
            str: Comma-separated semantic tags
        """
        if not self.keywords_map:
            return ""
        
        text = document.get("terjemah", "").lower()
        if not text:
            return ""
        
        found_tags = set()
        
        # Method 1: Direct keyword matching with word boundaries
        for canonical_term, variants in self.keywords_map.items():
            for variant in variants:
                # Use word boundaries for precise matching
                pattern = rf"(?<!\w){re.escape(variant.lower())}(?!\w)"
                if re.search(pattern, text):
                    found_tags.add(canonical_term)
                    break  # Found match for this canonical term
        
        # Method 2: Enhanced contextual matching for compound terms
        compound_patterns = {
            'shalat_jumat': [r'shalat.*jumat', r'jumat.*shalat'],
            'puasa_ramadan': [r'puasa.*ramadan', r'ramadan.*puasa'],
            'zakat_fitrah': [r'zakat.*fitrah', r'fitrah.*zakat'],
            'minuman_keras': [r'minuman.*keras', r'khamr', r'arak'],
            'orang_tua': [r'orang.*tua', r'ibu.*bapak', r'ayah.*ibu'],
            'anak_yatim': [r'anak.*yatim', r'yatim.*piatu']
        }
        
        for concept, patterns in compound_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    found_tags.add(concept.replace('_', ' '))
                    break
        
        # Method 3: Islamic context boosting
        islamic_context_indicators = {
            'hukum': ['hukum', 'boleh', 'haram', 'halal', 'wajib', 'sunnah', 'makruh'],
            'ibadah': ['ibadah', 'pahala', 'ganjaran', 'amal'],
            'akhlak': ['akhlak', 'adab', 'perilaku', 'sikap'],
            'akhirat': ['akhirat', 'surga', 'neraka', 'pahala', 'dosa']
        }
        
        for context, indicators in islamic_context_indicators.items():
            if any(indicator in text for indicator in indicators):
                # Add more specific tags if context matches
                for tag in found_tags.copy():
                    if any(indicator in tag for indicator in indicators):
                        found_tags.add(f"{context}_{tag}")
        
        # Return sorted tags for consistency
        return ", ".join(sorted(found_tags))
    
    def prepare_enhanced_corpus(self, documents: List[Dict]) -> List[str]:
        """
        Prepare enhanced corpus with semantic tagging and context enhancement.
        
        Args:
            documents (List[Dict]): List of hadith documents
            
        Returns:
            List[str]: Enhanced corpus ready for embedding
        """
        logger.info(f"Preparing enhanced corpus for {len(documents)} documents...")
        
        enhanced_corpus = []
        
        # Show progress for first few documents
        sample_size = min(10, len(documents))
        logger.info("Sample semantic tag results:")
        
        for i, doc in enumerate(documents):
            if i < sample_size:
                tags = self.build_enhanced_semantic_tags(doc)
                logger.info(f"Doc {i+1} tags: {tags[:100]}{'...' if len(tags) > 100 else ''}")
        
        # Process all documents
        for i, doc in enumerate(tqdm(documents, desc="Building corpus")):
            if "id" not in doc:
                logger.warning(f"Document {i} missing ID field")
                continue
                
            base_text = doc.get("terjemah", "")
            if not base_text:
                logger.warning(f"Document {i} missing terjemah field")
                continue
            
            # Build semantic tags
            tags = self.build_enhanced_semantic_tags(doc)
            
            # Create enhanced corpus entry
            if tags and self.config.semantic_enhancement:
                # Enhanced format with semantic tags
                enhanced_text = f"passage: {base_text}. Kata kunci penting: {tags}"
            else:
                # Fallback format
                enhanced_text = f"passage: {base_text}"
            
            enhanced_corpus.append(enhanced_text)
            
            # Log documents with few tags for quality monitoring
            if len(tags.split(", ")) <= 1 and tags:
                logger.debug(f"Doc {doc.get('id', i)} has only 1 tag: {tags}")
        
        logger.info(f"Enhanced corpus prepared with {len(enhanced_corpus)} documents")
        return enhanced_corpus
    
    def embed_documents_batch(self, corpus: List[str]) -> np.ndarray:
        """
        Embed documents in batches with progress tracking and memory optimization.
        
        Args:
            corpus (List[str]): Corpus to embed
            
        Returns:
            np.ndarray: Document embeddings
        """
        if not self.model:
            raise ValueError("Model not initialized")
        
        logger.info(f"Embedding {len(corpus)} documents with batch size {self.config.batch_size}")
        
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(corpus), self.config.batch_size), desc="Embedding batches"):
            batch = corpus[i:i + self.config.batch_size]
            
            try:
                # Encode batch
                batch_embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings,
                    batch_size=len(batch),
                    show_progress_bar=False
                )
                
                # Additional normalization if requested
                if self.config.normalize_embeddings:
                    batch_embeddings = normalize(batch_embeddings, axis=1)
                
                all_embeddings.append(batch_embeddings)
                
                # Memory cleanup for large batches
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error embedding batch {i//self.config.batch_size}: {e}")
                # Create zero embeddings for failed batch
                batch_size = len(batch)
                embedding_dim = 384  # Default for e5-small-v2
                zero_embeddings = np.zeros((batch_size, embedding_dim))
                all_embeddings.append(zero_embeddings)
        
        # Concatenate all embeddings
        final_embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Embedding completed. Shape: {final_embeddings.shape}")
        return final_embeddings
    
    def process_documents(self, documents: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Complete pipeline: prepare corpus and generate embeddings.
        
        Args:
            documents (List[Dict]): Input documents
            
        Returns:
            Tuple[np.ndarray, List[str]]: Embeddings and enhanced corpus
        """
        logger.info("Starting enhanced document processing pipeline...")
        
        # Step 1: Prepare enhanced corpus
        enhanced_corpus = self.prepare_enhanced_corpus(documents)
        
        # Step 2: Generate embeddings
        embeddings = self.embed_documents_batch(enhanced_corpus)
        
        # Step 3: Quality checks
        self._validate_embeddings(embeddings, enhanced_corpus)
        
        logger.info("Document processing pipeline completed successfully")
        return embeddings, enhanced_corpus
    
    def _validate_embeddings(self, embeddings: np.ndarray, corpus: List[str]):
        """Validate embedding quality and log statistics."""
        if len(embeddings) != len(corpus):
            logger.error(f"Embedding count ({len(embeddings)}) doesn't match corpus count ({len(corpus)})")
            return
        
        # Check for zero embeddings (failed embeddings)
        zero_count = np.sum(np.all(embeddings == 0, axis=1))
        if zero_count > 0:
            logger.warning(f"Found {zero_count} zero embeddings (failed processing)")
        
        # Check embedding statistics
        mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
        std_norm = np.std(np.linalg.norm(embeddings, axis=1))
        
        logger.info(f"Embedding statistics:")
        logger.info(f"  - Shape: {embeddings.shape}")
        logger.info(f"  - Mean norm: {mean_norm:.4f}")
        logger.info(f"  - Std norm: {std_norm:.4f}")
        logger.info(f"  - Zero embeddings: {zero_count}")
    
    def save_embeddings(self, embeddings: np.ndarray, documents: List[Dict], 
                       enhanced_corpus: List[str] = None):
        """
        Save embeddings and metadata to file.
        
        Args:
            embeddings (np.ndarray): Document embeddings
            documents (List[Dict]): Original documents
            enhanced_corpus (List[str], optional): Enhanced corpus texts
        """
        logger.info(f"Saving embeddings to {self.config.output_path}")
        
        # Prepare output data
        output_data = {
            "embeddings": embeddings,
            "documents": documents,
            "metadata": {
                "model_name": self.config.model_name,
                "embedding_dimension": embeddings.shape[1],
                "document_count": len(documents),
                "semantic_enhancement": self.config.semantic_enhancement,
                "normalization": self.config.normalize_embeddings,
                "keywords_map_size": len(self.keywords_map),
                "processing_config": {
                    "batch_size": self.config.batch_size,
                    "max_length": self.config.max_length,
                    "keyword_boost_weight": self.config.keyword_boost_weight
                }
            }
        }
        
        # Add enhanced corpus if available
        if enhanced_corpus:
            output_data["enhanced_corpus"] = enhanced_corpus
        
        # Ensure output directory exists
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with pickle
        try:
            with open(output_path, "wb") as f:
                pickle.dump(output_data, f)
            logger.info(f"Embeddings saved successfully to {output_path}")
            
            # Log file size
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"Output file size: {file_size:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise
    
    def load_documents(self, path: str = None) -> List[Dict]:
        """Load documents from JSON file."""
        data_path = path or self.config.data_path
        
        logger.info(f"Loading documents from {data_path}")
        
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                documents = json.load(f)
            
            logger.info(f"Loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise


def run_enhanced_embedding_pipeline(config: EmbeddingConfig = None):
    """
    Run the complete enhanced embedding pipeline.
    
    Args:
        config (EmbeddingConfig, optional): Configuration object
    """
    config = config or EmbeddingConfig()
    
    logger.info("🚀 Starting Enhanced Embedding Pipeline - Fixed V1")
    logger.info("=" * 60)
    
    try:
        # Initialize system
        embedding_system = EnhancedEmbeddingSystem(config)
        
        # Load documents
        documents = embedding_system.load_documents()
        
        if not documents:
            logger.error("No documents found. Please check the data file.")
            return False
        
        # Process documents
        embeddings, enhanced_corpus = embedding_system.process_documents(documents)
        
        # Save results
        embedding_system.save_embeddings(embeddings, documents, enhanced_corpus)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 ENHANCED EMBEDDING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Documents processed: {len(documents):,}")
        logger.info(f"Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"Total embeddings: {embeddings.shape[0]:,}")
        logger.info(f"Model used: {config.model_name}")
        logger.info(f"Semantic enhancement: {config.semantic_enhancement}")
        logger.info(f"Output file: {config.output_path}")
        
        # Show sample enhanced corpus
        logger.info("\n🔍 Sample enhanced corpus entries:")
        for i, text in enumerate(enhanced_corpus[:3]):
            logger.info(f"  [{i+1}] {text[:150]}{'...' if len(text) > 150 else ''}")
        
        logger.info(f"\n✅ Enhanced embedding pipeline completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in embedding pipeline: {e}")
        return False


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Embedding System - Fixed V1")
    parser.add_argument("--model", default="intfloat/e5-small-v2", help="Model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic enhancement")
    parser.add_argument("--data-path", help="Path to documents JSON file")
    parser.add_argument("--keywords-path", help="Path to keywords map")
    parser.add_argument("--output-path", help="Output path for embeddings")
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = EmbeddingConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu,
        semantic_enhancement=not args.no_semantic
    )
    
    # Override paths if provided
    if args.data_path:
        config.data_path = args.data_path
    if args.keywords_path:
        config.keywords_path = args.keywords_path
    if args.output_path:
        config.output_path = args.output_path
    
    # Run pipeline
    success = run_enhanced_embedding_pipeline(config)
    sys.exit(0 if success else 1)