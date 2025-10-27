"""
Enhanced Indexing Pipeline - Fixed V1
====================================

Optimized indexing system that integrates all enhanced components:
- Enhanced keyword extraction with semantic grouping
- Advanced embedding generation with context awareness
- FAISS index building with optimized parameters
- Comprehensive metadata management
- Memory-efficient processing for large datasets

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import enhanced components
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from extraction.enhanced_keyword_extractor import EnhancedKeywordExtractor
from embedding.enhanced_embedding_system import EnhancedEmbeddingSystem, EmbeddingConfig
from preprocessing.query_preprocessor import EnhancedQueryPreprocessor

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS available for vector indexing")
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available - using fallback similarity search")


@dataclass
class IndexingConfig:
    """Configuration for enhanced indexing pipeline."""
    # Input data
    input_data_path: str = "../data/processed/hadits_docs.json"
    
    # Output paths
    output_dir: str = "../data/enhanced_index_v1"
    keywords_output: str = "enhanced_keywords_map_v1.json"
    embeddings_output: str = "enhanced_embeddings_v1.pkl"
    faiss_index_output: str = "enhanced_faiss_index_v1.index"
    metadata_output: str = "enhanced_metadata_v1.pkl"
    
    # Processing parameters
    keyword_min_frequency: int = 40
    keyword_max_ngram: int = 3
    embedding_batch_size: int = 32
    
    # FAISS index parameters
    faiss_index_type: str = "IVFFlat"  # Options: "Flat", "IVFFlat", "HNSW"
    faiss_nlist: int = 100  # For IVF indices
    faiss_nprobe: int = 10  # For IVF search
    
    # Memory optimization
    enable_memory_mapping: bool = True
    chunk_size: int = 1000
    
    # Quality control
    enable_validation: bool = True
    sample_queries: List[str] = None
    
    def __post_init__(self):
        if self.sample_queries is None:
            self.sample_queries = [
                "hukum shalat jumat",
                "cara berwudhu yang benar",
                "zakat fitrah dan zakat mal",
                "puasa ramadan bagi muslimah",
                "minuman keras dalam islam"
            ]


class EnhancedIndexingPipeline:
    """
    Comprehensive indexing pipeline integrating all enhanced components.
    """
    
    def __init__(self, config: IndexingConfig = None):
        """
        Initialize the enhanced indexing pipeline.
        
        Args:
            config (IndexingConfig): Configuration object
        """
        self.config = config or IndexingConfig()
        
        # Ensure output directory exists
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.keyword_extractor = None
        self.embedding_system = None
        self.query_preprocessor = None
        self.faiss_index = None
        
        # Data storage
        self.documents = []
        self.keywords_map = {}
        self.embeddings = None
        self.metadata = {}
        
        logger.info("Enhanced Indexing Pipeline initialized")
    
    def load_documents(self) -> List[Dict]:
        """Load documents from input JSON file."""
        logger.info(f"Loading documents from {self.config.input_data_path}")
        
        try:
            with open(self.config.input_data_path, 'r', encoding='utf-8') as f:
                documents = json.load(f)
            
            logger.info(f"Loaded {len(documents)} documents")
            
            # Validate documents
            valid_documents = []
            for i, doc in enumerate(documents):
                if 'id' in doc and 'terjemah' in doc and doc['terjemah'].strip():
                    valid_documents.append(doc)
                else:
                    logger.warning(f"Document {i} missing required fields - skipping")
            
            logger.info(f"Validated {len(valid_documents)} documents")
            self.documents = valid_documents
            return valid_documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def extract_keywords(self, force_regenerate: bool = False) -> Dict[str, List[str]]:
        """
        Extract enhanced keywords from documents.
        
        Args:
            force_regenerate (bool): Force regeneration even if output exists
            
        Returns:
            Dict[str, List[str]]: Enhanced keywords map
        """
        keywords_path = self.output_dir / self.config.keywords_output
        
        # Check if keywords already exist
        if keywords_path.exists() and not force_regenerate:
            logger.info(f"Loading existing keywords from {keywords_path}")
            try:
                with open(keywords_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.keywords_map = data.get('keywords', {})
                    return self.keywords_map
            except Exception as e:
                logger.warning(f"Error loading existing keywords: {e}")
        
        logger.info("Extracting enhanced keywords...")
        
        # Initialize keyword extractor
        self.keyword_extractor = EnhancedKeywordExtractor(
            min_frequency=self.config.keyword_min_frequency,
            max_ngram=self.config.keyword_max_ngram
        )
        
        # Extract texts
        texts = [doc['terjemah'] for doc in self.documents]
        
        # Create enhanced keywords map
        self.keywords_map = self.keyword_extractor.create_enhanced_keywords_map(texts)
        
        # Save results
        self._save_keywords_map(keywords_path)
        
        logger.info(f"Extracted {len(self.keywords_map)} keyword groups")
        return self.keywords_map
    
    def _save_keywords_map(self, output_path: Path):
        """Save keywords map with metadata."""
        result = {
            "metadata": {
                "description": "Enhanced Islamic keywords extracted from hadits collections - Fixed V1",
                "extraction_version": "enhanced_v1.0",
                "min_frequency": self.config.keyword_min_frequency,
                "max_ngram": self.config.keyword_max_ngram,
                "total_groups": len(self.keywords_map),
                "total_documents": len(self.documents),
                "extraction_method": "hybrid_enhanced_islamic_semantic_grouping",
                "created_at": datetime.now().isoformat(),
                "features": [
                    "cleaned_keywords_integration",
                    "indonesian_islamic_categories", 
                    "conservative_noise_filtering",
                    "hybrid_statistical_rule_based",
                    "phrase_component_extraction",
                    "islamic_term_prioritization"
                ]
            },
            "keywords": self.keywords_map
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Keywords map saved to {output_path}")
    
    def generate_embeddings(self, force_regenerate: bool = False) -> np.ndarray:
        """
        Generate enhanced embeddings for documents.
        
        Args:
            force_regenerate (bool): Force regeneration even if output exists
            
        Returns:
            np.ndarray: Document embeddings
        """
        embeddings_path = self.output_dir / self.config.embeddings_output
        
        # Check if embeddings already exist
        if embeddings_path.exists() and not force_regenerate:
            logger.info(f"Loading existing embeddings from {embeddings_path}")
            try:
                with open(embeddings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings']
                    return self.embeddings
            except Exception as e:
                logger.warning(f"Error loading existing embeddings: {e}")
        
        logger.info("Generating enhanced embeddings...")
        
        # Initialize embedding system
        embedding_config = EmbeddingConfig(
            batch_size=self.config.embedding_batch_size,
            keywords_path=str(self.output_dir / self.config.keywords_output),
            output_path=str(embeddings_path)
        )
        
        self.embedding_system = EnhancedEmbeddingSystem(embedding_config)
        
        # Generate embeddings
        self.embeddings, enhanced_corpus = self.embedding_system.process_documents(self.documents)
        
        # Save embeddings
        self.embedding_system.save_embeddings(
            self.embeddings, 
            self.documents, 
            enhanced_corpus
        )
        
        logger.info(f"Generated embeddings with shape {self.embeddings.shape}")
        return self.embeddings
    
    def build_faiss_index(self, force_rebuild: bool = False) -> Optional[Any]:
        """
        Build FAISS index for efficient similarity search.
        
        Args:
            force_rebuild (bool): Force rebuild even if index exists
            
        Returns:
            FAISS index object or None if FAISS not available
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - skipping index building")
            return None
        
        index_path = self.output_dir / self.config.faiss_index_output
        
        # Check if index already exists
        if index_path.exists() and not force_rebuild:
            logger.info(f"Loading existing FAISS index from {index_path}")
            try:
                self.faiss_index = faiss.read_index(str(index_path))
                return self.faiss_index
            except Exception as e:
                logger.warning(f"Error loading existing index: {e}")
        
        if self.embeddings is None:
            logger.error("No embeddings available for index building")
            return None
        
        logger.info("Building FAISS index...")
        
        # Get embedding dimension
        dimension = self.embeddings.shape[1]
        n_vectors = self.embeddings.shape[0]
        
        # Choose index type based on configuration and data size
        if self.config.faiss_index_type == "Flat" or n_vectors < 1000:
            # Use flat index for small datasets or when requested
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors
            logger.info("Using Flat index")
            
        elif self.config.faiss_index_type == "IVFFlat":
            # Use IVF index for medium to large datasets
            nlist = min(self.config.faiss_nlist, n_vectors // 39)  # Heuristic: nlist = sqrt(n)
            quantizer = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            logger.info(f"Using IVFFlat index with {nlist} clusters")
            
        elif self.config.faiss_index_type == "HNSW":
            # Use HNSW for very large datasets
            self.faiss_index = faiss.IndexHNSWFlat(dimension, 32)
            logger.info("Using HNSW index")
            
        else:
            logger.error(f"Unknown index type: {self.config.faiss_index_type}")
            return None
        
        # Ensure embeddings are in the right format (float32)
        embeddings = self.embeddings.astype(np.float32)
        
        # Train index if necessary
        if hasattr(self.faiss_index, 'is_trained') and not self.faiss_index.is_trained:
            logger.info("Training FAISS index...")
            self.faiss_index.train(embeddings)
        
        # Add vectors to index
        logger.info("Adding vectors to index...")
        self.faiss_index.add(embeddings)
        
        # Set search parameters
        if hasattr(self.faiss_index, 'nprobe'):
            self.faiss_index.nprobe = self.config.faiss_nprobe
        
        # Save index
        faiss.write_index(self.faiss_index, str(index_path))
        
        logger.info(f"FAISS index built and saved to {index_path}")
        logger.info(f"Index contains {self.faiss_index.ntotal} vectors")
        
        return self.faiss_index
    
    def build_metadata(self) -> Dict[str, Any]:
        """Build comprehensive metadata for the index."""
        logger.info("Building metadata...")
        
        self.metadata = {
            "index_info": {
                "version": "enhanced_v1.0",
                "created_at": datetime.now().isoformat(),
                "total_documents": len(self.documents),
                "total_keywords": len(self.keywords_map),
                "embedding_dimension": self.embeddings.shape[1] if self.embeddings is not None else None,
                "faiss_available": FAISS_AVAILABLE,
                "faiss_index_type": self.config.faiss_index_type if FAISS_AVAILABLE else None
            },
            "processing_config": {
                "keyword_min_frequency": self.config.keyword_min_frequency,
                "keyword_max_ngram": self.config.keyword_max_ngram,
                "embedding_batch_size": self.config.embedding_batch_size,
                "faiss_nlist": self.config.faiss_nlist,
                "faiss_nprobe": self.config.faiss_nprobe
            },
            "file_paths": {
                "keywords_map": self.config.keywords_output,
                "embeddings": self.config.embeddings_output,
                "faiss_index": self.config.faiss_index_output if FAISS_AVAILABLE else None,
                "metadata": self.config.metadata_output
            },
            "quality_metrics": self._compute_quality_metrics(),
            "sample_keywords": list(self.keywords_map.keys())[:20] if self.keywords_map else []
        }
        
        # Save metadata
        metadata_path = self.output_dir / self.config.metadata_output
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Metadata saved to {metadata_path}")
        return self.metadata
    
    def _compute_quality_metrics(self) -> Dict[str, Any]:
        """Compute quality metrics for the index."""
        metrics = {}
        
        if self.keywords_map:
            # Keyword statistics
            keyword_counts = [len(variants) for variants in self.keywords_map.values()]
            metrics["keyword_stats"] = {
                "total_groups": len(self.keywords_map),
                "avg_variants_per_group": np.mean(keyword_counts),
                "max_variants": np.max(keyword_counts),
                "min_variants": np.min(keyword_counts)
            }
        
        if self.embeddings is not None:
            # Embedding statistics
            norms = np.linalg.norm(self.embeddings, axis=1)
            metrics["embedding_stats"] = {
                "mean_norm": float(np.mean(norms)),
                "std_norm": float(np.std(norms)),
                "dimension": self.embeddings.shape[1]
            }
        
        # Document statistics
        if self.documents:
            text_lengths = [len(doc.get('terjemah', '')) for doc in self.documents]
            metrics["document_stats"] = {
                "total_documents": len(self.documents),
                "avg_text_length": np.mean(text_lengths),
                "max_text_length": np.max(text_lengths),
                "min_text_length": np.min(text_lengths)
            }
        
        return metrics
    
    def validate_index(self) -> bool:
        """Validate the built index with sample queries."""
        if not self.config.enable_validation:
            return True
        
        logger.info("Validating index with sample queries...")
        
        try:
            # Initialize query preprocessor
            self.query_preprocessor = EnhancedQueryPreprocessor(
                str(self.output_dir / self.config.keywords_output)
            )
            
            # Test with sample queries
            validation_results = []
            
            for query in self.config.sample_queries:
                try:
                    # Preprocess query
                    processed_query = self.query_preprocessor.preprocess_query(query)
                    key_terms = self.query_preprocessor.extract_key_terms(query)
                    
                    # Simple validation - check if we get meaningful results
                    has_results = len(key_terms) > 0 and processed_query.strip()
                    
                    validation_results.append({
                        "query": query,
                        "processed": processed_query,
                        "key_terms": key_terms,
                        "valid": has_results
                    })
                    
                    logger.debug(f"Query '{query}' -> '{processed_query}' -> {key_terms}")
                    
                except Exception as e:
                    logger.warning(f"Validation failed for query '{query}': {e}")
                    validation_results.append({
                        "query": query,
                        "error": str(e),
                        "valid": False
                    })
            
            # Check validation results
            valid_count = sum(1 for r in validation_results if r.get('valid', False))
            success_rate = valid_count / len(validation_results)
            
            logger.info(f"Validation results: {valid_count}/{len(validation_results)} queries passed ({success_rate:.1%})")
            
            # Add validation results to metadata
            self.metadata["validation"] = {
                "success_rate": success_rate,
                "results": validation_results,
                "validated_at": datetime.now().isoformat()
            }
            
            return success_rate >= 0.8  # 80% success rate threshold
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def run_full_pipeline(self, force_regenerate: bool = False) -> bool:
        """
        Run the complete enhanced indexing pipeline.
        
        Args:
            force_regenerate (bool): Force regeneration of all components
            
        Returns:
            bool: Success status
        """
        logger.info("🚀 Starting Enhanced Indexing Pipeline - Fixed V1")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load documents
            self.load_documents()
            
            if not self.documents:
                logger.error("No valid documents found")
                return False
            
            # Step 2: Extract keywords
            self.extract_keywords(force_regenerate)
            
            # Step 3: Generate embeddings
            self.generate_embeddings(force_regenerate)
            
            # Step 4: Build FAISS index
            self.build_faiss_index(force_regenerate)
            
            # Step 5: Build metadata
            self.build_metadata()
            
            # Step 6: Validate index
            validation_success = self.validate_index()
            
            # Step 7: Print summary
            self._print_pipeline_summary(validation_success)
            
            return validation_success
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
    
    def _print_pipeline_summary(self, validation_success: bool):
        """Print comprehensive pipeline summary."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 ENHANCED INDEXING PIPELINE SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"Documents processed: {len(self.documents):,}")
        logger.info(f"Keywords extracted: {len(self.keywords_map):,} groups")
        
        if self.embeddings is not None:
            logger.info(f"Embeddings generated: {self.embeddings.shape[0]:,} x {self.embeddings.shape[1]}D")
        
        if FAISS_AVAILABLE and self.faiss_index:
            logger.info(f"FAISS index built: {self.faiss_index.ntotal:,} vectors")
        else:
            logger.info("FAISS index: Not available")
        
        logger.info(f"Validation: {'✅ PASSED' if validation_success else '❌ FAILED'}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Show file sizes
        logger.info("\nGenerated files:")
        for filename in [self.config.keywords_output, self.config.embeddings_output, 
                        self.config.faiss_index_output, self.config.metadata_output]:
            file_path = self.output_dir / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info(f"  - {filename}: {size_mb:.2f} MB")
        
        # Show top keywords
        if self.keywords_map:
            logger.info(f"\n🕌 Top keyword groups:")
            sorted_groups = sorted(self.keywords_map.items(), key=lambda x: len(x[1]), reverse=True)
            for group_name, variants in sorted_groups[:10]:
                logger.info(f"  • {group_name}: {len(variants)} variants")
        
        logger.info(f"\n✅ Enhanced indexing pipeline completed!")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Indexing Pipeline - Fixed V1")
    parser.add_argument("--input", help="Input documents JSON file")
    parser.add_argument("--output-dir", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all components")
    parser.add_argument("--keyword-freq", type=int, default=40, help="Minimum keyword frequency")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--index-type", choices=["Flat", "IVFFlat", "HNSW"], default="IVFFlat", help="FAISS index type")
    parser.add_argument("--no-validation", action="store_true", help="Skip validation")
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = IndexingConfig(
        keyword_min_frequency=args.keyword_freq,
        embedding_batch_size=args.batch_size,
        faiss_index_type=args.index_type,
        enable_validation=not args.no_validation
    )
    
    # Override paths if provided
    if args.input:
        config.input_data_path = args.input
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Run pipeline
    pipeline = EnhancedIndexingPipeline(config)
    success = pipeline.run_full_pipeline(force_regenerate=args.force)
    
    sys.exit(0 if success else 1)
