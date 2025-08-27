"""
Enhanced Retrieval System - Fixed V1
===================================

Advanced retrieval system combining:
- Auto-adaptive keyword filtering with smart min_match logic
- Literal overlap boosting for query-document matching
- Conservative query preprocessing integration
- Comprehensive scoring with multiple ranking factors

Features:
- Hybrid semantic + keyword-based retrieval
- Auto-adaptive filtering based on keyword count
- Literal overlap boosting for better relevance
- Multiple ranking factors (semantic similarity + keyword match + literal overlap)
- Comprehensive result filtering and ranking
- Performance optimization with batch processing

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import os
import json
import pickle
import re
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    logger.error("Required dependencies not available. Please install: faiss-cpu sentence-transformers")

# Import enhanced preprocessing
sys_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

try:
    from preprocessing.query_preprocessor import preprocess_query, extract_key_terms
except ImportError:
    logger.warning("Could not import enhanced query preprocessor. Using basic preprocessing.")
    
    def preprocess_query(query: str, **kwargs) -> str:
        return re.sub(r'[^\w\s]', '', query.lower()).strip()
    
    def extract_key_terms(query: str) -> List[str]:
        return preprocess_query(query).split()


class EnhancedRetrievalSystem:
    """
    Next-generation retrieval system with hybrid semantic and keyword-based search.
    """
    
    def __init__(self,
                 embeddings_path: str = "data/processed/enhanced_embeddings_v1.pkl",
                 index_path: str = "data/processed/enhanced_faiss_index_v1.idx",
                 metadata_path: str = "data/processed/enhanced_metadata_v1.pkl",
                 model_name: str = "intfloat/e5-small-v2"):
        
        self.embeddings_path = embeddings_path
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        
        # Lazy loading
        self.index = None
        self.metadata = None
        self.model = None
        
        # Performance statistics
        self.stats = {
            'total_queries': 0,
            'avg_query_time': 0,
            'cache_hits': 0,
            'total_documents': 0
        }
        
        # Simple query cache for performance
        self.query_cache = {}
        self.max_cache_size = 100
    
    def _load_model(self) -> SentenceTransformer:
        """Lazy loading of embedding model."""
        if self.model is None:
            if not DEPENDENCIES_AVAILABLE:
                raise ImportError("Required dependencies not available")
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        
        return self.model
    
    def _load_index_and_metadata(self) -> Tuple[Any, List[Dict]]:
        """Load FAISS index and metadata."""
        if self.index is None or self.metadata is None:
            if not DEPENDENCIES_AVAILABLE:
                raise ImportError("Required dependencies not available")
            
            # Load FAISS index
            if Path(self.index_path).exists():
                logger.info(f"Loading FAISS index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)
            else:
                logger.warning("FAISS index not found. Need to build index first.")
                return None, None
            
            # Load metadata
            if Path(self.metadata_path).exists():
                logger.info(f"Loading metadata from {self.metadata_path}")
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                logger.warning("Metadata not found. Need to build index first.")
                return None, None
            
            # Update stats
            self.stats['total_documents'] = len(self.metadata)
            logger.info(f"Loaded index with {self.stats['total_documents']:,} documents")
        
        return self.index, self.metadata
    
    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        Build FAISS index from embeddings file.
        
        Args:
            force_rebuild: Whether to force rebuild if index exists
            
        Returns:
            bool: Success status
        """
        if not force_rebuild and Path(self.index_path).exists() and Path(self.metadata_path).exists():
            logger.info("Index already exists. Use force_rebuild=True to rebuild.")
            return True
        
        if not DEPENDENCIES_AVAILABLE:
            logger.error("Cannot build index: missing dependencies")
            return False
        
        if not Path(self.embeddings_path).exists():
            logger.error(f"Embeddings file not found: {self.embeddings_path}")
            return False
        
        try:
            logger.info("🔨 Building enhanced FAISS index...")
            
            # Load embeddings data
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = data['embeddings']
            documents = data['documents']
            
            # Build FAISS index (using cosine similarity via inner product)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            logger.info(f"Adding {len(embeddings)} embeddings to index...")
            index.add(embeddings.astype(np.float32))
            
            # Prepare metadata with enhanced tags
            metadata = []
            for i, doc in enumerate(documents):
                # Include semantic tags if available from corpus
                tags = ""
                if 'corpus' in data and i < len(data['corpus']):
                    corpus_entry = data['corpus'][i]
                    # Extract tags from "Kata kunci penting: ..." part
                    match = re.search(r'Kata kunci penting:\s*([^.]+)', corpus_entry)
                    if match:
                        tags = match.group(1).strip()
                
                metadata.append({
                    "id": doc.get("id", f"doc_{i}"),
                    "kitab": doc.get("kitab", ""),
                    "arab": doc.get("arab_bersih", ""),
                    "arab_asli": doc.get("arab_asli", doc.get("arab_bersih", "")),
                    "terjemah": doc.get("terjemah", ""),
                    "tags": tags,
                    "index": i
                })
            
            # Save index and metadata
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.metadata_path).parent.mkdir(parents=True, exist_ok=True)
            
            faiss.write_index(index, self.index_path)
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"✅ Index built successfully!")
            logger.info(f"Index saved: {self.index_path}")
            logger.info(f"Metadata saved: {self.metadata_path}")
            logger.info(f"Total documents indexed: {len(metadata):,}")
            
            # Update instance variables
            self.index = index
            self.metadata = metadata
            self.stats['total_documents'] = len(metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    def calculate_literal_overlap(self, query_terms: List[str], document_text: str) -> float:
        """
        Calculate literal overlap between query terms and document.
        
        Args:
            query_terms: List of query terms
            document_text: Document text to compare against
            
        Returns:
            float: Literal overlap score (0-1)
        """
        if not query_terms or not document_text:
            return 0.0
        
        doc_text_lower = document_text.lower()
        matched_terms = 0
        
        for term in query_terms:
            # Use word boundary matching for better precision
            pattern = rf'\b{re.escape(term.lower())}\b'
            if re.search(pattern, doc_text_lower):
                matched_terms += 1
        
        return matched_terms / len(query_terms)
    
    def filter_by_keywords(self, candidates: List[Dict], 
                          required_keywords: List[str],
                          min_match: int = 1) -> List[Dict]:
        """
        Filter candidates by required keywords with flexible matching.
        
        Args:
            candidates: List of candidate documents
            required_keywords: Keywords that should be present
            min_match: Minimum number of keywords that must match
            
        Returns:
            List[Dict]: Filtered candidates
        """
        if not required_keywords or min_match <= 0:
            return candidates
        
        filtered_candidates = []
        
        for candidate in candidates:
            # Combine text and tags for keyword matching
            search_text = candidate.get('terjemah', '') + ' ' + candidate.get('tags', '')
            search_text_lower = search_text.lower()
            
            # Count keyword matches
            matched_keywords = 0
            for keyword in required_keywords:
                pattern = rf'\b{re.escape(keyword.lower())}\b'
                if re.search(pattern, search_text_lower):
                    matched_keywords += 1
            
            # Include if meets minimum match requirement
            if matched_keywords >= min_match:
                candidate['keyword_matches'] = matched_keywords
                candidate['keyword_match_ratio'] = matched_keywords / len(required_keywords)
                filtered_candidates.append(candidate)
        
        return filtered_candidates
    
    def determine_auto_min_match(self, required_keywords: List[str]) -> int:
        """
        Automatically determine optimal min_match based on keyword count.
        
        Args:
            required_keywords: List of required keywords
            
        Returns:
            int: Optimal min_match value
        """
        keyword_count = len(required_keywords)
        
        if keyword_count == 0:
            return 0  # No filtering if no keywords
        elif keyword_count == 1:
            return 1  # Must match the only keyword
        elif keyword_count <= 3:
            return 1  # At least 1 for small keyword sets
        elif keyword_count <= 5:
            return max(1, keyword_count // 2)  # Half for medium sets
        else:
            return max(2, keyword_count // 3)  # One-third for large sets
    
    def apply_literal_overlap_boost(self, candidates: List[Dict],
                                   query_terms: List[str],
                                   boost_factor: float = 0.2) -> List[Dict]:
        """
        Apply literal overlap boosting to candidate scores.
        
        Args:
            candidates: List of candidate documents
            query_terms: Original query terms
            boost_factor: Boost factor for literal overlap
            
        Returns:
            List[Dict]: Candidates with boosted scores
        """
        if not query_terms or boost_factor <= 0:
            return candidates
        
        for candidate in candidates:
            # Calculate literal overlap
            overlap_score = self.calculate_literal_overlap(query_terms, candidate.get('terjemah', ''))
            
            # Apply boost to semantic score
            original_score = candidate.get('score', 0)
            boost = overlap_score * boost_factor
            candidate['score'] = original_score + boost
            candidate['literal_overlap'] = overlap_score
            candidate['boost_applied'] = boost
        
        return candidates
    
    def query_enhanced(self,
                      raw_query: str,
                      top_k: int = 5,
                      required_keywords: Optional[List[str]] = None,
                      min_match: Optional[int] = None,
                      apply_literal_boost: bool = True,
                      boost_factor: float = 0.2) -> List[Dict]:
        """
        Enhanced query processing with all optimizations.
        
        Args:
            raw_query: Original user query
            top_k: Number of results to return
            required_keywords: Keywords that should be present (auto-extracted if None)
            min_match: Minimum keyword matches (auto-determined if None)
            apply_literal_boost: Whether to apply literal overlap boosting
            boost_factor: Factor for literal overlap boost
            
        Returns:
            List[Dict]: Enhanced search results
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{raw_query}_{top_k}_{min_match}_{apply_literal_boost}"
        if cache_key in self.query_cache:
            self.stats['cache_hits'] += 1
            logger.info("Retrieved result from cache")
            return self.query_cache[cache_key]
        
        try:
            # Step 1: Load index and metadata
            index, metadata = self._load_index_and_metadata()
            if index is None or metadata is None:
                logger.error("Index or metadata not available. Please build index first.")
                return []
            
            # Step 2: Preprocess query and extract keywords
            if required_keywords is None:
                required_keywords = extract_key_terms(raw_query)
            
            processed_query = preprocess_query(raw_query, preserve_query_intent=True)
            
            # Create optimized query for embedding
            if required_keywords:
                optimized_query = f"passage: {raw_query}. Kata kunci penting: {' '.join(required_keywords)}"
            else:
                optimized_query = f"passage: {raw_query}"
            
            logger.info(f"Processing query: '{raw_query}'")
            logger.info(f"Optimized query: '{optimized_query}'")
            logger.info(f"Required keywords: {required_keywords}")
            
            # Step 3: Auto-determine min_match if not specified
            if min_match is None:
                min_match = self.determine_auto_min_match(required_keywords)
            
            logger.info(f"Min match (auto-determined): {min_match}")
            
            # Step 4: Generate query embedding
            model = self._load_model()
            query_embedding = model.encode([optimized_query], normalize_embeddings=True)
            
            # Step 5: Semantic search
            # Get more candidates for filtering
            search_k = min(top_k * 4, len(metadata))  
            scores, indices = index.search(query_embedding.astype(np.float32), search_k)
            
            # Step 6: Prepare candidates
            candidates = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0:  # Valid index
                    candidate = metadata[idx].copy()
                    candidate['score'] = float(score)
                    candidate['semantic_rank'] = i
                    candidates.append(candidate)
            
            logger.info(f"Semantic search found {len(candidates)} candidates")
            
            # Step 7: Filter by keywords
            if required_keywords and min_match > 0:
                filtered_candidates = self.filter_by_keywords(candidates, required_keywords, min_match)
                logger.info(f"Keyword filtering: {len(candidates)} -> {len(filtered_candidates)} candidates")
                candidates = filtered_candidates
            
            # Step 8: Apply literal overlap boosting
            if apply_literal_boost and required_keywords:
                candidates = self.apply_literal_overlap_boost(
                    candidates, required_keywords, boost_factor
                )
                logger.info(f"Applied literal overlap boosting with factor {boost_factor}")
            
            # Step 9: Re-rank and limit results
            candidates.sort(key=lambda x: x['score'], reverse=True)
            final_results = candidates[:top_k]
            
            # Step 10: Add processing metadata
            processing_time = time.time() - start_time
            
            for result in final_results:
                result['processing_time'] = processing_time
                result['query_info'] = {
                    'original_query': raw_query,
                    'processed_query': processed_query,
                    'optimized_query': optimized_query,
                    'required_keywords': required_keywords,
                    'min_match_used': min_match
                }
            
            # Step 11: Update statistics and cache
            self.stats['total_queries'] += 1
            self.stats['avg_query_time'] = (
                (self.stats['avg_query_time'] * (self.stats['total_queries'] - 1) + processing_time)
                / self.stats['total_queries']
            )
            
            # Cache management
            if len(self.query_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
            
            self.query_cache[cache_key] = final_results
            
            # Log results
            logger.info(f"Query processing completed in {processing_time*1000:.2f}ms")
            logger.info(f"Filter stats: {{'total_candidates': {search_k}, 'keyword_filtered': {len(candidates)}, 'final_results': {len(final_results)}}}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        return {
            **self.stats,
            'cache_size': len(self.query_cache),
            'index_loaded': self.index is not None,
            'metadata_loaded': self.metadata is not None,
            'model_loaded': self.model is not None
        }


# Convenience functions for backward compatibility
def query_hadits_return(raw_query: str, 
                       optimized_query: str = None,
                       top_k: int = 5, 
                       required_keywords: List[str] = [], 
                       min_match: int = None,
                       apply_literal_boost: bool = True,
                       boost_factor: float = 0.2) -> List[Dict]:
    """
    Convenience function matching the original API.
    """
    # Initialize global retrieval system
    if not hasattr(query_hadits_return, '_system'):
        query_hadits_return._system = EnhancedRetrievalSystem()
    
    system = query_hadits_return._system
    
    return system.query_enhanced(
        raw_query=raw_query,
        top_k=top_k,
        required_keywords=required_keywords if required_keywords else None,
        min_match=min_match,
        apply_literal_boost=apply_literal_boost,
        boost_factor=boost_factor
    )


def main():
    """Test the enhanced retrieval system."""
    
    # Check dependencies
    if not DEPENDENCIES_AVAILABLE:
        logger.error("Missing required dependencies. Please install:")
        logger.error("pip install faiss-cpu sentence-transformers")
        return
    
    # Initialize system
    retrieval_system = EnhancedRetrievalSystem()
    
    # Build index if needed
    if not retrieval_system.build_index():
        logger.error("Failed to build/load index")
        return
    
    # Test queries
    test_queries = [
        "apa hukum riba?",
        "bagaimana cara shalat yang benar?",
        "berikan hadis tentang sedekah",
        "apa itu zakat fitrah?",
        "hukum minuman keras dalam islam"
    ]
    
    logger.info("Testing enhanced retrieval system...")
    
    for query in test_queries:
        logger.info(f"\n🔍 Testing query: '{query}'")
        
        results = retrieval_system.query_enhanced(
            raw_query=query,
            top_k=3,
            apply_literal_boost=True
        )
        
        logger.info(f"Found {len(results)} results:")
        for i, result in enumerate(results):
            logger.info(f"  [{i+1}] Score: {result['score']:.4f} | {result['kitab']} | {result['terjemah'][:100]}...")
    
    # Show system stats
    stats = retrieval_system.get_system_stats()
    logger.info(f"\n📊 System Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()