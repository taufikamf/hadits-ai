"""
Enhanced Retrieval System - Fixed V1
===================================

Advanced retrieval system combining:
- Auto-adaptive keyword filtering with smart min_match logic
- Literal overlap boosting for query-document matching
- Conservative query preprocessing integration
- Comprehensive scoring with multiple ranking factors
- Semantic similarity with keyword-aware filtering

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import json
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.metrics.pairwise import cosine_similarity

# Import our enhanced preprocessing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.query_preprocessor import (
    EnhancedQueryPreprocessor, preprocess_query, extract_key_terms, analyze_query_intent
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for enhanced retrieval system."""
    # Embedding and data paths
    embeddings_path: str = "../../data/enhanced_index_v1/enhanced_embeddings_v1.pkl"
    keywords_map_path: str = "../../data/enhanced_index_v1/enhanced_keywords_map_v1.json"
    
    # Retrieval parameters
    top_k: int = 50
    min_score_threshold: float = 0.1
    
    # Keyword filtering
    enable_keyword_filtering: bool = True
    auto_adaptive_min_match: bool = True
    base_min_match: float = 0.3
    adaptive_scaling: float = 0.1
    
    # Scoring weights
    semantic_weight: float = 0.7
    keyword_weight: float = 0.2
    literal_overlap_weight: float = 0.1
    
    # Query analysis
    enable_query_analysis: bool = True
    islamic_context_boost: float = 1.2
    query_intent_boost: float = 1.1
    
    # Result processing
    enable_reranking: bool = True
    diversity_factor: float = 0.1
    max_results: int = 20


@dataclass
class RetrievalResult:
    """Single retrieval result with comprehensive metadata."""
    document_id: str
    document: Dict[str, Any]
    score: float
    semantic_score: float
    keyword_score: float
    literal_overlap_score: float
    matched_keywords: List[str] = field(default_factory=list)
    query_analysis: Optional[Dict] = None
    rank: int = 0


class EnhancedRetrievalSystem:
    """
    Advanced retrieval system with multi-factor scoring and adaptive filtering.
    """
    
    def __init__(self, config: RetrievalConfig = None):
        """
        Initialize the enhanced retrieval system.
        
        Args:
            config (RetrievalConfig): Configuration object
        """
        self.config = config or RetrievalConfig()
        self.query_preprocessor = EnhancedQueryPreprocessor(self.config.keywords_map_path)
        
        # Data storage
        self.embeddings = None
        self.documents = None
        self.enhanced_corpus = None
        self.keywords_map = {}
        self.reverse_keywords_map = {}
        
        # Load components
        self._load_embeddings_and_documents()
        self._load_keywords_map()
        
        logger.info("Enhanced Retrieval System initialized")
    
    def _load_embeddings_and_documents(self):
        """Load embeddings and documents from pickle file."""
        try:
            logger.info(f"Loading embeddings from {self.config.embeddings_path}")
            
            with open(self.config.embeddings_path, "rb") as f:
                data = pickle.load(f)
            
            self.embeddings = data["embeddings"]
            self.documents = data["documents"]
            self.enhanced_corpus = data.get("enhanced_corpus", [])
            
            logger.info(f"Loaded {len(self.documents)} documents with {self.embeddings.shape[1]}D embeddings")
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    def _load_keywords_map(self):
        """Load keywords map for filtering and scoring."""
        try:
            with open(self.config.keywords_map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.keywords_map = data.get('keywords', {})
            
            # Build reverse map for fast lookup
            self.reverse_keywords_map = {}
            for canonical_term, variants in self.keywords_map.items():
                for variant in variants:
                    self.reverse_keywords_map[variant.lower()] = canonical_term
                self.reverse_keywords_map[canonical_term.lower()] = canonical_term
            
            logger.info(f"Loaded {len(self.keywords_map)} keyword groups")
            
        except Exception as e:
            logger.warning(f"Could not load keywords map: {e}")
            self.keywords_map = {}
            self.reverse_keywords_map = {}
    
    def compute_semantic_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute semantic similarity between query and all documents.
        
        Args:
            query_embedding (np.ndarray): Query embedding
            
        Returns:
            np.ndarray: Similarity scores for all documents
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        return similarities
    
    def compute_keyword_overlap(self, query_terms: List[str], document_index: int) -> Tuple[float, List[str]]:
        """
        Compute keyword overlap score between query and document.
        
        Args:
            query_terms (List[str]): Preprocessed query terms
            document_index (int): Document index
            
        Returns:
            Tuple[float, List[str]]: Keyword score and matched keywords
        """
        if document_index >= len(self.documents):
            return 0.0, []
        
        document = self.documents[document_index]
        doc_text = document.get("terjemah", "").lower()
        
        matched_keywords = []
        total_matches = 0
        
        for term in query_terms:
            # Check direct term match
            if term.lower() in doc_text:
                matched_keywords.append(term)
                total_matches += 1
                continue
            
            # Check if term maps to a keyword group
            canonical_term = self.reverse_keywords_map.get(term.lower())
            if canonical_term and canonical_term in self.keywords_map:
                # Check if any variant of this keyword group appears in document
                variants = self.keywords_map[canonical_term]
                for variant in variants:
                    if variant.lower() in doc_text:
                        matched_keywords.append(canonical_term)
                        total_matches += 1
                        break
        
        # Calculate overlap score
        if query_terms:
            keyword_score = total_matches / len(query_terms)
        else:
            keyword_score = 0.0
        
        return keyword_score, matched_keywords
    
    def compute_literal_overlap(self, query: str, document_index: int) -> float:
        """
        Compute literal text overlap between query and document.
        
        Args:
            query (str): Original query
            document_index (int): Document index
            
        Returns:
            float: Literal overlap score
        """
        if document_index >= len(self.documents):
            return 0.0
        
        document = self.documents[document_index]
        doc_text = document.get("terjemah", "").lower()
        query_lower = query.lower()
        
        # Simple word-level overlap
        query_words = set(query_lower.split())
        doc_words = set(doc_text.split())
        
        if query_words:
            overlap = len(query_words.intersection(doc_words)) / len(query_words)
        else:
            overlap = 0.0
        
        return overlap
    
    def apply_adaptive_keyword_filtering(self, query_terms: List[str], 
                                       keyword_scores: List[Tuple[int, float, List[str]]]) -> List[Tuple[int, float, List[str]]]:
        """
        Apply auto-adaptive keyword filtering based on query characteristics.
        
        Args:
            query_terms (List[str]): Query terms
            keyword_scores (List[Tuple[int, float, List[str]]]): (index, score, matched_keywords)
            
        Returns:
            List[Tuple[int, float, List[str]]]: Filtered results
        """
        if not self.config.enable_keyword_filtering or not keyword_scores:
            return keyword_scores
        
        # Calculate adaptive minimum match threshold
        if self.config.auto_adaptive_min_match:
            num_query_terms = len(query_terms)
            
            if num_query_terms <= 2:
                # For short queries, require high precision
                min_match = max(0.5, self.config.base_min_match)
            elif num_query_terms <= 4:
                # For medium queries, moderate precision
                min_match = self.config.base_min_match
            else:
                # For long queries, lower precision but still meaningful
                min_match = max(0.2, self.config.base_min_match - self.config.adaptive_scaling * (num_query_terms - 4))
                
            logger.debug(f"Adaptive min_match for {num_query_terms} terms: {min_match:.2f}")
        else:
            min_match = self.config.base_min_match
        
        # Filter based on keyword overlap
        filtered_results = []
        for doc_idx, keyword_score, matched_keywords in keyword_scores:
            if keyword_score >= min_match:
                filtered_results.append((doc_idx, keyword_score, matched_keywords))
        
        logger.debug(f"Keyword filtering: {len(filtered_results)}/{len(keyword_scores)} docs passed (min_match={min_match:.2f})")
        
        return filtered_results
    
    def apply_query_context_boosting(self, scores: List[float], query_analysis: Dict) -> List[float]:
        """
        Apply context-based score boosting based on query analysis.
        
        Args:
            scores (List[float]): Base scores
            query_analysis (Dict): Query analysis results
            
        Returns:
            List[float]: Boosted scores
        """
        if not self.config.enable_query_analysis:
            return scores
        
        boosted_scores = scores.copy()
        
        # Islamic context boost
        islamic_strength = query_analysis.get('islamic_context_strength', 0)
        if islamic_strength > 0.3:  # Significant Islamic context
            boost_factor = 1 + (islamic_strength * (self.config.islamic_context_boost - 1))
            boosted_scores = [score * boost_factor for score in boosted_scores]
            logger.debug(f"Applied Islamic context boost: {boost_factor:.2f}")
        
        # Query intent boost (for instructional queries)
        if query_analysis.get('has_action_intent', False):
            boost_factor = self.config.query_intent_boost
            boosted_scores = [score * boost_factor for score in boosted_scores]
            logger.debug(f"Applied query intent boost: {boost_factor:.2f}")
        
        return boosted_scores
    
    def rerank_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Apply advanced reranking to improve result quality and diversity.
        
        Args:
            results (List[RetrievalResult]): Initial results
            
        Returns:
            List[RetrievalResult]: Reranked results
        """
        if not self.config.enable_reranking or len(results) <= 1:
            return results
        
        # Simple diversity-aware reranking
        reranked = []
        remaining = results.copy()
        
        # Always take the top result
        if remaining:
            top_result = remaining.pop(0)
            reranked.append(top_result)
        
        # For remaining results, balance score and diversity
        while remaining and len(reranked) < self.config.max_results:
            best_idx = 0
            best_score = 0
            
            for i, candidate in enumerate(remaining):
                # Base score
                score = candidate.score
                
                # Diversity penalty (avoid very similar documents)
                diversity_penalty = 0
                for existing in reranked:
                    # Simple diversity based on matched keywords
                    existing_keywords = set(existing.matched_keywords)
                    candidate_keywords = set(candidate.matched_keywords)
                    
                    if existing_keywords and candidate_keywords:
                        overlap = len(existing_keywords.intersection(candidate_keywords)) / len(existing_keywords.union(candidate_keywords))
                        diversity_penalty += overlap * self.config.diversity_factor
                
                final_score = score - diversity_penalty
                
                if final_score > best_score:
                    best_score = final_score
                    best_idx = i
            
            # Add best candidate to reranked results
            reranked.append(remaining.pop(best_idx))
        
        # Update ranks
        for i, result in enumerate(reranked):
            result.rank = i + 1
        
        logger.debug(f"Reranked {len(reranked)} results")
        return reranked
    
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """
        Enhanced retrieval with multi-factor scoring and adaptive filtering.
        
        Args:
            query (str): User query
            top_k (int, optional): Number of results to return
            
        Returns:
            List[RetrievalResult]: Ranked retrieval results
        """
        top_k = top_k or self.config.top_k
        
        logger.info(f"Processing query: '{query}'")
        
        # Step 1: Query analysis and preprocessing
        query_analysis = analyze_query_intent(query) if self.config.enable_query_analysis else {}
        query_terms = extract_key_terms(query)
        processed_query = preprocess_query(query)
        
        logger.debug(f"Query terms: {query_terms}")
        logger.debug(f"Processed query: '{processed_query}'")
        
        # Step 2: Generate query embedding
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("intfloat/e5-small-v2")
        
        # Use enhanced query format for embedding
        enhanced_query = f"query: {processed_query}"
        query_embedding = model.encode([enhanced_query], normalize_embeddings=True)[0]
        
        # Step 3: Compute semantic similarities
        semantic_scores = self.compute_semantic_similarity(query_embedding)
        
        # Step 4: Compute keyword overlaps for all documents
        keyword_data = []
        for i in range(len(self.documents)):
            keyword_score, matched_keywords = self.compute_keyword_overlap(query_terms, i)
            keyword_data.append((i, keyword_score, matched_keywords))
        
        # Step 5: Apply adaptive keyword filtering
        filtered_keyword_data = self.apply_adaptive_keyword_filtering(query_terms, keyword_data)
        
        # Step 6: Compute comprehensive scores for filtered documents
        results = []
        
        for doc_idx, keyword_score, matched_keywords in filtered_keyword_data:
            # Get scores
            semantic_score = float(semantic_scores[doc_idx])
            literal_overlap_score = self.compute_literal_overlap(query, doc_idx)
            
            # Combined score
            combined_score = (
                self.config.semantic_weight * semantic_score +
                self.config.keyword_weight * keyword_score +
                self.config.literal_overlap_weight * literal_overlap_score
            )
            
            # Apply minimum score threshold
            if combined_score >= self.config.min_score_threshold:
                result = RetrievalResult(
                    document_id=self.documents[doc_idx].get("id", str(doc_idx)),
                    document=self.documents[doc_idx],
                    score=combined_score,
                    semantic_score=semantic_score,
                    keyword_score=keyword_score,
                    literal_overlap_score=literal_overlap_score,
                    matched_keywords=matched_keywords,
                    query_analysis=query_analysis
                )
                results.append(result)
        
        # Step 7: Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Step 8: Apply query context boosting
        if results and self.config.enable_query_analysis:
            scores = [r.score for r in results]
            boosted_scores = self.apply_query_context_boosting(scores, query_analysis)
            
            for result, boosted_score in zip(results, boosted_scores):
                result.score = boosted_score
            
            # Re-sort after boosting
            results.sort(key=lambda x: x.score, reverse=True)
        
        # Step 9: Take top-k results
        top_results = results[:top_k]
        
        # Step 10: Apply reranking for final optimization
        if self.config.enable_reranking:
            top_results = self.rerank_results(top_results)
        
        # Step 11: Final ranking assignment
        for i, result in enumerate(top_results):
            result.rank = i + 1
        
        logger.info(f"Retrieved {len(top_results)} results (from {len(results)} candidates)")
        
        return top_results[:self.config.max_results]
    
    def get_retrieval_stats(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for retrieval results.
        
        Args:
            results (List[RetrievalResult]): Retrieval results
            
        Returns:
            Dict[str, Any]: Statistics
        """
        if not results:
            return {"error": "No results to analyze"}
        
        scores = [r.score for r in results]
        semantic_scores = [r.semantic_score for r in results]
        keyword_scores = [r.keyword_score for r in results]
        
        all_matched_keywords = []
        for r in results:
            all_matched_keywords.extend(r.matched_keywords)
        
        return {
            "total_results": len(results),
            "score_stats": {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores))
            },
            "semantic_score_stats": {
                "mean": float(np.mean(semantic_scores)),
                "std": float(np.std(semantic_scores))
            },
            "keyword_score_stats": {
                "mean": float(np.mean(keyword_scores)),
                "std": float(np.std(keyword_scores))
            },
            "keyword_coverage": len(set(all_matched_keywords)),
            "avg_keywords_per_result": len(all_matched_keywords) / len(results) if results else 0,
            "top_matched_keywords": list(set(all_matched_keywords))[:10]
        }


def format_retrieval_results(results: List[RetrievalResult], include_scores: bool = True) -> str:
    """
    Format retrieval results for display.
    
    Args:
        results (List[RetrievalResult]): Results to format
        include_scores (bool): Whether to include score details
        
    Returns:
        str: Formatted results
    """
    if not results:
        return "No results found."
    
    formatted = []
    formatted.append(f"Found {len(results)} results:")
    formatted.append("=" * 50)
    
    for i, result in enumerate(results):
        formatted.append(f"\n[{result.rank}] ID: {result.document_id}")
        
        # Document text (truncated)
        text = result.document.get("terjemah", "")
        formatted.append(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")
        
        # Matched keywords
        if result.matched_keywords:
            formatted.append(f"Keywords: {', '.join(result.matched_keywords)}")
        
        # Scores (if requested)
        if include_scores:
            formatted.append(f"Score: {result.score:.3f} (S:{result.semantic_score:.3f}, K:{result.keyword_score:.3f}, L:{result.literal_overlap_score:.3f})")
    
    return "\n".join(formatted)


# Main execution and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Retrieval System - Fixed V1")
    parser.add_argument("--query", required=True, help="Query to test")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--embeddings-path", help="Path to embeddings file")
    parser.add_argument("--keywords-path", help="Path to keywords map")
    parser.add_argument("--show-scores", action="store_true", help="Show detailed scores")
    parser.add_argument("--stats", action="store_true", help="Show retrieval statistics")
    
    args = parser.parse_args()
    
    # Create config
    config = RetrievalConfig(top_k=args.top_k)
    if args.embeddings_path:
        config.embeddings_path = args.embeddings_path
    if args.keywords_path:
        config.keywords_map_path = args.keywords_path
    
    try:
        # Initialize retrieval system
        logger.info("Initializing Enhanced Retrieval System...")
        retrieval_system = EnhancedRetrievalSystem(config)
        
        # Perform retrieval
        logger.info(f"Processing query: '{args.query}'")
        results = retrieval_system.retrieve(args.query, args.top_k)
        
        # Display results
        print(format_retrieval_results(results, include_scores=args.show_scores))
        
        # Show statistics if requested
        if args.stats:
            stats = retrieval_system.get_retrieval_stats(results)
            print(f"\n\nRetrieval Statistics:")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        
    except Exception as e:
        logger.error(f"Error in retrieval system: {e}")
        sys.exit(1)
