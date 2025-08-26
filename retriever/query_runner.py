"""
Enhanced Query Runner for Hadith AI
===================================

Enhanced retrieval system with:
- Keywords-based FAISS filtering
- Literal overlap ranking boost  
- Enhanced scoring algorithm
- Comprehensive logging

Author: Hadith AI Team
Date: 2024
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available")

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not available")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: SentenceTransformers not available")

try:
    from sklearn.preprocessing import normalize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available")
from typing import List, Dict, Optional, Tuple
from utils import query_optimizer
import pickle


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup query logs directory
LOGS_DIR = Path("data/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
QUERY_LOG_FILE = LOGS_DIR / "query_logs.jsonl"

# --- Env & Config ---
# Remove the load_dotenv call that's causing issues
FAISS_INDEX_PATH = "./db/hadits_faiss.index"
FAISS_METADATA_PATH = "./db/hadits_metadata.pkl"
MODEL_NAME = "intfloat/e5-small-v2"

try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("Warning: python-dotenv not available")

# --- Init once ---
if SENTENCE_TRANSFORMERS_AVAILABLE:
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Warning: Could not load SentenceTransformer: {e}")
        model = None
else:
    model = None

def load_chroma_collection():
    """Load ChromaDB collection (legacy)."""
    if not CHROMADB_AVAILABLE:
        raise ImportError("ChromaDB not available")
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)

def load_faiss_index() -> Tuple[Optional["faiss.IndexFlatIP"], List[Dict]]:
    """Load FAISS index and metadata."""
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not available")
    
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, "rb") as f:
            metadatas = pickle.load(f)
        return index, metadatas
    except Exception as e:
        logger.error(f"Error loading FAISS index: {e}")
        raise

def get_query_embedding(query: str) -> Optional["numpy.ndarray"]:
    """Generate embedding for query."""
    if not model:
        raise RuntimeError("SentenceTransformer model not available")
    
    try:
        embedding = model.encode([query], convert_to_numpy=True)
        if SKLEARN_AVAILABLE:
            embedding = normalize(embedding, axis=1)
        return embedding.astype("float32")
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        raise

def calculate_literal_overlap(query: str, text: str) -> float:
    """
    Calculate literal overlap score between query and text.
    
    Args:
        query: Original query string
        text: Document text to compare
        
    Returns:
        Overlap score between 0 and 1
    """
    if not query or not text:
        return 0.0
    
    # Preprocess both texts
    from preprocessing.query_preprocessor import preprocess_query
    
    query_processed = preprocess_query(query, remove_stopwords=True)
    text_processed = preprocess_query(text, remove_stopwords=True)
    
    query_words = set(query_processed.split())
    text_words = set(text_processed.split())
    
    if not query_words:
        return 0.0
    
    # Calculate overlap ratio
    overlap = query_words.intersection(text_words)
    overlap_score = len(overlap) / len(query_words)
    
    return min(overlap_score, 1.0)  # Cap at 1.0

def boost_literal_overlap_score(documents: List[Dict], query: str, boost_factor: float = 0.2) -> List[Dict]:
    """
    Boost ranking scores based on literal overlap with query.
    
    Args:
        documents: List of document results
        query: Original query string
        boost_factor: How much to boost scores (0.0 to 1.0)
        
    Returns:
        Documents with boosted scores
    """
    boosted_docs = []
    
    for doc in documents:
        # Calculate literal overlap for terjemah text
        terjemah_text = doc.get("terjemah", "")
        overlap_score = calculate_literal_overlap(query, terjemah_text)
        
        # Apply boost to original score
        original_score = doc.get("score", 0.0)
        boost_amount = overlap_score * boost_factor
        boosted_score = original_score + boost_amount
        
        # Create boosted document
        boosted_doc = doc.copy()
        boosted_doc["score"] = boosted_score
        boosted_doc["literal_overlap"] = overlap_score
        boosted_doc["boost_applied"] = boost_amount
        
        boosted_docs.append(boosted_doc)
        
        if overlap_score > 0:
            logger.debug(f"Applied literal overlap boost: {overlap_score:.3f} -> +{boost_amount:.3f}")
    
    # Re-sort by boosted scores
    boosted_docs.sort(key=lambda x: x["score"], reverse=False)  # FAISS uses distance (lower is better)
    
    return boosted_docs

def enhanced_keyword_match(text: str, keywords: List[str], min_match: int = 2) -> Tuple[bool, Dict]:
    """
    Enhanced keyword matching with detailed statistics.
    
    Args:
        text: Text to search in
        keywords: Keywords to search for
        min_match: Minimum number of keywords that must match
        
    Returns:
        Tuple of (matches_criteria, match_details)
    """
    if not keywords:
        return True, {"matched_keywords": [], "match_count": 0, "match_ratio": 0.0}
    
    text_lower = text.lower()
    matched_keywords = []
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            matched_keywords.append(keyword)
    
    match_count = len(matched_keywords)
    match_ratio = match_count / len(keywords) if keywords else 0.0
    meets_criteria = match_count >= min_match
    
    match_details = {
        "matched_keywords": matched_keywords,
        "match_count": match_count,
        "match_ratio": match_ratio,
        "total_keywords": len(keywords),
        "min_required": min_match
    }
    
    return meets_criteria, match_details

def log_query_results(query: str, keywords: List[str], results: List[Dict], 
                     processing_time: Optional[float] = None):
    """
    Log query processing results in JSONL format.
    
    Args:
        query: Original query
        keywords: Extracted keywords
        results: Top retrieval results
        processing_time: Time taken for processing
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "keywords": keywords,
            "results_count": len(results),
            "processing_time_ms": processing_time,
            "top_results": [
                {
                    "rank": result.get("rank", 0),
                    "kitab": result.get("kitab", ""),
                    "id": result.get("id", ""),
                    "score": result.get("score", 0.0),
                    "literal_overlap": result.get("literal_overlap", 0.0),
                    "boost_applied": result.get("boost_applied", 0.0),
                    "terjemah_preview": result.get("terjemah", "")[:100] + "..." if result.get("terjemah") else ""
                }
                for result in results[:5]  # Log top 5 results
            ]
        }
        
        # Append to JSONL log file
        with open(QUERY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
    except Exception as e:
        logger.error(f"Error logging query results: {e}")

def contains_all_keywords(text: str, keywords: list[str]) -> bool:
    """Legacy function for backward compatibility."""
    text = text.lower()
    return all(kw.lower() in text for kw in keywords)

def keyword_match(text: str, keywords: List[str], min_match: int = 2) -> bool:
    """
    Legacy keyword matching function for backward compatibility.
    
    Args:
        text (str): Teks yang akan dicek
        keywords (List[str]): Daftar keywords untuk dicari
        min_match (int): Jumlah minimum keyword yang harus match
        
    Returns:
        bool: True jika text mengandung minimal min_match keywords
    """
    meets_criteria, _ = enhanced_keyword_match(text, keywords, min_match)
    return meets_criteria

def query_hadits_return(
    raw_query: str, 
    optimized_query: str = None, 
    top_k: int = 5, 
    required_keywords: List[str] = [], 
    min_match: int = 2,
    apply_literal_boost: bool = True,
    boost_factor: float = 0.2
) -> List[Dict]:
    """
    Enhanced hadith search with semantic search + keyword filtering + literal overlap boosting.
    
    Args:
        raw_query (str): Query asli dari user
        optimized_query (str, optional): Query yang sudah dioptimasi. Defaults to None.
        top_k (int): Jumlah hasil yang diinginkan. Defaults to 5.
        required_keywords (List[str]): Keywords yang harus ada di hasil. Defaults to [].
        min_match (int): Minimum jumlah keywords yang harus match. Defaults to 2.
        apply_literal_boost (bool): Whether to apply literal overlap boosting. Defaults to True.
        boost_factor (float): Boost factor for literal overlap. Defaults to 0.2.
        
    Returns:
        List[Dict]: List hasil pencarian dengan rank, score, dan metadata
    """
    start_time = datetime.now()
    
    try:
        # Step 1: Optimize query if not provided
        if optimized_query is None:
            if hasattr(query_optimizer, 'optimize_query'):
                optimized_query, extracted_keywords = query_optimizer.optimize_query(raw_query, return_keywords=True)
                # Use extracted keywords if no required_keywords specified
                if not required_keywords and extracted_keywords:
                    required_keywords = extracted_keywords
            else:
                optimized_query = f"passage: {raw_query}"
        
        logger.info(f"Processing query: '{raw_query}'")
        logger.info(f"Optimized query: '{optimized_query}'")
        logger.info(f"Required keywords: {required_keywords}")
        
        # Step 2: Generate embedding and search FAISS
        embedding = get_query_embedding(optimized_query)
        index, metadatas = load_faiss_index()
        
        # Get more results if we need to filter by keywords
        multiplier = 4 if required_keywords else 1
        search_k = min(top_k * multiplier, len(metadatas))
        
        D, I = index.search(embedding, search_k)
        
        # Step 3: Process and filter results
        candidates = []
        filter_stats = {"total_candidates": 0, "keyword_filtered": 0, "final_results": 0}
        
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(metadatas):
                continue
            
            meta = metadatas[idx]
            filter_stats["total_candidates"] += 1
            
            # Apply keyword filtering if required
            if required_keywords:
                terjemah_text = meta.get("terjemah", "")
                meets_criteria, match_details = enhanced_keyword_match(
                    terjemah_text, required_keywords, min_match
                )
                
                if not meets_criteria:
                    filter_stats["keyword_filtered"] += 1
                    continue
                
                # Add match details to result
                result = {
                    "kitab": meta.get("kitab", ""),
                    "id": meta.get("id", ""),
                    "arab": meta.get("arab", ""),
                    "terjemah": meta.get("terjemah", ""),
                    "score": float(score),
                    "tags": meta.get("tags", []),
                    "keyword_match_details": match_details
                }
            else:
                result = {
                    "kitab": meta.get("kitab", ""),
                    "id": meta.get("id", ""),
                    "arab": meta.get("arab", ""),
                    "terjemah": meta.get("terjemah", ""),
                    "score": float(score),
                    "tags": meta.get("tags", [])
                }
            
            candidates.append(result)
            
            # Stop when we have enough candidates
            if len(candidates) >= top_k * 2:  # Get extra for boosting
                break
        
        filter_stats["final_results"] = len(candidates)
        
        # Step 4: Apply literal overlap boosting
        if apply_literal_boost and candidates:
            logger.info(f"Applying literal overlap boosting with factor {boost_factor}")
            candidates = boost_literal_overlap_score(candidates, raw_query, boost_factor)
        
        # Step 5: Limit to top_k results and add ranking
        final_results = candidates[:top_k]
        for i, result in enumerate(final_results):
            result["rank"] = i + 1
        
        # Step 6: Log processing statistics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Query processing completed in {processing_time:.2f}ms")
        logger.info(f"Filter stats: {filter_stats}")
        
        # Step 7: Log results for analysis
        log_query_results(raw_query, required_keywords, final_results, processing_time)
        
        return final_results
        
    except Exception as e:
        logger.error(f"Error in query_hadits_return: {e}")
        # Log the error as well
        error_time = (datetime.now() - start_time).total_seconds() * 1000
        log_query_results(raw_query, required_keywords, [], error_time)
        raise

