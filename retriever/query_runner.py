import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import faiss
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List, Dict
from utils import query_optimizer

# --- Env & Config ---
load_dotenv()
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./db/hadits_index")
METADATA_PATH = os.getenv("METADATA_PATH", "./db/hadits_metadata.json")
MODEL_NAME = "intfloat/e5-small-v2"

# --- Init once ---
model = SentenceTransformer(MODEL_NAME)

# Global variables to cache loaded data
_faiss_index = None
_metadata = None

def load_faiss_index_and_metadata():
    """Load FAISS index and metadata once and cache them"""
    global _faiss_index, _metadata
    
    if _faiss_index is None or _metadata is None:
        # Load FAISS index
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index tidak ditemukan di: {INDEX_PATH}")
        
        _faiss_index = faiss.read_index(INDEX_PATH)
        
        # Load metadata
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata tidak ditemukan di: {METADATA_PATH}")
        
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            _metadata = json.load(f)
    
    return _faiss_index, _metadata

def get_query_embedding(query: str):
    embedding = model.encode([query], convert_to_numpy=True)
    embedding = normalize(embedding, axis=1)
    # Convert to float32 and normalize for FAISS cosine similarity
    embedding = embedding.astype('float32')
    faiss.normalize_L2(embedding)
    return embedding

def contains_all_keywords(text: str, keywords: list[str]) -> bool:
    text = text.lower()
    return all(kw.lower() in text for kw in keywords)

def keyword_match(text: str, keywords: List[str], min_match: int = 2) -> bool:
    """
    Menghitung apakah teks mengandung minimal min_match keywords.
    
    Args:
        text (str): Teks yang akan dicek
        keywords (List[str]): Daftar keywords untuk dicari
        min_match (int): Jumlah minimum keyword yang harus match
        
    Returns:
        bool: True jika text mengandung minimal min_match keywords
    """
    match_count = sum(1 for kw in keywords if kw.lower() in text.lower())
    return match_count >= min_match

def query_hadits_return(
    raw_query: str, 
    optimized_query: str = None, 
    top_k: int = 5, 
    required_keywords: List[str] = [], 
    min_match: int = 2
) -> List[Dict]:
    """
    Melakukan pencarian hadits dengan semantic search + keyword filtering menggunakan FAISS.
    
    Args:
        raw_query (str): Query asli dari user
        optimized_query (str, optional): Query yang sudah dioptimasi. Defaults to None.
        top_k (int): Jumlah hasil yang diinginkan. Defaults to 5.
        required_keywords (List[str]): Keywords yang harus ada di hasil. Defaults to [].
        min_match (int): Minimum jumlah keywords yang harus match. Defaults to 2.
        
    Returns:
        List[Dict]: List hasil pencarian dengan rank, score, dan metadata
    """
    if optimized_query is None:
        optimized_query = query_optimizer.optimize_query(raw_query)

    embedding = get_query_embedding(optimized_query)
    index, metadata = load_faiss_index_and_metadata()

    # Ambil lebih banyak hasil untuk di-filter nanti
    multiplier = 4 if required_keywords else 1
    search_k = min(top_k * multiplier, index.ntotal)  # Don't search for more than available
    
    # Search using FAISS - returns scores (cosine similarity) and indices
    scores, indices = index.search(embedding, search_k)
    
    # Convert to lists for easier processing
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    filtered = []
    for idx, score in zip(indices, scores):
        if idx == -1:  # FAISS returns -1 for empty results
            continue
            
        # Get metadata for this document
        meta = metadata[idx]
        
        # Post-filtering berdasarkan terjemah dengan required_keywords
        if required_keywords:
            terjemah_text = meta.get("terjemah", "")
            if not keyword_match(terjemah_text, required_keywords, min_match):
                continue  # skip kalau tidak memenuhi min_match requirement
        
        filtered.append({
            "kitab": meta["kitab"],
            "id": meta.get("id"),
            "score": float(score),  # FAISS returns cosine similarity (higher is better)
            "arab": meta.get("arab_asli", ""),
            "terjemah": meta.get("terjemah", "")
        })
        
        # Stop ketika sudah dapat cukup hasil
        if len(filtered) == top_k:
            break

    # Tambahkan ranking
    return [
        {**r, "rank": i + 1}
        for i, r in enumerate(filtered)
    ]

