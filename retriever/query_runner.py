import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List, Dict
from utils import query_optimizer

# --- Env & Config ---
load_dotenv()
DB_PATH = os.getenv("CHROMA_DB_PATH", "./db/hadits_index")
COLLECTION_NAME = "hadits"
MODEL_NAME = "intfloat/e5-small-v2"

# --- Init once ---
model = SentenceTransformer(MODEL_NAME)

def load_chroma_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)

def get_query_embedding(query: str):
    embedding = model.encode([query], convert_to_numpy=True)
    embedding = normalize(embedding, axis=1)
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
    Melakukan pencarian hadits dengan semantic search + keyword filtering.
    
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
    collection = load_chroma_collection()

    # Ambil lebih banyak hasil untuk di-filter nanti
    multiplier = 4 if required_keywords else 1
    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k * multiplier,
        include=["metadatas", "distances"]
    )

    filtered = []
    for meta, score in zip(results["metadatas"][0], results["distances"][0]):
        # Post-filtering berdasarkan terjemah dengan required_keywords
        if required_keywords:
            terjemah_text = meta.get("terjemah", "")
            if not keyword_match(terjemah_text, required_keywords, min_match):
                continue  # skip kalau tidak memenuhi min_match requirement
        
        filtered.append({
            "kitab": meta["kitab"],
            "id": meta.get("id"),
            "score": float(score),
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

