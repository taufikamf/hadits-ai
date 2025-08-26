import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import faiss
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import List, Dict
from utils import query_optimizer
import pickle

# --- Env & Config ---
# load_dotenv()
# DB_PATH = os.getenv("CHROMA_DB_PATH", "./db/hadits_index")
# COLLECTION_NAME = "hadits"
# MODEL_NAME = "intfloat/e5-small-v2"
load_dotenv()
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./db/hadits_faiss.index")
FAISS_METADATA_PATH = os.getenv("FAISS_METADATA_PATH", "./db/hadits_metadata.pkl")
MODEL_NAME = "intfloat/e5-small-v2"

# --- Init once ---
model = SentenceTransformer(MODEL_NAME)

def load_chroma_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(name=COLLECTION_NAME)

def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(FAISS_METADATA_PATH, "rb") as f:
        metadatas = pickle.load(f)
    return index, metadatas

# def get_query_embedding(query: str):
#     embedding = model.encode([query], convert_to_numpy=True)
#     embedding = normalize(embedding, axis=1)
#     return embedding

def get_query_embedding(query: str):
    embedding = model.encode([query], convert_to_numpy=True)
    embedding = normalize(embedding, axis=1)
    return embedding.astype("float32")

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
    index, metadatas = load_faiss_index()
    # collection = load_chroma_collection()

    # Ambil lebih banyak hasil untuk di-filter nanti
    # multiplier = 4 if required_keywords else 1
    # results = collection.query(
    #     query_embeddings=embedding,
    #     n_results=top_k * multiplier,
    #     include=["metadatas", "distances"]
    # )
    multiplier = 4 if required_keywords else 1
    D, I = index.search(embedding, top_k * multiplier)

    filtered = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(metadatas):
            continue
        meta = metadatas[idx]
        if required_keywords:
            terjemah_text = meta.get("terjemah", "")
            if not keyword_match(terjemah_text, required_keywords, min_match):
                continue
        filtered.append({
            "kitab": meta["kitab"],
            "id": meta.get("id"),
            "arab": meta.get("arab"),
            "terjemah": meta.get("terjemah"),
            "score": float(score),
            "tags": meta.get("tags", [])
        })
        if len(filtered) >= top_k:
            break

    # Tambahkan ranking
    return [
        {**r, "rank": i + 1}
        for i, r in enumerate(filtered)
    ]

