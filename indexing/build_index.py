import os
import pickle
import sys
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
import faiss
from tqdm import tqdm
from embedding.embed_model import build_semantic_tags, load_keyword_map

# Muat environment
load_dotenv()
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./db/hadits_index")
METADATA_PATH = os.getenv("METADATA_PATH", "./db/hadits_metadata.json")
EMBEDDING_PATH = "data/processed/hadits_embeddings.pkl"

def load_embeddings(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["documents"]

def index_to_faiss(embeddings, documents, batch_size=5000):
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    
    # Convert embeddings to numpy array if not already
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings.astype('float32')
    faiss.normalize_L2(embeddings)
    
    # Build FAISS index - using IndexFlatIP for exact cosine similarity
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity with normalized vectors
    
    # Persiapan data
    keyword_map = load_keyword_map()
    total_docs = len(documents)
    
    print(f"[~] Memproses {total_docs} dokumen dalam batch {batch_size}...")
    print(f"[~] Embedding dimension: {dimension}")
    
    # Prepare metadata storage
    metadata_list = []
    
    # Process documents in batches
    for start_idx in tqdm(range(0, total_docs, batch_size), desc="Adding batches to FAISS"):
        end_idx = min(start_idx + batch_size, total_docs)
        
        # Prepare batch data
        batch_embeddings = embeddings[start_idx:end_idx]
        
        for i in range(start_idx, end_idx):
            doc = documents[i]
            tags = build_semantic_tags(doc, keyword_map)
            
            if "id" not in doc:
                raise ValueError(f"[!] Dokumen ke-{i} tidak memiliki ID.")

            # Store metadata separately since FAISS doesn't store metadata
            metadata_list.append({
                "index": i,  # FAISS index position
                "id": doc["id"],  # Original document ID
                "kitab": doc["kitab"],
                "arab": doc.get("arab_bersih", ""),
                "arab_asli": doc.get("arab_asli", doc.get("arab_bersih", "")),
                "terjemah": doc["terjemah"],
                "tags": tags
            })
        
        # Add batch to FAISS index
        index.add(batch_embeddings)
        
        print(f"[✓] Batch {start_idx//batch_size + 1}: Added {len(batch_embeddings)} documents")

    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)
    
    # Save metadata as JSON
    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    
    print(f"[✓] FAISS index selesai & disimpan di: {INDEX_PATH}")
    print(f"[✓] Metadata disimpan di: {METADATA_PATH}")
    print(f"[✓] Total {total_docs} dokumen berhasil diindeks")

if __name__ == "__main__":
    embeddings, documents = load_embeddings(EMBEDDING_PATH)
    index_to_faiss(embeddings, documents)
