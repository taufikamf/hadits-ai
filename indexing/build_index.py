import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from embedding.embed_model import build_semantic_tags, load_keyword_map

# Muat environment
load_dotenv()
DB_PATH = os.getenv("CHROMA_DB_PATH", "./db/hadits_index")
EMBEDDING_PATH = "data/processed/hadits_embeddings.pkl"

def load_embeddings(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["documents"]

def index_to_chroma(embeddings, documents, collection_name="hadits", batch_size=5000):
    # Init Chroma
    client = chromadb.PersistentClient(path=DB_PATH)

    # Hapus koleksi lama jika ada
    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(name=collection_name)

    # Buat koleksi baru
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # Persiapan data
    keyword_map = load_keyword_map()
    total_docs = len(documents)
    
    print(f"[~] Memproses {total_docs} dokumen dalam batch {batch_size}...")
    
    # Process documents in batches
    for start_idx in tqdm(range(0, total_docs, batch_size), desc="Adding batches to ChromaDB"):
        end_idx = min(start_idx + batch_size, total_docs)
        
        # Prepare batch data
        batch_ids = []
        batch_texts = []
        batch_metadatas = []
        batch_embeddings = []
        
        for i in range(start_idx, end_idx):
            doc = documents[i]
            tags = build_semantic_tags(doc, keyword_map)
            
            if "id" not in doc:
                raise ValueError(f"[!] Dokumen ke-{i} tidak memiliki ID.")

            batch_ids.append(str(doc["id"]))
            batch_texts.append(doc["terjemah"])
            batch_embeddings.append(embeddings[i].tolist())

            batch_metadatas.append({
                "id": doc["id"],  # ID eksplisit
                "kitab": doc["kitab"],
                "arab": doc.get("arab_bersih", ""),
                "arab_asli": doc.get("arab_asli", doc.get("arab_bersih", "")),
                "terjemah": doc["terjemah"],
                "tags": tags
            })
        
        # Add batch to collection
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas
        )
        
        print(f"[✓] Batch {start_idx//batch_size + 1}: Added {len(batch_ids)} documents")

    print(f"[✓] Index selesai & disimpan di: {DB_PATH}")
    print(f"[✓] Total {total_docs} dokumen berhasil diindeks")

if __name__ == "__main__":
    embeddings, documents = load_embeddings(EMBEDDING_PATH)
    index_to_chroma(embeddings, documents)
