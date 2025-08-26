import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pickle
import re
from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.preprocessing import normalize

# Muat environment variable
load_dotenv()
DATA_PATH = os.getenv("DATA_CLEAN_PATH", "data/processed/hadits_docs.json")
OUTPUT_PATH = "data/processed/hadits_embeddings.pkl"
KEYWORDS_PATH = "data/processed/keywords_map_grouped.json"
MODEL_NAME = "intfloat/e5-small-v2"

# ------------------------------
# ✅ Load model embedding sekali
def get_embedding_model():
    return SentenceTransformer(MODEL_NAME)

# ------------------------------
# 1. Load dokumen
def load_documents():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def load_keyword_map():
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_semantic_tags(doc, keyword_map):
    tags = set()
    teks = doc["terjemah"].lower()

    for key, variants in keyword_map.items():
        for v in variants:
            pattern = rf"(?<!\w){re.escape(v)}(?!\w)" 
            if re.search(pattern, teks):
                tags.add(v)

    return ", ".join(sorted(tags))

# 4. Ubah dokumen jadi corpus format "passage: ... Kata kunci penting: ..."
def prepare_corpus(docs, keyword_map):
    corpus = []

    for i, doc in enumerate(docs[:10]):
        print(f"[DEBUG] 📄 Doc {i+1} tags: {build_semantic_tags(doc, keyword_map)}")

    for i, doc in enumerate(docs):
        if "id" not in doc:
            raise ValueError(f"[!] Dokumen ke-{i} tidak memiliki ID.")

        base = doc["terjemah"]
        tags = build_semantic_tags(doc, keyword_map)

        if tags:
            full = f"passage: {base}. Kata kunci penting: {tags}"
            if len(tags) <= 1:
                print(f"[⚠️] Doc {doc['id']} hanya memiliki 1 tag: {tags}")
        else:
            full = f"passage: {base}"

        corpus.append(full)

    return corpus

# 5. Embedding dokumen
def embed_documents(corpus, model_name=MODEL_NAME):
    print(f"[~] Embedding {len(corpus)} dokumen...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = normalize(embeddings, axis=1)
    return embeddings

# 6. Simpan hasil embedding + dokumen asli
def save_output(embeddings, documents):
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "documents": documents
        }, f)
    print(f"[✓] Embeddings berhasil disimpan ke {OUTPUT_PATH}")

# ------------------------------
# Main: jalankan semua
if __name__ == "__main__":
    print("[~] Memuat dokumen...")
    docs = load_documents()

    print("[~] Memuat keyword map...")
    keyword_map = load_keyword_map()

    print("[~] Menyusun corpus semantik...")
    corpus = prepare_corpus(docs, keyword_map)

    print("\n[📄] Contoh Corpus:")
    for i in range(min(3, len(corpus))):
        print(f"[{i+1}] {corpus[i]}")

    print("[~] Proses embedding...")
    embeddings = embed_documents(corpus)

    save_output(embeddings, docs)
