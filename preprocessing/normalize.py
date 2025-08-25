from typing import List
import pandas as pd
import json
import re
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    return text.strip()

def normalize_indo(text: str) -> str:
    text = re.sub(r'[\[\]"\n\r]', '', text)
    return text.strip()

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess_csv(csv_path: str, output_path: str):
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        doc = {
            "id": str(row["id"]),
            "kitab": row["kitab"],
            "arab_asli": row["arab"],
            "arab_bersih": normalize_arabic(row["arab"]),
            "terjemah": normalize_indo(row["terjemah"]),
        }
        documents.append(doc)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"[✓] Preprocessed {len(documents)} documents saved to {output_path}")

if __name__ == "__main__":
    # Daftar semua file CSV yang akan diproses
    csv_files = [
        "shahih_bukhari.csv",
        "shahih_muslim.csv",
        "sunan_abu_daud.csv",
        "sunan_tirmidzi.csv",
        "sunan_nasai.csv",
        "sunan_ibnu_majah.csv"
    ]
    
    # Inisialisasi list untuk menyimpan semua dokumen
    all_documents = []
    
    # Proses setiap file CSV
    for csv_file in csv_files:
        csv_path = f"data/csv/{csv_file}"
        print(f"[~] Memproses {csv_path}...")
        
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                doc = {
                    "id": f"{row['kitab']}_{row['id']}",
                    "kitab": row["kitab"],
                    "arab_asli": row["arab"],
                    "arab_bersih": normalize_arabic(row["arab"]),
                    "terjemah": normalize_indo(row["terjemah"]),
                }
                all_documents.append(doc)
            print(f"[✓] Berhasil memproses {len(df)} baris dari {csv_file}")
        except Exception as e:
            print(f"[!] Error saat memproses {csv_file}: {e}")
    
    # Simpan semua dokumen ke file JSON
    output_path = "data/processed/hadits_docs.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
    
    print(f"[✓] Preprocessed {len(all_documents)} documents saved to {output_path}")

from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer