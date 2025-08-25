import pandas as pd
import json
import re
import unicodedata
from pathlib import Path

# Fungsi normalisasi teks Arab: hilangkan harakat, simbol non-Arab
def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    # Hapus harakat (tanda baca Arab)
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]', '', text)
    # Hapus karakter non-Arab
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    return text.strip()

# Fungsi normalisasi teks Indonesia
def normalize_indo(text: str) -> str:
    text = re.sub(r'[\[\]"\n\r]', '', text)  # hilangkan simbol dan newline
    return text.strip()

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)  # hapus simbol kecuali apostrof
    text = re.sub(r"\s+", " ", text)  # multiple space jadi satu
    return text.strip()

# Fungsi utama preprocessing
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

# Contoh pemanggilan langsung
if __name__ == "__main__":
    preprocess_csv(
        csv_path="data/raw/shahih_bukhari.csv",
        output_path="data/processed/hadits_docs.json"
    )
