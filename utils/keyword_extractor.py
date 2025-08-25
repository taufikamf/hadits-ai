import pandas as pd
from collections import Counter, defaultdict
import re
from pathlib import Path
import json

CSV_DIR = Path("data/csv")
OUTPUT_PATH = Path("data/processed/keywords_map.json")
TERJEMAH_FIELD = "terjemah"
MIN_WORD_LEN = 4
TOP_N = 1000

# Kata-kata utama yang ingin dicari dan dikelompokkan
DOMAIN_TERMS = [
    "khamr", "riba", "zina", "shalat", "salat", "sholat", "puasa", "shaum",
    "zakat", "najis", "halal", "haram", "niat", "iman", "neraka", "surga", "wudhu"
]

def normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())

def tokenize(text: str):
    return normalize(text).split()

def collect_terjemah_texts(directory: Path):
    all_texts = []
    for file in directory.glob("*.csv"):
        try:
            df = pd.read_csv(file)
            if TERJEMAH_FIELD in df.columns:
                all_texts.extend(df[TERJEMAH_FIELD].dropna().tolist())
                print(f"[✓] Terbaca: {file.name} — {len(df)} baris")
            else:
                print(f"[!] Kolom '{TERJEMAH_FIELD}' tidak ditemukan di {file.name}")
        except Exception as e:
            print(f"[!] Gagal membaca {file.name}: {e}")
    return all_texts

def extract_keyword_map(texts, domain_terms):
    counter = Counter()
    for teks in texts:
        tokens = tokenize(teks)
        counter.update(tokens)

    keyword_map = defaultdict(list)
    for word, _ in counter.items():
        for domain in domain_terms:
            if word == domain:
                keyword_map[domain].append(word)

    return dict(keyword_map)

def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[✓] Hasil disimpan ke {path}")

if __name__ == "__main__":
    print("[~] Ekstraksi keyword dari semua file CSV...")
    texts = collect_terjemah_texts(CSV_DIR)
    keyword_map = extract_keyword_map(texts, DOMAIN_TERMS)
    save_json(keyword_map, OUTPUT_PATH)
