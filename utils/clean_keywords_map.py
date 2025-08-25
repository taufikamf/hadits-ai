import json
import re
from collections import defaultdict

# Konfigurasi file
INPUT_PATH = "data/processed/keywords_map_grouped.json"
OUTPUT_PATH = "data/processed/keywords_map_cleaned.json"
REMOVED_PATH = "data/processed/_removed_keywords_map.json"

# ✅ Frasa/frasa noise (salah konteks, sanad, kata sambung)
NOISE_PHRASES = [
    "telah", "menceritakan", "berkata", "mengabarkan",
    "dari", "kepada", "kami", "aku", "ia", "wahai", "ya",
    "radliallahu anhu", "nabi saw", "rasulullah", "saw",
    "bin", "binti", "ibnu", "abu", "ummu", "ayah", "anak", "suami", "istri",
    "hadis ini", "orang yang", "orang orang", "laki laki", "apa yang"
]

def is_noise(phrase: str) -> bool:
    phrase = phrase.strip().lower()
    # Buang jika hanya 1 kata dan sangat umum
    if len(phrase.split()) == 1 and phrase in NOISE_PHRASES:
        return True
    # Buang jika mengandung noise phrases di awal
    if any(phrase.startswith(n + " ") for n in NOISE_PHRASES):
        return True
    # Buang jika terlalu pendek atau kosong
    if len(phrase) < 4:
        return True
    return False

def clean_keywords_map(data):
    cleaned = defaultdict(list)
    removed = defaultdict(list)

    for key, variants in data.items():
        valid_variants = []
        for phrase in variants:
            if not is_noise(phrase):
                valid_variants.append(phrase)
            else:
                removed[key].append(phrase)

        # Tambahkan key hanya jika masih ada variannya
        if valid_variants:
            cleaned[key] = valid_variants

    return dict(cleaned), dict(removed)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned, removed = clean_keywords_map(raw_data)

    save_json(OUTPUT_PATH, cleaned)
    save_json(REMOVED_PATH, removed)

    print(f"[✓] keywords_map_cleaned.json disimpan dengan {sum(len(v) for v in cleaned.values())} keyword valid")
    print(f"[!] {sum(len(v) for v in removed.values())} keyword dibuang (lihat {REMOVED_PATH})")
