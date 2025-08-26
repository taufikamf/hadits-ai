from preprocessing.normalize import normalize_text
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import re
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load keyword map
KEYWORD_PATH = "data/processed/keywords_map_grouped.json"
with open(KEYWORD_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
    KEYWORDS_MAP = data["keywords"]  # Akses bagian keywords dari JSON

# Buat reverse lookup: variant -> main_key
REVERSE_MAP = {}
for main_key, variants in KEYWORDS_MAP.items():
    for variant in variants:
        REVERSE_MAP[variant] = main_key

# 🔧 Prioritaskan frasa panjang ke pendek untuk matching yang lebih akurat
SORTED_PHRASES = sorted(REVERSE_MAP.keys(), key=lambda x: (-len(x), x))

# Rule tambahan: ekspansi semantik untuk kata 'hukum'
HUKUM_SEMANTIC_RULES = [
    "haram", "halal", "mengharamkan", "menghalalkan", 
    "dilarang", "dibolehkan", "diperbolehkan"
]

def optimize_query(query: str, return_keywords=False):
    # Normalisasi query
    norm = normalize_text(query.lower())
    original_query = query.strip()
    
    matched_phrases = set()
    matched_main_keys = set()
    
    # Cari keyword matches dengan whole phrase matching
    for phrase in SORTED_PHRASES:
        # Whole word boundary untuk memastikan match yang tepat
        pattern = rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, norm):
            matched_phrases.add(phrase)
            main_key = REVERSE_MAP[phrase]
            matched_main_keys.add(main_key)
    
    # 🚨 Rule Khusus: 'hukum' → tambahkan semantik larangan/ijin
    if re.search(r'\bhukum\b', norm):
        matched_phrases.update(HUKUM_SEMANTIC_RULES)
        logger.info(f"Query contains 'hukum', added semantic rules: {HUKUM_SEMANTIC_RULES}")
    
    # Untuk setiap main key yang cocok, tambahkan juga keyword utama main key nya
    for main_key in matched_main_keys:
        matched_phrases.add(main_key)
    
    # Debug logging jika tidak ada keyword ditemukan
    if not matched_phrases:
        logger.info(f"No keywords found for query: '{original_query}'")
    else:
        logger.info(f"Found {len(matched_phrases)} keywords for query: '{original_query}'")
    
    # Format optimized query
    base = f"passage: {original_query}"
    if matched_phrases:
        # Sort keywords untuk output yang konsisten
        sorted_keywords = sorted(matched_phrases)
        enriched = base + ". Kata kunci penting: " + " ".join(sorted_keywords)
    else:
        enriched = base
    
    # Return sesuai parameter
    if return_keywords:
        return enriched, sorted(matched_phrases) if matched_phrases else []
    return enriched
