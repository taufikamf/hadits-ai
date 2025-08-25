import json
from pathlib import Path
from collections import defaultdict

SOURCE_PATH = Path("data/processed/keywords_map.json")
TARGET_PATH = Path("data/processed/keywords_map_grouped.json")

# Definisi grup sinonim semantik (ejaan berbeda, makna sama)
GROUP_MAP = {
    "shalat": ["shalat", "salat", "sholat"],
    "puasa": ["puasa", "shaum"],
    # Tambahkan lainnya jika perlu
}

def load_original(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def merge_groups(original, group_map):
    new_map = defaultdict(set)

    # 1. Tambahkan entry yang termasuk grup
    for root_term, variants in group_map.items():
        for v in variants:
            if v in original:
                new_map[root_term].update(original[v])

    # 2. Tambahkan entry yang tidak termasuk grup
    for term, values in original.items():
        if all(term not in g for g in group_map.values()):
            new_map[term].update(values)

    # 3. Konversi ke list dan urutkan
    return {k: sorted(list(v)) for k, v in new_map.items()}

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[✓] Hasil grup tersimpan ke: {path}")

if __name__ == "__main__":
    print("[~] Membaca keywords_map.json...")
    original = load_original(SOURCE_PATH)

    print("[~] Menggabungkan sinonim ejaan...")
    grouped = merge_groups(original, GROUP_MAP)

    save(TARGET_PATH, grouped)
