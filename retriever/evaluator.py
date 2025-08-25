import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from utils import query_optimizer
from retriever import query_runner

load_dotenv()

OUTPUT_PATH = "retriever/retrieval_results.json"
BATCH_QUERIES = [
    "apa hukum riba?",
    "apa itu zakat fitrah?", 
    "bagaimana cara shalat malam?",
    "apa hukum minuman keras?",
    "berikan hadis tentang mati syahid",
    "berikan hadis tentang shalat jum'at",
    "berikan hadis tentang perang",
    "berikan hadis tentang sodaqoh",
    "berikan hadits tentang shalat dhuhur"
]

print("[DEBUG] Menggunakan query_optimizer dari:", query_optimizer.__file__)

def run_batch_queries():
    """
    Menjalankan batch queries dengan menggunakan query_optimizer dan query_runner yang telah direfactor.
    Menggunakan parameter baru: required_keywords dan min_match untuk post-filtering.
    """
    all_results = []

    for q in BATCH_QUERIES:
        print(f"\n[INFO] Processing query: '{q}'")
        
        # Get optimized query and required keywords from query_optimizer
        optimized, required_keywords = query_optimizer.optimize_query(q, return_keywords=True)
        
        print(f"[INFO] Found {len(required_keywords)} keywords: {required_keywords}")
        
        # Query dengan parameter baru yang sesuai signature yang telah direfactor
        results = query_runner.query_hadits_return(
            raw_query=q,
            optimized_query=optimized, 
            top_k=5,
            required_keywords=required_keywords,
            min_match=2  # Default: minimal 2 keywords harus match dalam terjemah
        )
        
        print(f"[INFO] Retrieved {len(results)} results after filtering")

        all_results.append({
            "original_query": q,
            "optimized_query": optimized,
            "required_keywords": required_keywords,
            "min_match_used": 2,
            "total_results": len(results),
            "results": results
        })

    # Save results to JSON file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Batch hasil disimpan di: {OUTPUT_PATH}")
    print(f"[✓] Total queries processed: {len(BATCH_QUERIES)}")

if __name__ == "__main__":
    run_batch_queries()
