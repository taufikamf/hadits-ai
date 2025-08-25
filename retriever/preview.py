# preview.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from retriever import query_runner
from utils import query_optimizer
from dotenv import load_dotenv
import json

# Muat env
load_dotenv()

# Prompt untuk user
def interactive_preview():
    while True:
        user_input = input("\n🔎 Masukkan pertanyaan Anda ('exit' untuk keluar): ")
        if user_input.lower() in ["exit", "quit"]:
            break

        optimized = query_optimizer.optimize_query(user_input)
        results = query_runner.query_hadits_return(user_input, optimized, top_k=5)

        print("="*50)
        print(f"📥 Query: {user_input}")
        print(f"📈 Optimized: {optimized}")
        print("📊 Hasil:")
        print("="*50)

        for i, doc in enumerate(results):
            print(f"[{i+1}] ({doc['kitab']}) — Score: {doc['score']:.4f}")
            print(doc.get("arab", ""))
            print(doc["terjemah"])
            print("-" * 50)

# Menyimpan ke file untuk batch / review
def save_preview_to_file(query: str, output_path="preview_results.json"):
    optimized = query_optimizer.optimize_query(query)
    results = query_runner.query_hadits_return(query, optimized, top_k=5)

    output = {
        "query": query,
        "optimized_query": optimized,
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[✓] Hasil disimpan ke: {output_path}")

if __name__ == "__main__":
    mode = input("Pilih mode (1 = interactive, 2 = save to file): ").strip()
    if mode == "1":
        interactive_preview()
    else:
        q = input("Masukkan query yang ingin disimpan: ")
        save_preview_to_file(q)
