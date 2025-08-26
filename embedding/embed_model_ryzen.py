import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pickle
import re
import time
import gc
import multiprocessing as mp
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.preprocessing import normalize
import psutil

# Load environment variables
load_dotenv()
DATA_PATH = os.getenv("DATA_CLEAN_PATH", "data/processed/hadits_docs.json")
OUTPUT_PATH = "data/processed/hadits_embeddings.pkl"
KEYWORDS_PATH = "data/processed/keywords_map_grouped.json"
MODEL_NAME = "intfloat/e5-small-v2"

# ==============================
# 🚀 AMD RYZEN 9950X OPTIMIZATIONS
# ==============================

def check_system_specs():
    """Check and display system specifications"""
    print("🔍 AMD Ryzen 9950X System Check:")
    print(f"CPU cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB total")
    print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available (just for information)
    if torch.cuda.is_available():
        print("⚠️ CUDA detected but will use CPU optimization for AMD Ryzen")
    else:
        print("✅ Using CPU-only optimization (perfect for AMD Ryzen)")
    
    return psutil.cpu_count(logical=True), psutil.virtual_memory().total / (1024**3)

def get_optimal_cpu_config():
    """Get optimal configuration for AMD Ryzen 9950X"""
    cpu_count = psutil.cpu_count(logical=True)  # 32 threads on 9950X
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # AMD Ryzen 9950X specific optimizations
    if cpu_count >= 32:  # Ryzen 9950X has 32 threads
        # Use 75% of threads to leave room for system processes
        num_workers = min(24, cpu_count - 8)
        batch_size = 16  # Smaller batches for better parallelization
        torch_threads = 4  # Threads per model instance
    elif cpu_count >= 16:
        num_workers = min(12, cpu_count - 4)
        batch_size = 24
        torch_threads = 6
    else:
        num_workers = max(2, cpu_count // 2)
        batch_size = 32
        torch_threads = 8
    
    # Memory optimization for 32GB RAM
    if ram_gb >= 24:
        # Plenty of RAM - can use larger models and more parallel workers
        enable_large_batch = True
        memory_efficient = False
    else:
        enable_large_batch = False
        memory_efficient = True
    
    print(f"✅ Optimal configuration for your system:")
    print(f"   Workers: {num_workers} parallel processes")
    print(f"   Batch size: {batch_size}")
    print(f"   PyTorch threads per worker: {torch_threads}")
    print(f"   Large batch mode: {enable_large_batch}")
    
    return num_workers, batch_size, torch_threads, enable_large_batch, memory_efficient

def set_torch_optimizations(num_threads=4):
    """Set PyTorch optimizations for AMD Ryzen"""
    # Optimize for AMD CPU
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(2)
    
    # Enable optimizations
    if hasattr(torch.backends.mkldnn, 'is_available') and torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True
        print("✅ MKL-DNN optimization enabled")
    
    # Memory optimization
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)

# ==============================
# 📊 DATA PROCESSING FUNCTIONS
# ==============================

def load_documents():
    """Load hadits documents from JSON file"""
    print(f"📖 Loading documents from: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)
    print(f"✅ Loaded {len(docs):,} documents")
    return docs

def load_keyword_map():
    """Load keyword mapping for semantic tagging"""
    print(f"🗝️ Loading keyword map from: {KEYWORDS_PATH}")
    with open(KEYWORDS_PATH, "r", encoding="utf-8") as f:
        keyword_map = json.load(f)
    print(f"✅ Loaded {len(keyword_map)} keyword groups")
    return keyword_map

def build_semantic_tags_batch(doc_batch, keyword_map):
    """Build semantic tags for a batch of documents - optimized for parallel processing"""
    results = []
    
    for doc in doc_batch:
        tags = set()
        teks = doc["terjemah"].lower()
        
        for key, variants in keyword_map.items():
            for v in variants:
                # Use word boundary to avoid partial matches
                pattern = rf"(?<!\w){re.escape(v)}(?!\w)"
                if re.search(pattern, teks):
                    tags.add(v)
        
        result = ", ".join(sorted(tags)) if tags else ""
        results.append((doc, result))
    
    return results

def prepare_corpus_parallel(docs, keyword_map, num_workers=8):
    """Prepare corpus with parallel processing for AMD Ryzen"""
    print(f"🏗️ Preparing corpus with {num_workers} parallel workers...")
    
    # Split documents into chunks for parallel processing
    chunk_size = max(50, len(docs) // num_workers)
    doc_chunks = [docs[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]
    
    corpus = []
    processed_count = 0
    no_tags_count = 0
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create partial function with keyword_map
        process_func = partial(build_semantic_tags_batch, keyword_map=keyword_map)
        
        # Submit all chunks
        futures = [executor.submit(process_func, chunk) for chunk in doc_chunks]
        
        # Collect results with progress bar
        for future in tqdm(futures, desc="Processing document chunks"):
            chunk_results = future.result()
            
            for doc, tags in chunk_results:
                if "id" not in doc:
                    raise ValueError(f"Document missing ID field")
                
                base_text = doc["terjemah"]
                
                if tags:
                    # Enhanced format with semantic tags
                    full_text = f"passage: {base_text}. Kata kunci penting: {tags}"
                    processed_count += 1
                else:
                    # Fallback for documents without tags
                    full_text = f"passage: {base_text}"
                    no_tags_count += 1
                
                corpus.append(full_text)
    
    print(f"✅ Corpus prepared: {processed_count:,} docs with tags, {no_tags_count:,} without tags")
    return corpus

# ==============================
# 🤖 EMBEDDING FUNCTIONS
# ==============================

def create_model_instance():
    """Create a model instance for worker processes"""
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    return model

def encode_batch_worker(args):
    """Worker function for encoding batches"""
    batch_texts, batch_idx, torch_threads = args
    
    # Set thread limit for this worker
    torch.set_num_threads(torch_threads)
    
    # Create model instance in worker process
    model = SentenceTransformer(MODEL_NAME, device='cpu')
    
    # Encode batch
    embeddings = model.encode(
        batch_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    return batch_idx, embeddings

def embed_documents_ryzen_optimized(corpus, num_workers=8, batch_size=32, torch_threads=4):
    """Create embeddings optimized for AMD Ryzen 9950X"""
    print(f"\n🚀 Starting AMD Ryzen 9950X optimized embedding...")
    print(f"📊 Documents: {len(corpus):,}")
    print(f"⚙️ Workers: {num_workers}, Batch size: {batch_size}")
    
    # Split corpus into batches
    batches = []
    for i in range(0, len(corpus), batch_size):
        batch_texts = corpus[i:i + batch_size]
        batches.append((batch_texts, i // batch_size, torch_threads))
    
    print(f"📦 Created {len(batches)} batches")
    
    # Performance estimation
    print("⏱️ Estimating performance...")
    test_batch = batches[0]
    start_test = time.time()
    
    # Test single batch performance
    _, test_embedding = encode_batch_worker(test_batch)
    test_time = time.time() - start_test
    
    estimated_total = (test_time * len(batches)) / num_workers
    print(f"📈 Estimated time: {estimated_total/60:.1f} minutes")
    
    # Process all batches in parallel
    print("🔄 Processing batches in parallel...")
    start_time = time.time()
    
    all_embeddings = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        futures = [executor.submit(encode_batch_worker, batch) for batch in batches]
        
        # Collect results with progress bar
        for future in tqdm(futures, desc="Encoding batches"):
            batch_idx, batch_embeddings = future.result()
            all_embeddings[batch_idx] = batch_embeddings
    
    # Concatenate all embeddings in order
    print("🔗 Concatenating embeddings...")
    final_embeddings = []
    for i in sorted(all_embeddings.keys()):
        final_embeddings.append(all_embeddings[i])
    
    # Convert to single numpy array
    import numpy as np
    embeddings = np.vstack(final_embeddings)
    
    # Final normalization
    embeddings = normalize(embeddings, axis=1)
    
    total_time = time.time() - start_time
    speed = len(corpus) / total_time
    
    print(f"✅ Embedding completed!")
    print(f"⏱️ Total time: {total_time/60:.1f} minutes")
    print(f"🚄 Speed: {speed:.1f} documents/second")
    print(f"📐 Shape: {embeddings.shape}")
    
    return embeddings

def save_ryzen_output(embeddings, documents):
    """Save embeddings with metadata for Ryzen system"""
    print(f"💾 Saving results to: {OUTPUT_PATH}")
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # System info
    cpu_count = psutil.cpu_count(logical=True)
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    save_data = {
        "embeddings": embeddings,
        "documents": documents,
        "metadata": {
            "model_name": MODEL_NAME,
            "embedding_dim": embeddings.shape[1],
            "total_documents": len(documents),
            "creation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "cpu_model": "AMD Ryzen 9 9950X",
                "cpu_threads": cpu_count,
                "ram_gb": ram_gb,
                "optimization": "AMD Ryzen Multi-Core"
            }
        }
    }
    
    start_time = time.time()
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    save_time = time.time() - start_time
    file_size = os.path.getsize(OUTPUT_PATH) / (1024**2)
    
    print(f"✅ Saved in {save_time:.2f} seconds")
    print(f"📁 File size: {file_size:.1f} MB")

# ==============================
# 🎯 MAIN EXECUTION
# ==============================

def main():
    """Main execution optimized for AMD Ryzen 9950X"""
    print("🚀 AMD RYZEN 9950X HADITS EMBEDDING GENERATOR")
    print("=" * 70)
    
    try:
        # System analysis
        cpu_count, ram_gb = check_system_specs()
        print()
        
        # Get optimal configuration
        num_workers, batch_size, torch_threads, large_batch, memory_efficient = get_optimal_cpu_config()
        print()
        
        # Set PyTorch optimizations
        set_torch_optimizations(torch_threads)
        print()
        
        # Load data
        docs = load_documents()
        keyword_map = load_keyword_map()
        print()
        
        # Prepare corpus with parallel processing
        corpus = prepare_corpus_parallel(docs, keyword_map, num_workers)
        print()
        
        # Show samples
        print("📄 Sample corpus entries:")
        for i in range(min(3, len(corpus))):
            sample = corpus[i]
            print(f"[{i+1}] {sample[:120]}{'...' if len(sample) > 120 else ''}")
        print()
        
        # Generate embeddings
        embeddings = embed_documents_ryzen_optimized(
            corpus, 
            num_workers=num_workers,
            batch_size=batch_size,
            torch_threads=torch_threads
        )
        print()
        
        # Save results
        save_ryzen_output(embeddings, docs)
        
        print("\n" + "=" * 70)
        print("🎉 AMD RYZEN OPTIMIZATION COMPLETE!")
        print("=" * 70)
        
        # Performance summary
        print("\n💡 Performance Tips for next runs:")
        print("- Close unnecessary applications to free up CPU cores")
        print("- Run during low system activity for best performance")
        print("- Monitor CPU usage to ensure all cores are utilized")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup
        gc.collect()
        raise

if __name__ == "__main__":
    # Set multiprocessing start method for Windows
    if sys.platform.startswith('win'):
        mp.set_start_method('spawn', force=True)
    
    main()