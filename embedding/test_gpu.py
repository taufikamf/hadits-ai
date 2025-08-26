import torch
from sentence_transformers import SentenceTransformer
import time
import psutil
import platform

def check_system_info():
    print("=== SYSTEM INFORMATION ===")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"CPU: {platform.processor()}")
    print(f"Python version: {platform.python_version()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()

def check_gpu_setup():
    print("=== GPU SETUP CHECK ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  - Total Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
            print(f"  - Multiprocessors: {props.multi_processor_count}")
        
        # Test GPU memory
        print(f"Current GPU memory usage:")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.memory_allocated(i) / 1024**2:.1f} MB allocated, "
                  f"{torch.cuda.memory_reserved(i) / 1024**2:.1f} MB reserved")
    else:
        print("❌ CUDA not available!")
        print("Suggestions:")
        print("1. Install CUDA-enabled PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("2. Make sure NVIDIA drivers are up to date")
        print("3. Restart after installation")
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available: True")
    
    print()

def test_model_performance():
    print("=== MODEL PERFORMANCE TEST ===")
    
    model_name = "intfloat/e5-small-v2"
    print(f"Testing model: {model_name}")
    
    # Test CPU performance
    print("\n--- CPU Test ---")
    start_time = time.time()
    model_cpu = SentenceTransformer(model_name, device='cpu')
    cpu_load_time = time.time() - start_time
    print(f"CPU model loaded in: {cpu_load_time:.2f} seconds")
    
    test_texts = ["This is a test sentence for performance benchmarking."] * 50
    start_time = time.time()
    embeddings_cpu = model_cpu.encode(test_texts, convert_to_numpy=True, batch_size=8)
    cpu_encode_time = time.time() - start_time
    print(f"CPU encoding {len(test_texts)} texts: {cpu_encode_time:.2f} seconds")
    print(f"CPU speed: {len(test_texts)/cpu_encode_time:.1f} texts/second")
    
    # Test GPU performance if available
    if torch.cuda.is_available():
        print("\n--- GPU Test ---")
        start_time = time.time()
        model_gpu = SentenceTransformer(model_name, device='cuda')
        gpu_load_time = time.time() - start_time
        print(f"GPU model loaded in: {gpu_load_time:.2f} seconds")
        print(f"Model device: {model_gpu.device}")
        
        start_time = time.time()
        embeddings_gpu = model_gpu.encode(test_texts, convert_to_numpy=True, batch_size=32)
        gpu_encode_time = time.time() - start_time
        print(f"GPU encoding {len(test_texts)} texts: {gpu_encode_time:.2f} seconds")
        print(f"GPU speed: {len(test_texts)/gpu_encode_time:.1f} texts/second")
        
        if gpu_encode_time > 0:
            speedup = cpu_encode_time / gpu_encode_time
            print(f"GPU speedup: {speedup:.1f}x faster than CPU")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        print("GPU cache cleared")
    
    print(f"Embedding shape: {embeddings_cpu.shape}")
    print()

def optimal_batch_size_test():
    if not torch.cuda.is_available():
        print("Skipping batch size test - CUDA not available")
        return
        
    print("=== OPTIMAL BATCH SIZE TEST ===")
    model = SentenceTransformer("intfloat/e5-small-v2", device='cuda')
    test_texts = ["Performance test sentence."] * 100
    
    batch_sizes = [4, 8, 16, 32, 64, 128]
    results = {}
    
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            start_time = time.time()
            embeddings = model.encode(test_texts, batch_size=batch_size, convert_to_numpy=True)
            encode_time = time.time() - start_time
            
            speed = len(test_texts) / encode_time
            results[batch_size] = speed
            print(f"Batch size {batch_size:3d}: {speed:6.1f} texts/second ({encode_time:.2f}s)")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size:3d}: Out of memory")
                break
            else:
                raise e
    
    if results:
        optimal_batch = max(results, key=results.get)
        print(f"\nOptimal batch size: {optimal_batch} ({results[optimal_batch]:.1f} texts/second)")
    
    torch.cuda.empty_cache()

def main():
    print("🚀 GPU PERFORMANCE DIAGNOSTIC TOOL")
    print("=" * 50)
    
    check_system_info()
    check_gpu_setup()
    test_model_performance()
    optimal_batch_size_test()
    
    print("=" * 50)
    print("✅ Diagnostic complete!")
    
    if not torch.cuda.is_available():
        print("\n🔧 RECOMMENDED ACTIONS:")
        print("1. Install CUDA-enabled PyTorch:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("2. Restart your terminal/IDE")
        print("3. Run this script again to verify")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
