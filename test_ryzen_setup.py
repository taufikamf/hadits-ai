#!/usr/bin/env python3
"""
AMD Ryzen 9950X Setup Test
Quick test to verify your system is properly configured for optimal performance
"""

import torch
import psutil
import time
import sys
import os
from sentence_transformers import SentenceTransformer

def print_header():
    """Print test header"""
    print("🧪 AMD RYZEN 9950X SETUP TEST")
    print("=" * 50)

def test_system_specs():
    """Test system specifications"""
    print("🔍 System Specifications:")
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=False)
    thread_count = psutil.cpu_count(logical=True)
    print(f"  CPU cores: {cpu_count} physical, {thread_count} logical")
    
    # RAM info
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    ram_available = ram.available / (1024**3)
    print(f"  RAM: {ram_gb:.1f}GB total, {ram_available:.1f}GB available")
    
    # Python info
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  Python: {python_version}")
    
    # PyTorch info
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CPU threads: {torch.get_num_threads()}")
    
    # Check MKL-DNN
    if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
        print("  ✅ MKL-DNN optimization: Available")
    else:
        print("  ⚠️ MKL-DNN optimization: Not available")
    
    print()
    return thread_count >= 16 and ram_gb >= 24

def test_pytorch_performance():
    """Test PyTorch performance"""
    print("⚡ PyTorch Performance Test:")
    
    # Test tensor operations
    print("  Testing tensor operations...")
    start = time.time()
    x = torch.randn(1000, 1000)
    y = torch.randn(1000, 1000)
    z = torch.matmul(x, y)
    tensor_time = time.time() - start
    print(f"  Matrix multiplication (1000x1000): {tensor_time:.3f}s")
    
    # Test CPU utilization
    print("  Testing CPU thread utilization...")
    start = time.time()
    for _ in range(5):
        x = torch.randn(2000, 2000)
        y = torch.randn(2000, 2000)
        z = torch.matmul(x, y)
    cpu_test_time = time.time() - start
    print(f"  Multi-core test (5 iterations): {cpu_test_time:.3f}s")
    
    print()
    return tensor_time < 1.0  # Should be fast on Ryzen 9950X

def test_model_loading():
    """Test sentence transformer model loading"""
    print("🤖 Model Loading Test:")
    
    try:
        print("  Loading intfloat/e5-small-v2...")
        start = time.time()
        model = SentenceTransformer("intfloat/e5-small-v2", device="cpu")
        load_time = time.time() - start
        print(f"  ✅ Model loaded in {load_time:.2f} seconds")
        
        # Test device
        print(f"  Model device: {model.device}")
        
        print()
        return model, load_time < 30  # Should load within 30 seconds
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        print()
        return None, False

def test_encoding_speed(model):
    """Test encoding speed"""
    print("📝 Encoding Speed Test:")
    
    if model is None:
        print("  ❌ Skipped - model not loaded")
        print()
        return False
    
    try:
        # Small test
        test_texts = [
            "This is a test sentence for performance evaluation.",
            "Testing the embedding speed on AMD Ryzen processor.",
            "Optimized for multi-core CPU performance."
        ] * 50  # 150 sentences total
        
        print(f"  Testing with {len(test_texts)} sentences...")
        start = time.time()
        embeddings = model.encode(test_texts, show_progress_bar=False, batch_size=32)
        encode_time = time.time() - start
        
        speed = len(test_texts) / encode_time
        print(f"  ✅ Speed: {speed:.1f} sentences/second")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Total time: {encode_time:.2f} seconds")
        
        # Performance evaluation
        if speed >= 50:
            print("  🚀 Excellent performance!")
        elif speed >= 30:
            print("  ✅ Good performance")
        elif speed >= 15:
            print("  ⚠️ Moderate performance - check system optimization")
        else:
            print("  ❌ Poor performance - system may need optimization")
        
        print()
        return speed >= 30
        
    except Exception as e:
        print(f"  ❌ Encoding test failed: {e}")
        print()
        return False

def test_multiprocessing():
    """Test multiprocessing capability"""
    print("🔄 Multiprocessing Test:")
    
    try:
        import multiprocessing as mp
        
        # Test available cores
        mp_cores = mp.cpu_count()
        print(f"  Available for multiprocessing: {mp_cores} cores")
        
        # Test process creation
        def test_worker(x):
            return x * x
        
        start = time.time()
        with mp.Pool(processes=min(8, mp_cores)) as pool:
            results = pool.map(test_worker, range(100))
        mp_time = time.time() - start
        
        print(f"  ✅ Multiprocessing test completed in {mp_time:.3f}s")
        print(f"  Results sample: {results[:5]}...")
        
        print()
        return len(results) == 100
        
    except Exception as e:
        print(f"  ❌ Multiprocessing test failed: {e}")
        print()
        return False

def test_memory_usage():
    """Test memory usage patterns"""
    print("💾 Memory Usage Test:")
    
    try:
        # Get initial memory
        initial_memory = psutil.virtual_memory().used / (1024**2)  # MB
        
        # Create large tensors to simulate embedding workload
        print("  Simulating embedding workload...")
        large_tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 384)  # Typical embedding size
            large_tensors.append(tensor)
        
        peak_memory = psutil.virtual_memory().used / (1024**2)  # MB
        memory_used = peak_memory - initial_memory
        
        # Cleanup
        del large_tensors
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        final_memory = psutil.virtual_memory().used / (1024**2)  # MB
        
        print(f"  Memory used during test: {memory_used:.1f}MB")
        print(f"  Memory recovered: {peak_memory - final_memory:.1f}MB")
        
        available_gb = psutil.virtual_memory().available / (1024**3)
        print(f"  Available for embedding: {available_gb:.1f}GB")
        
        print()
        return available_gb >= 8  # Need at least 8GB for large embeddings
        
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        print()
        return False

def run_full_test():
    """Run complete system test"""
    print_header()
    
    test_results = []
    
    # Run all tests
    test_results.append(("System Specs", test_system_specs()))
    test_results.append(("PyTorch Performance", test_pytorch_performance()))
    
    model, model_ok = test_model_loading()
    test_results.append(("Model Loading", model_ok))
    test_results.append(("Encoding Speed", test_encoding_speed(model)))
    test_results.append(("Multiprocessing", test_multiprocessing()))
    test_results.append(("Memory Usage", test_memory_usage()))
    
    # Results summary
    print("📊 TEST RESULTS SUMMARY:")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<20} {status}")
        if result:
            passed += 1
    
    print()
    print(f"Tests passed: {passed}/{total}")
    
    # Final recommendation
    if passed == total:
        print("🎉 PERFECT! Your AMD Ryzen 9950X is optimally configured!")
        print("💡 Ready to run: python embedding/embed_model_ryzen.py")
    elif passed >= total * 0.8:
        print("✅ GOOD! System is well configured with minor issues.")
        print("💡 Check AMD_RYZEN_SETUP.md for optimization tips")
    elif passed >= total * 0.6:
        print("⚠️ MODERATE: Some performance issues detected.")
        print("🔧 Review system optimization settings")
    else:
        print("❌ POOR: Significant configuration issues found.")
        print("🆘 Check requirements installation and system settings")
    
    print("\n📋 Next steps:")
    print("1. Review any failed tests above")
    print("2. Check AMD_RYZEN_SETUP.md for troubleshooting")
    print("3. Run: python embedding/embed_model_ryzen.py")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    try:
        success = run_full_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)