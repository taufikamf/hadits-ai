"""
Windows CUDA PyTorch Installer and Optimizer
Optimizes PyTorch installation for NVIDIA GPUs on Windows
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return False

def check_system():
    """Check system requirements"""
    print("🔍 SYSTEM CHECK")
    print("=" * 40)
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    if platform.system() != "Windows":
        print("❌ This script is designed for Windows")
        return False
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        return False
    
    print("✅ System requirements OK")
    return True

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    print("\n🎮 NVIDIA DRIVER CHECK")
    print("=" * 40)
    
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA driver detected")
            print("Driver info:")
            # Extract driver version from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines[:5]:
                if line.strip():
                    print(f"  {line.strip()}")
            return True
        else:
            print("❌ NVIDIA driver not found or not working")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA driver may not be installed")
        return False

def uninstall_existing_pytorch():
    """Remove existing PyTorch installations"""
    print("\n🗑️ REMOVING EXISTING PYTORCH")
    print("=" * 40)
    
    packages_to_remove = [
        "torch",
        "torchvision", 
        "torchaudio",
        "torch-audio"
    ]
    
    for package in packages_to_remove:
        command = f"{sys.executable} -m pip uninstall {package} -y"
        run_command(command, f"Uninstalling {package}")

def install_cuda_pytorch():
    """Install CUDA-enabled PyTorch"""
    print("\n🚀 INSTALLING CUDA PYTORCH")
    print("=" * 40)
    
    # PyTorch with CUDA 11.8 (compatible with most RTX cards)
    command = (
        f"{sys.executable} -m pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu118"
    )
    
    success = run_command(command, "Installing CUDA PyTorch")
    
    if success:
        print("✅ CUDA PyTorch installation completed")
    else:
        print("❌ Failed to install CUDA PyTorch")
        print("💡 Trying alternative installation...")
        
        # Fallback: install with conda if available
        alt_command = "conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y"
        conda_success = run_command(alt_command, "Installing with conda")
        
        if not conda_success:
            print("❌ Alternative installation also failed")
            return False
    
    return True

def install_optimized_dependencies():
    """Install other optimized dependencies"""
    print("\n📦 INSTALLING OPTIMIZED DEPENDENCIES")
    print("=" * 40)
    
    dependencies = [
        "sentence-transformers>=2.6.1",
        "transformers>=4.41.1", 
        "scikit-learn",
        "psutil",  # For system monitoring
        "accelerate",  # For model optimization
    ]
    
    for dep in dependencies:
        command = f"{sys.executable} -m pip install {dep} --upgrade"
        run_command(command, f"Installing {dep}")

def test_installation():
    """Test the PyTorch CUDA installation"""
    print("\n🧪 TESTING INSTALLATION")
    print("=" * 40)
    
    test_script = '''
import torch
import sys

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
    # Test tensor operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x)
    print("✅ GPU tensor operations working")
    
    # Test sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        test_encoding = model.encode(["Test sentence"], convert_to_numpy=True)
        print("✅ Sentence-transformers GPU support working")
        print(f"Model device: {model.device}")
    except Exception as e:
        print(f"⚠️ Sentence-transformers issue: {e}")
else:
    print("❌ CUDA not available")
'''
    
    print("Running PyTorch CUDA test...")
    try:
        result = subprocess.run([sys.executable, "-c", test_script], 
                              capture_output=True, text=True, timeout=60)
        
        print("Test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
            
        if "CUDA available: True" in result.stdout:
            print("🎉 CUDA installation successful!")
            return True
        else:
            print("❌ CUDA not detected after installation")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_performance_script():
    """Create a performance testing script"""
    print("\n📝 CREATING PERFORMANCE TEST SCRIPT")
    print("=" * 40)
    
    script_content = '''
# Performance test for Windows RTX setup
import torch
from sentence_transformers import SentenceTransformer
import time

def performance_test():
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
        
    print("🚀 RTX Performance Test")
    print("=" * 30)
    
    model = SentenceTransformer("intfloat/e5-small-v2", device="cuda")
    print(f"Model loaded on: {model.device}")
    
    # Test different batch sizes
    test_texts = ["Performance test sentence."] * 1000
    batch_sizes = [16, 32, 64, 128]
    
    print("\\nBatch size performance:")
    for batch_size in batch_sizes:
        try:
            torch.cuda.empty_cache()
            start = time.time()
            embeddings = model.encode(test_texts, batch_size=batch_size)
            elapsed = time.time() - start
            speed = len(test_texts) / elapsed
            print(f"Batch {batch_size:3d}: {speed:6.1f} texts/sec")
        except RuntimeError as e:
            if "memory" in str(e):
                print(f"Batch {batch_size:3d}: Out of memory")
                break

if __name__ == "__main__":
    performance_test()
'''
    
    with open("test_rtx_performance.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created test_rtx_performance.py")

def print_final_instructions():
    """Print final setup instructions"""
    print("\n" + "=" * 60)
    print("🎉 INSTALLATION COMPLETE!")
    print("=" * 60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Restart your terminal/IDE")
    print("2. Run: python test_gpu.py")
    print("3. Run: python embedding/embed_model_optimized.py")
    print("4. For performance testing: python test_rtx_performance.py")
    
    print("\n💡 PERFORMANCE TIPS:")
    print("• Close other GPU-intensive applications")
    print("• Set Windows power mode to 'High Performance'")
    print("• Update to latest NVIDIA drivers")
    print("• Make sure your PSU can handle RTX 3060 power requirements")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("• If CUDA still not detected, restart computer")
    print("• Check Windows Device Manager for GPU status")
    print("• Verify NVIDIA Control Panel shows RTX 3060")

def main():
    """Main installation process"""
    print("🚀 WINDOWS RTX 3060 PYTORCH CUDA OPTIMIZER")
    print("=" * 60)
    print("This script will optimize PyTorch for RTX 3060 on Windows")
    print("⚠️  This will uninstall existing PyTorch installations")
    
    response = input("\\nContinue? (y/N): ").lower()
    if response != 'y':
        print("Installation cancelled")
        return
    
    # Check system
    if not check_system():
        return
    
    # Check NVIDIA driver
    if not check_nvidia_driver():
        print("\\n❌ Please install NVIDIA drivers first:")
        print("https://www.nvidia.com/Download/index.aspx")
        return
    
    # Remove existing PyTorch
    uninstall_existing_pytorch()
    
    # Install CUDA PyTorch
    if not install_cuda_pytorch():
        print("❌ PyTorch installation failed")
        return
    
    # Install other dependencies
    install_optimized_dependencies()
    
    # Test installation
    if test_installation():
        create_performance_script()
        print_final_instructions()
    else:
        print("❌ Installation verification failed")
        print("Please check the error messages above")

if __name__ == "__main__":
    main()
