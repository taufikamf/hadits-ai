#!/usr/bin/env python3
"""
AMD Ryzen 9950X Optimization Installer
Automatically installs and configures the system for optimal CPU performance
"""

import os
import sys
import subprocess
import platform
import psutil
import time
from pathlib import Path

def print_banner():
    """Print installation banner"""
    print("=" * 70)
    print("🚀 AMD RYZEN 9950X HADITS AI OPTIMIZER")
    print("=" * 70)
    print("Automatically configures your system for maximum CPU performance")
    print()

def check_system_compatibility():
    """Check if system is compatible"""
    print("🔍 System Compatibility Check:")
    
    # Check OS
    if not platform.system() == "Windows":
        print("❌ This installer is designed for Windows systems")
        return False
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python {python_version.major}.{python_version.minor} detected. Need Python 3.8+")
        return False
    
    # Check CPU cores
    cpu_cores = psutil.cpu_count(logical=True)
    if cpu_cores < 8:
        print(f"⚠️ Warning: Only {cpu_cores} CPU threads detected. Recommended: 16+")
    
    # Check RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    if ram_gb < 16:
        print(f"⚠️ Warning: Only {ram_gb:.1f}GB RAM detected. Recommended: 32GB+")
    
    print(f"✅ Windows {platform.release()}")
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"✅ CPU threads: {cpu_cores}")
    print(f"✅ RAM: {ram_gb:.1f}GB")
    print()
    
    return True

def run_command(cmd, description, capture_output=False):
    """Run a command with error handling"""
    print(f"🔄 {description}...")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Failed: {result.stderr}")
                return False
            return result.stdout.strip()
        else:
            result = subprocess.run(cmd, shell=True)
            if result.returncode != 0:
                print(f"❌ Command failed: {cmd}")
                return False
            print(f"✅ {description} completed")
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def install_cpu_pytorch():
    """Install CPU-optimized PyTorch"""
    print("\n🧠 Installing CPU-Optimized PyTorch:")
    
    # Uninstall existing PyTorch
    uninstall_cmd = "pip uninstall torch torchvision torchaudio -y"
    run_command(uninstall_cmd, "Removing existing PyTorch")
    
    # Install CPU-optimized PyTorch
    install_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    success = run_command(install_cmd, "Installing CPU-optimized PyTorch")
    
    if success:
        # Verify installation
        verify_code = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CPU threads: {torch.get_num_threads()}")
print(f"MKL-DNN available: {torch.backends.mkldnn.is_available() if hasattr(torch.backends, 'mkldnn') else 'N/A'}")
'''
        
        result = run_command(f'python -c "{verify_code}"', "Verifying PyTorch", capture_output=True)
        if result:
            print("📊 PyTorch verification:")
            print(result)
    
    return success

def install_requirements():
    """Install Ryzen-specific requirements"""
    print("\n📦 Installing Ryzen-Optimized Dependencies:")
    
    requirements_file = "requirements_ryzen.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found!")
        return False
    
    cmd = f"pip install -r {requirements_file}"
    return run_command(cmd, "Installing requirements")

def optimize_system_settings():
    """Apply system optimizations"""
    print("\n⚙️ Applying System Optimizations:")
    
    optimizations = [
        {
            "cmd": "powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
            "desc": "Setting High Performance power plan"
        },
        {
            "cmd": "powercfg /setacvalueindex scheme_current sub_processor PROCTHROTTLEMAX 100",
            "desc": "Disabling CPU throttling (AC power)"
        },
        {
            "cmd": "powercfg /setdcvalueindex scheme_current sub_processor PROCTHROTTLEMAX 100", 
            "desc": "Disabling CPU throttling (battery power)"
        }
    ]
    
    for opt in optimizations:
        run_command(opt["cmd"], opt["desc"])

def create_test_script():
    """Create a quick test script"""
    test_script = '''import torch
import psutil
import time
from sentence_transformers import SentenceTransformer

def test_ryzen_setup():
    print("🧪 AMD Ryzen Setup Test")
    print("=" * 40)
    
    # System info
    print(f"CPU cores: {psutil.cpu_count(logical=True)}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    
    # Model loading test
    print("\\n🤖 Testing model loading...")
    start = time.time()
    model = SentenceTransformer("intfloat/e5-small-v2", device="cpu")
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.2f} seconds")
    
    # Encoding test
    print("\\n📝 Testing encoding speed...")
    test_texts = ["This is a test sentence."] * 100
    start = time.time()
    embeddings = model.encode(test_texts, show_progress_bar=False)
    encode_time = time.time() - start
    speed = len(test_texts) / encode_time
    print(f"✅ Speed: {speed:.1f} sentences/second")
    print(f"📐 Embedding shape: {embeddings.shape}")
    
    print("\\n🎉 Setup test completed successfully!")

if __name__ == "__main__":
    test_ryzen_setup()
'''
    
    with open("test_ryzen_setup.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ Created test_ryzen_setup.py")

def set_environment_variables():
    """Set optimization environment variables"""
    print("\n🌐 Setting Environment Variables:")
    
    env_vars = {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", 
        "VECLIB_MAXIMUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4"
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ Set {var}={value}")

def main():
    """Main installation process"""
    print_banner()
    
    # Check compatibility
    if not check_system_compatibility():
        print("\n❌ System compatibility check failed!")
        sys.exit(1)
    
    print("🚀 Starting AMD Ryzen optimization installation...\n")
    
    try:
        # Install CPU-optimized PyTorch
        if not install_cpu_pytorch():
            raise Exception("PyTorch installation failed")
        
        # Install requirements
        if not install_requirements():
            raise Exception("Requirements installation failed") 
        
        # Apply system optimizations
        optimize_system_settings()
        
        # Set environment variables
        set_environment_variables()
        
        # Create test script
        create_test_script()
        
        print("\n" + "=" * 70)
        print("🎉 AMD RYZEN OPTIMIZATION COMPLETED!")
        print("=" * 70)
        
        print("\n🎯 Next Steps:")
        print("1. Run test: python test_ryzen_setup.py")
        print("2. Run embedding: python embedding/embed_model_ryzen.py")
        print("3. Check AMD_RYZEN_SETUP.md for detailed usage")
        
        print("\n💡 Performance Tips:")
        print("- Close unnecessary applications before running")
        print("- Monitor CPU usage to ensure all cores are utilized")
        print("- Expected speed: 80-150 documents/second")
        
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        print("\n🔧 Manual steps:")
        print("1. Check AMD_RYZEN_SETUP.md for manual installation")
        print("2. Ensure Python 3.8+ is installed")
        print("3. Try installing requirements manually")
        sys.exit(1)

if __name__ == "__main__":
    main()