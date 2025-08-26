"""
Fixed Windows CUDA PyTorch Installer
Handles timeout issues and provides multiple installation methods
"""

import subprocess
import sys
import os
import platform
import time

def run_command_with_retry(command, description, max_retries=3, timeout=3600):
    """Run a command with retry mechanism and longer timeout"""
    print(f"🔄 {description}...")
    
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}")
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=timeout  # 1 hour timeout
            )
            print(f"✅ {description} completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                print("  Retrying...")
                time.sleep(5)
            continue
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error on attempt {attempt + 1}: {e}")
            if e.output:
                print(f"Output: {e.output}")
            if e.stderr:
                print(f"Error: {e.stderr}")
            
            if attempt < max_retries - 1:
                print("  Retrying...")
                time.sleep(10)
            continue
    
    print(f"❌ All attempts failed for: {description}")
    return False

def check_system():
    """Check system requirements"""
    print("🔍 SYSTEM CHECK")
    print("=" * 40)
    
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
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
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ NVIDIA driver detected")
            print("Driver info:")
            lines = result.stdout.split('\n')
            for line in lines[:10]:
                if line.strip() and ('NVIDIA-SMI' in line or 'Driver Version' in line or 'RTX' in line or 'GeForce' in line):
                    print(f"  {line.strip()}")
            return True
        else:
            print("❌ NVIDIA driver not found or not working")
            return False
    except Exception as e:
        print(f"❌ Error checking NVIDIA driver: {e}")
        return False

def upgrade_pip():
    """Upgrade pip to latest version"""
    print("\n🔧 UPGRADING PIP")
    print("=" * 40)
    
    command = f'"{sys.executable}" -m pip install --upgrade pip'
    return run_command_with_retry(command, "Upgrading pip", timeout=300)

def uninstall_existing_pytorch():
    """Remove existing PyTorch installations"""
    print("\n🗑️ REMOVING EXISTING PYTORCH")
    print("=" * 40)
    
    packages_to_remove = [
        "torch",
        "torchvision", 
        "torchaudio",
    ]
    
    success = True
    for package in packages_to_remove:
        command = f'"{sys.executable}" -m pip uninstall {package} -y'
        if not run_command_with_retry(command, f"Uninstalling {package}", timeout=120):
            success = False
    
    return success

def install_cuda_pytorch_method1():
    """Install CUDA PyTorch - Method 1: Direct from PyTorch index"""
    print("\n🚀 METHOD 1: INSTALLING FROM PYTORCH INDEX")
    print("=" * 40)
    
    # Configure pip for better download handling
    pip_config_commands = [
        f'"{sys.executable}" -m pip config set global.timeout 3600',
        f'"{sys.executable}" -m pip config set global.retries 5',
    ]
    
    for cmd in pip_config_commands:
        run_command_with_retry(cmd, "Configuring pip", timeout=30)
    
    # Install PyTorch with CUDA 11.8
    command = (
        f'"{sys.executable}" -m pip install torch torchvision torchaudio '
        '--index-url https://download.pytorch.org/whl/cu118 '
        '--timeout 3600 --retries 5'
    )
    
    return run_command_with_retry(command, "Installing CUDA PyTorch (Method 1)", max_retries=2, timeout=3600)

def install_cuda_pytorch_method2():
    """Install CUDA PyTorch - Method 2: Direct wheel download"""
    print("\n🚀 METHOD 2: INSTALLING VIA DIRECT DOWNLOAD")
    print("=" * 40)
    
    # For Python 3.10 on Windows
    if sys.version_info[:2] == (3, 10):
        torch_wheel = "https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp310-cp310-win_amd64.whl"
        torchvision_wheel = "https://download.pytorch.org/whl/cu118/torchvision-0.16.2%2Bcu118-cp310-cp310-win_amd64.whl"
        torchaudio_wheel = "https://download.pytorch.org/whl/cu118/torchaudio-2.1.2%2Bcu118-cp310-cp310-win_amd64.whl"
    else:
        print("❌ Method 2 only supports Python 3.10")
        return False
    
    wheels = [
        (torch_wheel, "PyTorch"),
        (torchvision_wheel, "TorchVision"),
        (torchaudio_wheel, "TorchAudio"),
    ]
    
    for wheel_url, name in wheels:
        command = f'"{sys.executable}" -m pip install "{wheel_url}" --timeout 3600'
        if not run_command_with_retry(command, f"Installing {name}", timeout=3600):
            return False
    
    return True

def install_cuda_pytorch_method3():
    """Install CUDA PyTorch - Method 3: Local download then install"""
    print("\n🚀 METHOD 3: DOWNLOAD THEN INSTALL")
    print("=" * 40)
    
    print("This method requires manual download:")
    print("1. Go to: https://pytorch.org/get-started/locally/")
    print("2. Select: Stable, Windows, Pip, Python, CUDA 11.8")
    print("3. Download the wheel files manually")
    print("4. Install using: pip install downloaded_wheel_file.whl")
    
    response = input("Have you downloaded the wheel files manually? (y/N): ").lower()
    if response == 'y':
        wheel_dir = input("Enter the directory path where wheel files are located: ")
        if os.path.exists(wheel_dir):
            command = f'"{sys.executable}" -m pip install "{wheel_dir}\\torch*.whl" "{wheel_dir}\\torchvision*.whl" "{wheel_dir}\\torchaudio*.whl"'
            return run_command_with_retry(command, "Installing from local wheels", timeout=300)
    
    return False

def install_cuda_pytorch_method4():
    """Install CUDA PyTorch - Method 4: CPU version first, then CUDA"""
    print("\n🚀 METHOD 4: CPU FIRST, THEN UPGRADE TO CUDA")
    print("=" * 40)
    
    # Install CPU version first (smaller download)
    print("Step 1: Installing CPU version...")
    cpu_command = f'"{sys.executable}" -m pip install torch torchvision torchaudio'
    if not run_command_with_retry(cpu_command, "Installing CPU PyTorch", timeout=1800):
        return False
    
    # Then upgrade to CUDA version
    print("Step 2: Upgrading to CUDA version...")
    cuda_command = (
        f'"{sys.executable}" -m pip install torch torchvision torchaudio '
        '--upgrade --index-url https://download.pytorch.org/whl/cu118 '
        '--force-reinstall --timeout 3600'
    )
    
    return run_command_with_retry(cuda_command, "Upgrading to CUDA PyTorch", timeout=3600)

def install_optimized_dependencies():
    """Install other optimized dependencies"""
    print("\n📦 INSTALLING OPTIMIZED DEPENDENCIES")
    print("=" * 40)
    
    dependencies = [
        "sentence-transformers>=2.6.1",
        "transformers>=4.41.1", 
        "scikit-learn",
        "psutil",
        "accelerate",
        "tqdm",
    ]
    
    success = True
    for dep in dependencies:
        command = f'"{sys.executable}" -m pip install "{dep}" --timeout 600'
        if not run_command_with_retry(command, f"Installing {dep}", timeout=600):
            success = False
    
    return success

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
        
    # Test basic GPU operations
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.mm(x, x)
        print("✅ Basic GPU tensor operations working")
        
        # Test sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        test_encoding = model.encode(["Test sentence"], convert_to_numpy=True)
        print("✅ Sentence-transformers GPU support working")
        print(f"Model device: {model.device}")
        
    except Exception as e:
        print(f"⚠️ GPU operations issue: {e}")
else:
    print("❌ CUDA not available")
    print("Possible solutions:")
    print("1. Restart your computer")
    print("2. Check NVIDIA driver installation")
    print("3. Verify GPU is properly seated")
'''
    
    print("Running comprehensive PyTorch test...")
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script], 
            capture_output=True, 
            text=True, 
            timeout=120
        )
        
        print("Test output:")
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
            
        if "CUDA available: True" in result.stdout and "GPU tensor operations working" in result.stdout:
            print("🎉 CUDA installation successful!")
            return True
        else:
            print("❌ CUDA not working properly")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_quick_test_script():
    """Create a quick GPU test script"""
    print("\n📝 CREATING QUICK TEST SCRIPT")
    print("=" * 40)
    
    script_content = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA not available - check installation")
'''
    
    with open("quick_gpu_test.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created quick_gpu_test.py")
    print("Run: python quick_gpu_test.py")

def print_troubleshooting_guide():
    """Print detailed troubleshooting guide"""
    print("\n" + "=" * 60)
    print("🔧 TROUBLESHOOTING GUIDE")
    print("=" * 60)
    
    print("\n📋 If installation still fails:")
    print("1. Network Issues:")
    print("   - Use VPN or different internet connection")
    print("   - Download wheels manually from pytorch.org")
    print("   - Try installation during off-peak hours")
    
    print("\n2. System Issues:")
    print("   - Restart computer after NVIDIA driver installation")
    print("   - Run Command Prompt as Administrator")
    print("   - Disable antivirus temporarily during installation")
    
    print("\n3. Python Environment:")
    print("   - Create fresh virtual environment")
    print("   - Use Python 3.9 or 3.10 (most compatible)")
    print("   - Ensure pip is latest version")
    
    print("\n4. Manual Installation:")
    print("   - Download from: https://pytorch.org/get-started/locally/")
    print("   - Select: Stable -> Windows -> Pip -> Python -> CUDA 11.8")
    print("   - Download .whl files and install locally")
    
    print("\n5. Alternative Solutions:")
    print("   - Try CUDA 12.1 instead of 11.8")
    print("   - Use Windows Subsystem for Linux (WSL2)")
    print("   - Consider using Google Colab for testing")

def main():
    """Main installation process with multiple methods"""
    print("🚀 FIXED WINDOWS RTX CUDA PYTORCH INSTALLER")
    print("=" * 60)
    print("This installer tries multiple methods to handle network timeouts")
    
    # Check system
    if not check_system():
        return
    
    # Check NVIDIA driver
    if not check_nvidia_driver():
        print("\n❌ NVIDIA driver issues detected")
        print("Please install/update NVIDIA drivers and restart")
        return
    
    # Upgrade pip first
    if not upgrade_pip():
        print("⚠️ Failed to upgrade pip, continuing anyway...")
    
    # Remove existing PyTorch
    print("\n🧹 Cleaning existing installations...")
    uninstall_existing_pytorch()
    
    # Try multiple installation methods
    installation_methods = [
        ("Method 1: PyTorch Index", install_cuda_pytorch_method1),
        ("Method 2: Direct Wheels", install_cuda_pytorch_method2),
        ("Method 4: CPU then CUDA", install_cuda_pytorch_method4),
    ]
    
    installation_success = False
    
    for method_name, method_func in installation_methods:
        print(f"\n🔄 Trying {method_name}")
        if method_func():
            installation_success = True
            print(f"✅ {method_name} succeeded!")
            break
        else:
            print(f"❌ {method_name} failed, trying next method...")
    
    if not installation_success:
        print("\n❌ All automatic installation methods failed")
        print("🔧 Trying manual method...")
        if install_cuda_pytorch_method3():
            installation_success = True
    
    if installation_success:
        print("\n📦 Installing additional dependencies...")
        install_optimized_dependencies()
        
        print("\n🧪 Testing installation...")
        if test_installation():
            create_quick_test_script()
            print("\n🎉 Installation completed successfully!")
            print("\nNext steps:")
            print("1. Run: python quick_gpu_test.py")
            print("2. Run: python test_gpu.py")
            print("3. Run: python embedding/embed_model_optimized.py")
        else:
            print("⚠️ Installation completed but CUDA test failed")
            print("Please restart your computer and test again")
    else:
        print("\n❌ Installation failed with all methods")
        print_troubleshooting_guide()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Installation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
