"""
Simple PyTorch CUDA Installer for Windows
Handles timeout issues with alternative approaches
"""

import subprocess
import sys
import os

def run_cmd(command, description):
    """Run command with better error handling"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, text=True)
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Exit code: {e.returncode}")
        return False

def method_1_standard():
    """Standard PyTorch installation"""
    print("\n" + "="*50)
    print("METHOD 1: Standard Installation")
    print("="*50)
    
    commands = [
        (f'"{sys.executable}" -m pip install --upgrade pip', "Upgrading pip"),
        (f'"{sys.executable}" -m pip uninstall torch torchvision torchaudio -y', "Removing old PyTorch"),
        (f'"{sys.executable}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118', "Installing CUDA PyTorch"),
    ]
    
    for cmd, desc in commands:
        if not run_cmd(cmd, desc):
            print(f"❌ Method 1 failed at: {desc}")
            return False
    
    return True

def method_2_cpu_first():
    """Install CPU version first, then upgrade"""
    print("\n" + "="*50)
    print("METHOD 2: CPU First, Then CUDA")
    print("="*50)
    
    commands = [
        (f'"{sys.executable}" -m pip install torch torchvision torchaudio', "Installing CPU PyTorch"),
        (f'"{sys.executable}" -m pip install torch torchvision torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu118 --force-reinstall', "Upgrading to CUDA"),
    ]
    
    for cmd, desc in commands:
        if not run_cmd(cmd, desc):
            print(f"❌ Method 2 failed at: {desc}")
            return False
    
    return True

def method_3_manual_download():
    """Manual download method"""
    print("\n" + "="*50)
    print("METHOD 3: Manual Download")
    print("="*50)
    
    print("📥 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Open browser and go to: https://pytorch.org/get-started/locally/")
    print("2. Select: Stable -> Windows -> Pip -> Python -> CUDA 11.8")
    print("3. It will show a command like:")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("4. Copy that command and run it manually in your terminal")
    print("5. If download fails, try:")
    print("   - Using different internet connection")
    print("   - Downloading during off-peak hours")
    print("   - Using VPN")
    
    response = input("\nDid you successfully install PyTorch manually? (y/N): ")
    return response.lower() == 'y'

def test_pytorch():
    """Test PyTorch installation"""
    print("\n" + "="*50)
    print("TESTING PYTORCH INSTALLATION")
    print("="*50)
    
    test_code = '''
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test GPU operation
    x = torch.randn(10, 10).cuda()
    y = x @ x
    print("✅ GPU operations working")
else:
    print("❌ CUDA not available")
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=30)
        
        print("Test Results:")
        print(result.stdout)
        
        if result.stderr:
            print("Warnings:")
            print(result.stderr)
        
        if "CUDA available: True" in result.stdout and "GPU operations working" in result.stdout:
            print("🎉 PyTorch CUDA installation SUCCESS!")
            return True
        else:
            print("❌ PyTorch CUDA not working properly")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def install_additional_packages():
    """Install additional packages needed for the project"""
    print("\n" + "="*50)
    print("INSTALLING ADDITIONAL PACKAGES")
    print("="*50)
    
    packages = [
        "sentence-transformers",
        "transformers", 
        "scikit-learn",
        "accelerate",
        "psutil",
    ]
    
    success = True
    for package in packages:
        cmd = f'"{sys.executable}" -m pip install {package}'
        if not run_cmd(cmd, f"Installing {package}"):
            success = False
    
    return success

def main():
    print("🚀 SIMPLE PYTORCH CUDA INSTALLER")
    print("="*60)
    print("This script tries different methods to install PyTorch with CUDA")
    print("for your RTX 3060 on Windows")
    
    # Check NVIDIA driver
    print("\n🔍 Checking NVIDIA driver...")
    try:
        result = subprocess.run("nvidia-smi", capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA driver detected")
            lines = result.stdout.split('\n')[:5]
            for line in lines:
                if 'RTX' in line or 'NVIDIA-SMI' in line or 'Driver Version' in line:
                    print(f"  {line.strip()}")
        else:
            print("❌ NVIDIA driver not found")
            return
    except:
        print("❌ Cannot check NVIDIA driver")
        return
    
    # Try installation methods
    methods = [
        ("Standard Installation", method_1_standard),
        ("CPU First Method", method_2_cpu_first),
        ("Manual Download", method_3_manual_download),
    ]
    
    success = False
    for method_name, method_func in methods:
        print(f"\n🔄 Trying: {method_name}")
        if method_func():
            print(f"✅ {method_name} completed")
            if test_pytorch():
                success = True
                break
            else:
                print(f"⚠️ {method_name} installed but CUDA test failed")
        else:
            print(f"❌ {method_name} failed")
    
    if success:
        print("\n📦 Installing additional packages...")
        install_additional_packages()
        
        print("\n" + "="*60)
        print("🎉 INSTALLATION COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Restart your terminal/IDE")
        print("2. Run: python test_gpu.py")
        print("3. Run: python embedding/embed_model_optimized.py")
        
    else:
        print("\n" + "="*60)
        print("❌ ALL METHODS FAILED")
        print("="*60)
        print("\n🔧 MANUAL SOLUTIONS:")
        print("1. Network Issues:")
        print("   - Try different internet connection")
        print("   - Use mobile hotspot")
        print("   - Try installation at different time")
        
        print("\n2. Download wheel files manually:")
        print("   - Go to: https://download.pytorch.org/whl/cu118/")
        print("   - Download for your Python version (cp310 for Python 3.10):")
        print("     * torch-*-cp310-cp310-win_amd64.whl")
        print("     * torchvision-*-cp310-cp310-win_amd64.whl") 
        print("     * torchaudio-*-cp310-cp310-win_amd64.whl")
        print("   - Install: pip install downloaded_file.whl")
        
        print("\n3. Alternative:")
        print("   - Use Google Colab for testing")
        print("   - Try Windows Subsystem for Linux (WSL2)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Installation cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
