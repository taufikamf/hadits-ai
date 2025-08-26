# 🛠️ Manual PyTorch CUDA Installation Guide

## ❌ Problem: Timeout Error
```
TimeoutError: The read operation timed out
ReadTimeoutError: HTTPSConnectionPool(host='download.pytorch.org', port=443): Read timed out.
```

File PyTorch sangat besar (2.8 GB) dan sering timeout. Berikut solusi manual:

## 🎯 Solution 1: Manual Download (RECOMMENDED)

### Step 1: Download Wheel Files Manually

1. **Buka browser dan kunjungi**: https://download.pytorch.org/whl/cu118/

2. **Download file berikut untuk Python 3.10** (sesuaikan dengan versi Python Anda):
   ```
   torch-2.1.2+cu118-cp310-cp310-win_amd64.whl
   torchvision-0.16.2+cu118-cp310-cp310-win_amd64.whl  
   torchaudio-2.1.2+cu118-cp310-cp310-win_amd64.whl
   ```

3. **Cek versi Python Anda**:
   ```cmd
   python --version
   ```
   - Python 3.10 → gunakan `cp310`
   - Python 3.9 → gunakan `cp39`
   - Python 3.11 → gunakan `cp311`

### Step 2: Install dari File Lokal

1. **Buka Command Prompt di folder download**
2. **Jalankan perintah**:
   ```cmd
   cd F:\hadits-ai
   F:\hadits-ai\venv\Scripts\activate
   pip install path\to\torch-2.1.2+cu118-cp310-cp310-win_amd64.whl
   pip install path\to\torchvision-0.16.2+cu118-cp310-cp310-win_amd64.whl
   pip install path\to\torchaudio-2.1.2+cu118-cp310-cp310-win_amd64.whl
   ```

### Step 3: Verify Installation
```cmd
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🎯 Solution 2: Alternative Index

Coba index mirror yang lebih cepat:

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🎯 Solution 3: Increase Timeout

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --timeout 7200 --retries 10
```

## 🎯 Solution 4: Split Installation

Install satu per satu:

```cmd
# 1. Install torch first
pip install torch --index-url https://download.pytorch.org/whl/cu118 --timeout 3600

# 2. Then torchvision
pip install torchvision --index-url https://download.pytorch.org/whl/cu118 --timeout 3600

# 3. Finally torchaudio  
pip install torchaudio --index-url https://download.pytorch.org/whl/cu118 --timeout 3600
```

## 🎯 Solution 5: Use Different Network

- Gunakan mobile hotspot
- Gunakan VPN
- Coba install pada jam sepi (malam hari)

## 🧪 Quick Test Script

Setelah instalasi, test dengan script ini:

```python
# test_cuda.py
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Test GPU operation
    x = torch.randn(1000, 1000).cuda()
    y = x @ x
    print("✅ GPU operations working!")
```

Run: `python test_cuda.py`

## 📋 Complete Manual Installation Steps

### 1. Prepare Environment
```cmd
cd F:\hadits-ai
F:\hadits-ai\venv\Scripts\activate
pip install --upgrade pip
pip uninstall torch torchvision torchaudio -y
```

### 2. Download Files (Choose one method)

**Method A: Browser Download**
- Visit: https://download.pytorch.org/whl/cu118/
- Download 3 wheel files for your Python version

**Method B: wget/curl (if available)**
```cmd
# For Python 3.10
wget https://download.pytorch.org/whl/cu118/torch-2.1.2%2Bcu118-cp310-cp310-win_amd64.whl
wget https://download.pytorch.org/whl/cu118/torchvision-0.16.2%2Bcu118-cp310-cp310-win_amd64.whl
wget https://download.pytorch.org/whl/cu118/torchaudio-2.1.2%2Bcu118-cp310-cp310-win_amd64.whl
```

### 3. Install Local Wheels
```cmd
pip install torch-2.1.2+cu118-cp310-cp310-win_amd64.whl
pip install torchvision-0.16.2+cu118-cp310-cp310-win_amd64.whl
pip install torchaudio-2.1.2+cu118-cp310-cp310-win_amd64.whl
```

### 4. Install Additional Packages
```cmd
pip install sentence-transformers transformers scikit-learn accelerate psutil
```

### 5. Test Installation
```cmd
python test_cuda.py
python test_gpu.py
```

## 🎉 Expected Results

After successful installation:
```
PyTorch: 2.1.2+cu118
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
CUDA version: 11.8
✅ GPU operations working!
```

## 🔧 If Still Having Issues

1. **Restart computer** after installation
2. **Update NVIDIA drivers** to latest version
3. **Check Windows Device Manager** - ensure RTX 3060 is recognized
4. **Try different CUDA version** (cu121 instead of cu118)
5. **Use Google Colab** as alternative for testing

## 📞 Quick Commands Reference

```cmd
# Check Python version
python --version

# Check pip version  
pip --version

# Check NVIDIA driver
nvidia-smi

# Activate virtual environment
F:\hadits-ai\venv\Scripts\activate

# Test CUDA after installation
python -c "import torch; print(torch.cuda.is_available())"
```
