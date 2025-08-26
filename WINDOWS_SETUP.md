# 🚀 Windows RTX 3060 Optimization Guide

Panduan lengkap untuk mengoptimalkan Hadits AI system pada Windows dengan RTX 3060.

## 🎯 Masalah yang Dipecahkan

Jika Anda mengalami:
- Embedding process sangat lambat (>1 jam) pada Windows
- CPU usage tinggi saat proses embedding
- GPU tidak terdeteksi oleh PyTorch
- Performa lebih lambat dibanding MacBook

Maka panduan ini akan membantu Anda.

## 📋 Prasyarat

### 1. Hardware Requirements
- NVIDIA RTX 3060 atau GPU CUDA-compatible lainnya
- RAM minimal 16GB (recommended)
- SSD dengan space minimal 10GB
- PSU yang memadai untuk RTX 3060

### 2. Software Requirements
- Windows 10/11 (64-bit)
- Python 3.8+ 
- NVIDIA Driver terbaru

## 🔧 Langkah-langkah Setup

### Step 1: Install NVIDIA Driver
1. Download driver terbaru dari: https://www.nvidia.com/Download/index.aspx
2. Pilih RTX 3060 dan Windows version Anda
3. Install dan restart komputer
4. Verifikasi dengan menjalankan `nvidia-smi` di Command Prompt

### Step 2: Auto-Install CUDA PyTorch
```bash
# Jalankan installer otomatis
python install_cuda_pytorch.py
```

Atau manual:
```bash
# Remove PyTorch yang ada
pip uninstall torch torchvision torchaudio -y

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Verifikasi Setup
```bash
python test_gpu.py
```

Output yang diharapkan:
```
✅ CUDA available: True
✅ GPU: NVIDIA GeForce RTX 3060
✅ Model device: cuda
```

### Step 4: Test Performance
```bash
python embedding/embed_model_optimized.py
```

## 📊 Expected Performance

### Before Optimization (CPU only):
- **Speed**: ~5-10 texts/second
- **Time for 30K documents**: 60-90 minutes
- **GPU Usage**: 0%
- **CPU Usage**: 100%

### After Optimization (GPU accelerated):
- **Speed**: ~100-200 texts/second  
- **Time for 30K documents**: 3-8 minutes
- **GPU Usage**: 80-95%
- **CPU Usage**: 20-40%

**Expected speedup: 10-20x faster**

## 🏗️ Architecture Differences

### MacBook M1/M2 (MPS):
```python
device = "mps"  # Metal Performance Shaders
batch_size = 32
# Optimized for unified memory architecture
```

### Windows RTX 3060 (CUDA):
```python
device = "cuda"  # CUDA acceleration
batch_size = 64  # Larger batch for dedicated VRAM
# Optimized for discrete GPU with 12GB VRAM
```

## 🛠️ Troubleshooting

### Issue: "CUDA not available"
**Solutions:**
1. Restart terminal/IDE after PyTorch installation
2. Verify NVIDIA driver installation: `nvidia-smi`
3. Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
4. Check Windows Device Manager for GPU status

### Issue: "Out of memory" errors
**Solutions:**
1. Reduce batch size in `embed_model_optimized.py`
2. Close other applications using GPU
3. Monitor GPU memory: `nvidia-smi`

### Issue: Still slow performance
**Solutions:**
1. Verify GPU is actually being used: Check `nvidia-smi` during embedding
2. Update to latest NVIDIA drivers
3. Set Windows power mode to "High Performance"
4. Disable Windows Defender real-time scanning for project folder

### Issue: RTX 3060 not detected
**Solutions:**
1. Check physical GPU installation
2. Verify PSU provides adequate power
3. Update motherboard BIOS
4. Reseat GPU in PCIe slot

## 📈 Performance Optimization Tips

### 1. Windows Settings
- Set power plan to "High Performance"
- Disable Windows Defender for project folder
- Close unnecessary background applications
- Enable "Hardware-accelerated GPU scheduling"

### 2. NVIDIA Settings
- Set "Power Management Mode" to "Prefer Maximum Performance"
- Enable "CUDA - GPUs" in NVIDIA Control Panel
- Update to latest Game Ready drivers

### 3. Python Environment
```bash
# Use dedicated virtual environment
python -m venv venv_gpu
venv_gpu\Scripts\activate

# Install optimized packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install sentence-transformers accelerate
```

### 4. Code Optimizations
- Use optimal batch sizes (32-64 for RTX 3060)
- Enable mixed precision if supported
- Clear GPU cache between operations
- Use `torch.cuda.empty_cache()` periodically

## 🔍 Monitoring Performance

### Real-time GPU monitoring:
```bash
# In separate terminal
nvidia-smi -l 1  # Update every second
```

### Python performance monitoring:
```python
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB used")
print(f"GPU Utilization: Check nvidia-smi")
```

## 📂 File Structure untuk Windows

```
hadits-ai/
├── embedding/
│   ├── embed_model.py              # Original version
│   ├── embed_model_optimized.py    # Windows optimized version
│   └── test_gpu.py                 # GPU diagnostic tool
├── install_cuda_pytorch.py         # Auto installer
├── test_rtx_performance.py         # Performance benchmark
└── WINDOWS_SETUP.md                # This guide
```

## 🎯 Quick Start Commands

```bash
# 1. Setup (one-time)
python install_cuda_pytorch.py

# 2. Verify setup
python test_gpu.py

# 3. Run optimized embedding
python embedding/embed_model_optimized.py

# 4. Performance test
python test_rtx_performance.py
```

## 🆘 Support

Jika masih mengalami masalah:

1. **Hardware Issues**: Periksa instalasi fisik RTX 3060
2. **Driver Issues**: Reinstall NVIDIA drivers
3. **PyTorch Issues**: Gunakan installer otomatis
4. **Performance Issues**: Ikuti optimization tips di atas

## 📚 Additional Resources

- [PyTorch CUDA Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA RTX 3060 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)

---

🎉 **Setelah setup yang benar, embedding 30K+ dokumen hadits di RTX 3060 seharusnya hanya membutuhkan 3-8 menit, bukan 1+ jam!**
