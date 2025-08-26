# 🚀 AMD Ryzen 9950X Setup Guide

Complete setup guide for optimizing the Hadits AI system on AMD Ryzen 9 9950X with 32GB RAM.

## 🎯 System Optimizations

### Target Hardware:
- **CPU**: AMD Ryzen 9 9950X (16 cores, 32 threads)
- **RAM**: 32GB DDR5
- **Storage**: SSD recommended
- **GPU**: Not required (CPU-only optimization)

### Expected Performance:
- **Speed**: 80-150 documents/second
- **Time for 30K documents**: 3-8 minutes
- **CPU Usage**: 85-95% across all cores
- **RAM Usage**: 8-16GB during processing

## 📋 Prerequisites

### System Requirements:
- Windows 10/11 (64-bit)
- Python 3.8+ (3.10+ recommended)
- At least 16GB free disk space
- Administrative privileges for some optimizations

### Python Version Check:
```bash
python --version
# Should be 3.8 or higher
```

## 🔧 Installation Steps

### Step 1: Create Virtual Environment
```bash
# Create dedicated environment
python -m venv venv_ryzen
cd hadits-ai

# Activate environment (Windows)
venv_ryzen\Scripts\activate

# Verify activation
python -c "import sys; print(sys.prefix)"
```

### Step 2: Install CPU-Optimized PyTorch
```bash
# Remove any existing PyTorch installations
pip uninstall torch torchvision torchaudio -y

# Install CPU-optimized PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 3: Install Ryzen-Optimized Dependencies
```bash
# Install from Ryzen-specific requirements
pip install -r requirements_ryzen.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CPU threads: {torch.get_num_threads()}')"
```

### Step 4: System Optimizations

#### Windows Power Settings:
```bash
# Set to High Performance mode
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Disable CPU throttling
powercfg /setacvalueindex scheme_current sub_processor PROCTHROTTLEMAX 100
powercfg /setdcvalueindex scheme_current sub_processor PROCTHROTTLEMAX 100
```

#### CPU Thread Affinity (Optional):
```bash
# Check current CPU topology
wmic cpu get NumberOfCores,NumberOfLogicalProcessors

# For advanced users - set process affinity to all cores
# This is handled automatically by the script
```

## 🚀 Running the Optimized Version

### Basic Usage:
```bash
# Activate environment
venv_ryzen\Scripts\activate

# Run AMD Ryzen optimized embedding
python embedding/embed_model_ryzen.py
```

### Advanced Configuration:
You can modify these settings in `embed_model_ryzen.py`:

```python
# Adjust these based on your workload
num_workers = 24        # Use 24 out of 32 threads
batch_size = 16         # Smaller batches for better parallelization
torch_threads = 4       # Threads per model instance
```

## 📊 Performance Monitoring

### System Resource Monitoring:
```bash
# Monitor CPU usage during processing
# Windows Task Manager -> Performance -> CPU

# Or use PowerShell
Get-Counter "\Processor(_Total)\% Processor Time" -SampleInterval 2 -MaxSamples 10
```

### Memory Usage:
```python
# Built-in monitoring in the script shows:
# - CPU cores utilization
# - RAM usage
# - Processing speed
# - Estimated completion time
```

## 🔧 Troubleshooting

### Issue: Slow Performance
**Symptoms**: Less than 50 docs/second
**Solutions**:
1. Close unnecessary applications
2. Check if antivirus is scanning the project folder
3. Ensure Python process has high priority:
   ```bash
   # In Task Manager, set Python process priority to "High"
   ```
4. Verify all CPU cores are being used

### Issue: Memory Errors
**Symptoms**: "Out of Memory" errors
**Solutions**:
1. Reduce `batch_size` in the script:
   ```python
   batch_size = 8  # Reduce from 16
   ```
2. Reduce `num_workers`:
   ```python
   num_workers = 16  # Reduce from 24
   ```
3. Close other memory-intensive applications

### Issue: Process Hangs
**Symptoms**: Script stops responding
**Solutions**:
1. Check Windows Defender real-time protection
2. Ensure sufficient disk space for temp files
3. Restart and try with fewer workers:
   ```python
   num_workers = 8
   ```

### Issue: Import Errors
**Symptoms**: ModuleNotFoundError
**Solutions**:
1. Verify virtual environment activation
2. Reinstall requirements:
   ```bash
   pip install -r requirements_ryzen.txt --force-reinstall
   ```
3. Check Python path:
   ```python
   import sys
   print(sys.path)
   ```

## ⚡ Performance Optimizations

### 1. Windows System Optimizations
```bash
# Disable Windows Search indexing for project folder
# Control Panel -> Indexing Options -> Modify -> Uncheck project folder

# Disable Windows Defender for project folder (if safe)
# Windows Security -> Virus & threat protection -> Exclusions
```

### 2. Environment Variables
```bash
# Add to your environment or .env file
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
VECLIB_MAXIMUM_THREADS=4
NUMEXPR_NUM_THREADS=4
```

### 3. Python Optimizations
```python
# Already included in the script:
torch.set_num_threads(4)           # Optimal for Ryzen
torch.set_num_interop_threads(2)   # Minimize overhead
```

## 📈 Expected Benchmarks

### Performance Comparison:

| System | Speed (docs/sec) | 30K docs time | CPU Usage |
|--------|------------------|---------------|-----------|
| MacBook M1 (MPS) | 60-100 | 5-8 min | 80% |
| RTX 3060 (CUDA) | 150-250 | 2-4 min | 40% |
| **Ryzen 9950X (CPU)** | **80-150** | **3-6 min** | **90%** |
| Regular CPU (8-core) | 20-40 | 15-25 min | 100% |

### Memory Usage:
- **Baseline**: ~2GB
- **Peak processing**: ~8-12GB
- **Available for other apps**: ~20GB

## 🎯 Quick Start Commands

```bash
# 1. Setup (one-time)
python -m venv venv_ryzen
venv_ryzen\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements_ryzen.txt

# 2. Run optimized embedding
python embedding/embed_model_ryzen.py

# 3. Test the complete system
python main.py
```

## 📁 File Structure

```
hadits-ai/
├── embedding/
│   ├── embed_model.py              # MacBook version
│   ├── embed_model_optimized.py    # GPU version  
│   └── embed_model_ryzen.py        # 🆕 AMD Ryzen optimized
├── requirements_ryzen.txt          # 🆕 Ryzen-specific requirements
├── AMD_RYZEN_SETUP.md             # 🆕 This setup guide
└── venv_ryzen/                    # 🆕 Dedicated environment
```

## 🆘 Support & Tips

### Performance Tips:
1. **Run during low system activity** - best performance when system is idle
2. **Close browsers and heavy apps** - frees up CPU cores and RAM
3. **Use SSD storage** - faster file I/O for embeddings
4. **Keep system cool** - prevents CPU throttling

### Monitoring Tools:
- **Task Manager**: Real-time CPU/RAM monitoring
- **Resource Monitor**: Detailed process information
- **Built-in script monitoring**: Shows progress and ETA

### When to Use This Version:
- ✅ No NVIDIA GPU available
- ✅ High-end AMD Ryzen processor (8+ cores)
- ✅ Sufficient RAM (16GB+)
- ✅ Want reliable CPU-only performance
- ❌ Don't use if you have RTX GPU (use optimized version instead)

---

🎉 **With proper setup, your AMD Ryzen 9950X should process 30K+ hadits documents in just 3-6 minutes using pure CPU power!**