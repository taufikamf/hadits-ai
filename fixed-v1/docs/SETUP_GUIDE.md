# Setup Guide - Enhanced Hadith AI Fixed V1

This comprehensive guide will help you set up and configure the Enhanced Hadith AI system from start to finish.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended for large datasets)
- **Storage**: At least 5GB free space for indices and embeddings
- **GPU**: Optional but recommended for faster embedding generation

### Required Data Files
Ensure you have these files in your data directory:
```
data/
├── processed/
│   └── hadits_docs.json           # Main hadith documents
├── csv/                           # Original CSV files (optional)
│   ├── shahih_bukhari.csv
│   ├── shahih_muslim.csv
│   └── ...
└── processed/
    └── keywords_map_grouped_cleaned.json  # Cleaned keywords (if available)
```

## 🔧 Installation

### Step 1: Clone and Navigate
```bash
cd your-hadith-ai-project
cd fixed-v1
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv_fixed_v1

# Activate virtual environment
# On Windows:
venv_fixed_v1\Scripts\activate
# On Linux/Mac:
source venv_fixed_v1/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install core dependencies
pip install numpy pandas scikit-learn
pip install sentence-transformers torch
pip install flask flask-cors
pip install tqdm pathlib

# For FAISS (optional but recommended)
pip install faiss-cpu  # or faiss-gpu if you have CUDA

# For additional text processing
pip install unicodedata2
```

### Step 4: Set Environment Variables
```bash
# Linux/Mac
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONIOENCODING=utf-8

# Windows
set PYTHONPATH=%PYTHONPATH%;%cd%
set PYTHONIOENCODING=utf-8
```

## 🏗️ System Setup

### Step 1: Prepare Data Structure
Create the necessary directory structure:
```bash
mkdir -p data/enhanced_index_v1
mkdir -p logs
mkdir -p output
```

### Step 2: Validate Data Files
Check that your hadith documents are properly formatted:
```python
import json

# Check hadits_docs.json format
with open('../data/processed/hadits_docs.json', 'r', encoding='utf-8') as f:
    docs = json.load(f)
    
print(f"Found {len(docs)} documents")
print(f"Sample document keys: {list(docs[0].keys())}")

# Ensure each document has 'id' and 'terjemah' fields
required_fields = ['id', 'terjemah']
valid_docs = [doc for doc in docs if all(field in doc for field in required_fields)]
print(f"Valid documents: {len(valid_docs)}")
```

## 🚀 Component-by-Component Setup

### Step 1: Test Individual Components

#### A. Query Preprocessor
```bash
cd preprocessing
python query_preprocessor.py
```
Expected output: Test queries with preprocessing results

#### B. Keyword Extractor
```bash
cd extraction
python enhanced_keyword_extractor.py
```
Expected output: Enhanced keywords map generated

#### C. Embedding System
```bash
cd embedding
python enhanced_embedding_system.py
```
Expected output: Document embeddings generated

#### D. Retrieval System
```bash
cd retrieval
python enhanced_retrieval_system.py --query "hukum shalat jumat"
```
Expected output: Retrieval results with scores

### Step 2: Run Complete Indexing Pipeline
```bash
# Run the complete indexing pipeline
python indexing/enhanced_indexing_pipeline.py \
  --input ../data/processed/hadits_docs.json \
  --output-dir ../data/enhanced_index_v1 \
  --keyword-freq 40 \
  --batch-size 32
```

This will:
1. Extract enhanced keywords (15-20 minutes)
2. Generate embeddings (30-45 minutes)
3. Build FAISS index (5-10 minutes)
4. Create metadata and validate system

### Step 3: Test Service Layer
```bash
# Test the service
python service/hadith_ai_service.py \
  --query "bagaimana cara berwudhu yang benar?" \
  --max-results 5 \
  --show-scores
```

Expected output: Formatted response with hadith results

### Step 4: Start API Server
```bash
# Start the API server
python service/api_server.py --host 127.0.0.1 --port 5000 --debug
```

Test the API:
```bash
# Health check
curl http://127.0.0.1:5000/health

# Test query
curl -X POST http://127.0.0.1:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "hukum zakat fitrah", "max_results": 3}'
```

## 🔧 Configuration

### Environment Configuration
Create a `.env` file in the fixed-v1 directory:
```env
# Data paths
DATA_CLEAN_PATH=../data/processed/hadits_docs.json
KEYWORDS_MAP_PATH=data/enhanced_index_v1/enhanced_keywords_map_v1.json
EMBEDDINGS_PATH=data/enhanced_index_v1/enhanced_embeddings_v1.pkl

# API settings
API_HOST=127.0.0.1
API_PORT=5000
DEBUG_MODE=True

# Processing settings
BATCH_SIZE=32
MIN_KEYWORD_FREQUENCY=40
MAX_RESULTS_DEFAULT=5

# Logging
LOG_LEVEL=INFO
ANALYTICS_ENABLED=True
```

### Service Configuration
Customize the service behavior:
```python
from service import ServiceConfig

config = ServiceConfig(
    # Retrieval settings
    retrieval_config=RetrievalConfig(
        top_k=50,
        min_score_threshold=0.1,
        semantic_weight=0.7,
        keyword_weight=0.2,
        literal_overlap_weight=0.1
    ),
    
    # Response settings
    max_results_display=5,
    result_text_preview_length=200,
    include_scores=False,
    
    # Session settings
    enable_sessions=True,
    session_timeout_minutes=30,
    
    # Quality settings
    min_confidence_threshold=0.1,
    high_confidence_threshold=0.6
)
```

## 🧪 Testing and Validation

### Comprehensive Testing Script
Create `test_complete_system.py`:
```python
#!/usr/bin/env python3
"""Complete system test script."""

import sys
import json
from service import create_hadith_ai_service, ServiceConfig

def test_complete_system():
    """Test the complete system end-to-end."""
    
    # Test queries in Indonesian
    test_queries = [
        "apa hukum shalat jumat bagi wanita?",
        "bagaimana cara berwudhu yang benar?",
        "hukum minuman keras dalam islam",
        "zakat fitrah dan zakat mal",
        "puasa ramadan bagi muslimah",
        "berbakti kepada orang tua",
        "hukum riba dalam jual beli"
    ]
    
    print("🧪 Testing Enhanced Hadith AI System")
    print("=" * 50)
    
    try:
        # Initialize service
        config = ServiceConfig(max_results_display=3, enable_sessions=True)
        service = create_hadith_ai_service(config)
        
        # Create session
        session_id = service.create_session()
        print(f"✅ Session created: {session_id}")
        
        # Test queries
        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}] Query: {query}")
            
            response = service.process_query(query, session_id, max_results=3)
            
            if response.success:
                print(f"✅ Success: {len(response.results)} results")
                print(f"⏱️  Response time: {response.response_time_ms:.1f}ms")
                
                # Show first result
                if response.results:
                    first_result = response.results[0]
                    text = first_result.document.get('terjemah', '')
                    print(f"📖 Top result: {text[:100]}...")
                    print(f"🔍 Keywords: {', '.join(first_result.matched_keywords[:3])}")
            else:
                print(f"❌ Failed: {response.message}")
        
        # Test session stats
        stats = service.get_session_stats(session_id)
        print(f"\n📊 Session Stats:")
        print(f"   Queries: {stats['query_count']}")
        print(f"   Total results: {stats['total_results_returned']}")
        print(f"   Avg results per query: {stats['avg_results_per_query']:.1f}")
        
        # Test service health
        health = service.get_service_health()
        print(f"\n🏥 Service Health: {health['status']}")
        
        print(f"\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_complete_system()
    sys.exit(0 if success else 1)
```

Run the test:
```bash
python test_complete_system.py
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. **Import Errors**
```
ModuleNotFoundError: No module named 'xxx'
```
**Solution**: Ensure PYTHONPATH is set and all dependencies are installed
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pip install missing-package
```

#### 2. **Memory Errors During Embedding**
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU
```python
config = EmbeddingConfig(batch_size=16, use_gpu=False)
```

#### 3. **FAISS Not Available**
```
ImportError: No module named 'faiss'
```
**Solution**: Install FAISS or disable it
```bash
pip install faiss-cpu
# OR set use_faiss=False in configuration
```

#### 4. **Empty Results**
```
No results found for query
```
**Solution**: Check min_score_threshold and keywords map
```python
config.min_score_threshold = 0.05  # Lower threshold
config.auto_adaptive_min_match = True  # Enable adaptive filtering
```

#### 5. **Slow Performance**
```
Response takes >5 seconds
```
**Solution**: Optimize configuration
```python
config = RetrievalConfig(
    top_k=20,  # Reduce from 50
    enable_reranking=False,  # Disable for speed
    faiss_nprobe=5  # Reduce search scope
)
```

### Performance Optimization

#### For Large Datasets (>50k documents):
```python
config = IndexingConfig(
    embedding_batch_size=64,
    faiss_index_type="IVFFlat",
    faiss_nlist=200,
    chunk_size=2000
)
```

#### For Limited Memory (<8GB RAM):
```python
config = EmbeddingConfig(
    batch_size=16,
    use_gpu=False,
    normalize_embeddings=True
)
```

#### For Production Deployment:
```python
config = ServiceConfig(
    enable_analytics=True,
    session_timeout_minutes=60,
    max_results_display=5,
    min_confidence_threshold=0.15
)
```

## 📁 File Structure After Setup

After successful setup, your directory should look like:
```
fixed-v1/
├── data/enhanced_index_v1/
│   ├── enhanced_keywords_map_v1.json
│   ├── enhanced_embeddings_v1.pkl
│   ├── enhanced_faiss_index_v1.index
│   └── enhanced_metadata_v1.pkl
├── logs/
│   └── hadith_ai_analytics.jsonl
├── extraction/...
├── preprocessing/...
├── embedding/...
├── retrieval/...
├── indexing/...
├── service/...
└── docs/...
```

## ✅ Verification Checklist

- [ ] All dependencies installed without errors
- [ ] Data files present and properly formatted
- [ ] Keyword extraction runs successfully
- [ ] Embeddings generated (check file size ~100-500MB)
- [ ] FAISS index built (or graceful fallback)
- [ ] Individual component tests pass
- [ ] Complete system test passes
- [ ] API server starts without errors
- [ ] Health check returns "healthy" status
- [ ] Sample queries return meaningful results

## 🎯 Next Steps

1. **Customize Configuration**: Adjust parameters for your specific use case
2. **Integrate with Frontend**: Use the API endpoints for web/mobile integration
3. **Monitor Performance**: Check analytics logs for optimization opportunities
4. **Scale for Production**: Consider deployment options and scaling strategies

Congratulations! Your Enhanced Hadith AI Fixed V1 system is now ready for use. 🎉
