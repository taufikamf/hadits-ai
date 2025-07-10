# Hadits-AI: Retrieval-Augmented Generation System

Sistem chatbot pencari hadis berbasis Retrieval-Augmented Generation (RAG) yang menggunakan LLM Gemini dan vector database ChromaDB.

## 🎯 Fitur Utama

- **Preprocessing Dataset**: Normalisasi teks Arab dan pembersihan teks Indonesia
- **Semantic Search**: Pencarian hadis berdasarkan makna menggunakan embedding
- **RAG Pipeline**: Kombinasi retrieval + LLM untuk jawaban yang akurat
- **API REST**: Endpoint `/ask` untuk query hadis
- **Vector Database**: ChromaDB dengan embedding built-in

## 🏗️ Arsitektur Sistem

```
CSV Dataset → Preprocessing → Embedding → Vector Index → RAG Pipeline → API Response
```

### Komponen Utama:
1. **Data Loader**: Load dan preprocess dataset CSV
2. **Text Processor**: Normalisasi teks Arab dan Indonesia  
3. **Embedding Service**: ChromaDB dengan embedding built-in
4. **Vector Store**: Index dan search semantic
5. **LLM Service**: Gemini untuk generate response
6. **API Routes**: FastAPI endpoints

## 📋 Requirements

- Python 3.10/3.11
- Google Gemini API Key
- Dataset CSV dengan format: `id`, `kitab`, `arab`, `terjemah`

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd hadits-ai

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi

```bash
# Copy environment template
cp .env.example .env

# Edit .env dengan API key Gemini
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Preprocessing Dataset

```bash
# Preprocess dataset CSV ke JSON
python preprocess_data.py
```

### 4. Build Index

```bash
# Build embedding index
python build_index_simple.py
```

### 5. Run Server

```bash
# Start API server
python simple_main.py
```

Server akan berjalan di `http://localhost:8000`

## 📡 API Endpoints

### Health Check
```bash
GET /api/v1/health
```

### Ask Question
```bash
GET /api/v1/ask?q=apa itu niat
```

Response:
```json
{
  "query": "apa itu niat",
  "answer": "Jawaban dari Gemini berdasarkan hadis yang ditemukan...",
  "retrieved_documents": [
    {
      "id": "1",
      "kitab": "shahih_bukhari", 
      "arab_asli": "حَدَّثَنَا الْحُمَيْدِيُّ...",
      "arab_bersih": "حدثنا الحميدي...",
      "terjemah": "Telah menceritakan kepada kami...",
      "score": 0.85
    }
  ],
  "total_results": 5,
  "processing_time_ms": 1250.5
}
```

## 🔧 Konfigurasi

### Environment Variables (.env)

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
VECTOR_DB=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_data
DATASET_PATH=./data/hadits.csv
MAX_RETRIEVAL_RESULTS=5
SCORE_THRESHOLD=0.3
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

## 📊 Dataset Format

Dataset harus dalam format CSV dengan kolom:

| Kolom | Deskripsi | Contoh |
|-------|-----------|---------|
| `id` | ID unik hadis | `1` |
| `kitab` | Nama kitab | `shahih_bukhari` |
| `arab` | Teks Arab asli | `حَدَّثَنَا الْحُمَيْدِيُّ...` |
| `terjemah` | Terjemahan Indonesia | `Telah menceritakan kepada kami...` |

## 🧪 Testing

### Test Preprocessing
```bash
python preprocess_data.py
```

### Test Index Building
```bash
python build_index_simple.py
```

### Test API
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Ask question
curl "http://localhost:8000/api/v1/ask?q=apa%20itu%20niat"
```

## � Struktur Project

```
hadits-ai/
├── api/
│   ├── routes.py              # Original API routes
│   └── simple_routes.py       # Simple API routes
├── data/
│   ├── data_loader.py         # Data loading & preprocessing
│   ├── hadits.csv            # Sample dataset
│   └── processed_hadits.json # Preprocessed data
├── embedding/
│   ├── embedding_service.py   # Original embedding service
│   └── simple_embedding.py    # Simple embedding with ChromaDB
├── llm/
│   └── gemini_service.py      # Gemini LLM integration
├── retriever/
│   ├── vector_store.py        # Original vector store
│   └── simple_vector_store.py # Simple vector store
├── utils/
│   └── text_processor.py      # Text preprocessing utilities
├── config.py                  # Configuration settings
├── main.py                    # Original FastAPI app
├── simple_main.py             # Simple FastAPI app
├── preprocess_data.py         # Data preprocessing script
├── build_index_simple.py      # Index building script
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## � Troubleshooting

### Error: "GEMINI_API_KEY required"
- Pastikan file `.env` sudah dibuat dengan API key yang valid
- Cek format: `GEMINI_API_KEY=your_key_here`

### Error: "No documents found"
- Pastikan file CSV dataset ada di `./data/hadits.csv`
- Jalankan `python preprocess_data.py` terlebih dahulu

### Error: "Import sentence_transformers failed"
- Project menggunakan ChromaDB built-in embedding sebagai fallback
- Gunakan `simple_main.py` dan `build_index_simple.py`

### Error: "Vector store not initialized"
- Jalankan `python build_index_simple.py` untuk build index
- Atau server akan auto-index saat startup

## 🚀 Deployment

### Local Development
```bash
python simple_main.py
```

### Production
```bash
# Set environment variables
export GEMINI_API_KEY=your_key
export DEBUG=false

# Run with gunicorn
pip install gunicorn
gunicorn simple_main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📈 Monitoring

- Health check: `GET /api/v1/health`
- Processing time tersedia di response API
- Logs tersedia di console dengan level INFO/DEBUG

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - lihat file LICENSE untuk detail.

## 🙏 Acknowledgments

- Google Gemini API untuk LLM
- ChromaDB untuk vector database
- FastAPI untuk web framework
- Sentence Transformers untuk embedding models 