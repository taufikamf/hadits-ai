# Hadits-AI 🕌

**Retrieval-Augmented Generation (RAG) system untuk Q&A hadits berbasis AI**

Hadits-AI adalah backend independen yang meniru arsitektur dan kualitas dari Dify untuk sistem knowledge retrieval dan LLM-based chatbot, khusus dirancang untuk menjawab pertanyaan tentang hadits Islam.

## 🎯 Fitur Utama

- **📚 Knowledge Base Management**: Ingestion dan preprocessing dataset hadits dalam format CSV
- **🔍 Semantic Search**: Pencarian hadits berdasarkan makna menggunakan embedding multilingual
- **🤖 RAG Pipeline**: Integrasi retrieval dengan Google Gemini untuk jawaban yang akurat
- **⚡ FastAPI Backend**: REST API yang cepat dan scalable
- **🗄️ Vector Database**: ChromaDB untuk penyimpanan dan pencarian embedding
- **🌐 Multilingual Support**: Mendukung teks Arab dan Indonesia

## 🏗️ Arsitektur

Sistem ini mengikuti pola arsitektur Dify dengan komponen-komponen berikut:

```
Query → DatasetRetrieval → VectorSearch → Context → LLM → Response
```

### Struktur Project

```
hadits-ai/
├── data/                    # Dataset hadits
│   └── hadits.csv          # Contoh dataset
├── utils/                   # Text processing utilities  
│   └── text_processor.py   # Normalisasi Arab & cleaning Indonesia
├── embedding/               # Embedding services
│   └── embedding_service.py # Cache-enabled embedding dengan local/API models
├── retriever/               # Vector storage & retrieval
│   └── vector_store.py     # ChromaDB integration
├── llm/                     # LLM integration
│   └── gemini_service.py   # Google Gemini API service
├── data/                    # Data loading
│   └── data_loader.py      # CSV ingestion & processing
├── api/                     # FastAPI routes
│   └── routes.py           # REST API endpoints
├── config.py               # Configuration management
├── main.py                 # FastAPI application entry point
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- Google Gemini API key

### 2. Installation

```bash
# Clone atau copy project
cd hadits-ai

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### 3. Configuration

Edit file `.env`:

```env
# Google Gemini API (Required)
GEMINI_API_KEY=your_gemini_api_key_here

# Embedding Model (Optional - default menggunakan local model)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Optional: Gunakan OpenAI embedding
# EMBEDDING_MODEL=openai
# OPENAI_API_KEY=your_openai_api_key_here

# Vector Database
VECTOR_DB=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_data

# Dataset
DATASET_PATH=./data/hadits.csv
MAX_RETRIEVAL_RESULTS=5
SCORE_THRESHOLD=0.3
```

### 4. Persiapan Dataset

Pastikan file `data/hadits.csv` mengikuti format:

```csv
id,kitab,arab,terjemah
1,shahih_bukhari,"حَدَّثَنَا الْحُمَيْدِيُّ...","Telah menceritakan kepada kami..."
```

### 5. Run Application

```bash
python main.py
```

Server akan berjalan di `http://localhost:8000`

## 📡 API Endpoints

### 1. Ask Question (RAG Pipeline)

**Endpoint utama untuk Q&A hadits**

```bash
GET /api/v1/ask?q=Apa pentingnya niat dalam Islam?
```

**Response:**
```json
{
  "query": "Apa pentingnya niat dalam Islam?",
  "answer": "Berdasarkan hadits yang ditemukan, niat memiliki peran yang sangat penting...",
  "retrieved_hadits": [
    {
      "id": "1",
      "kitab": "shahih_bukhari", 
      "arab": "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ...",
      "terjemah": "Sesungguhnya setiap perbuatan tergantung niatnya...",
      "score": 0.95
    }
  ],
  "total_results": 1,
  "processing_time_ms": 245.5
}
```

### 2. Rebuild Index

**Re-index dataset hadits**

```bash
POST /api/v1/index
```

### 3. Dataset Info

**Informasi tentang dataset**

```bash
GET /api/v1/info
```

### 4. Health Check

```bash
GET /api/v1/health
```

## 🔧 Pipeline Details

### 1. Data Ingestion & Preprocessing

```python
# Text Processing (utils/text_processor.py)
- Normalisasi teks Arab: hapus harakat
- Cleaning terjemah Indonesia: hapus HTML, quotes, newlines  
- Output: arab_asli, arab_bersih, terjemah_bersih
```

### 2. Embedding & Indexing

```python
# Embedding Service (embedding/embedding_service.py)
- Local: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- API: OpenAI text-embedding-3-small
- Caching: Local pickle cache untuk performa
- Vector Store: ChromaDB dengan persistence
```

### 3. Semantic Retrieval

```python
# Vector Store (retriever/vector_store.py)
- Query embedding → Similarity search
- Top-k results dengan score threshold
- Metadata filtering berdasarkan kitab
```

### 4. LLM Integration

```python
# Gemini Service (llm/gemini_service.py)
- Context building dari retrieved hadits
- System prompt untuk hadits Q&A
- Response generation dengan rujukan
```

## 🧪 Testing

### Test Individual Components

```bash
# Test text processor
python utils/text_processor.py

# Test embedding service  
python embedding/embedding_service.py

# Test vector store
python retriever/vector_store.py

# Test LLM service
python llm/gemini_service.py

# Test data loader
python data/data_loader.py
```

### Test Full Pipeline

```bash
curl "http://localhost:8000/api/v1/ask?q=Bagaimana cara bersuci dalam Islam?"
```

## 📊 Performance Features

- **Embedding Caching**: Local cache untuk embedding yang sudah di-compute
- **Vector Persistence**: ChromaDB menyimpan index secara persisten
- **Batch Processing**: Embedding multiple documents sekaligus
- **Lazy Loading**: Services di-initialize on-demand

## 🔍 Customization

### Menambah Model Embedding

```python
# embedding/embedding_service.py
class CustomEmbeddingModel(BaseEmbeddingModel):
    def embed_documents(self, texts):
        # Your implementation
        pass
```

### Custom LLM Provider

```python  
# llm/custom_llm_service.py
class CustomLLMService:
    def generate_response(self, query, retrieved_docs):
        # Your implementation
        pass
```

### Extended Preprocessing

```python
# utils/text_processor.py  
class EnhancedTextProcessor(HaditsDocumentProcessor):
    def process_hadits_row(self, row):
        # Your enhanced processing
        pass
```

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Error**: Pastikan koneksi internet untuk download model pertama kali
2. **Gemini API Error**: Verifikasi API key dan quota
3. **ChromaDB Permission**: Pastikan directory writable untuk persistence
4. **Memory Issues**: Reduce batch size untuk dataset besar

### Debugging

```bash
# Enable debug mode
DEBUG=true python main.py

# Check logs
tail -f logs/hadits-ai.log
```

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Add tests for new features  
4. Submit pull request

## 📄 License

MIT License - lihat file LICENSE untuk detail.

## 🙏 Acknowledgments

- **Dify**: Inspirasi arsitektur dan patterns
- **ChromaDB**: Vector database solution
- **SentenceTransformers**: Multilingual embedding models
- **Google Gemini**: LLM capabilities

---

**Hadits-AI** - Bringing Islamic knowledge closer through AI 🌟 