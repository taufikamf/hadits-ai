# Hadits-AI: Chatbot Pencari Hadits berbasis RAG

Sistem chatbot pencari hadits berbasis Retrieval-Augmented Generation (RAG) yang menggunakan LLM Gemini dan vector database (ChromaDB/FAISS).

## 🌟 Fitur

- Pencarian hadits berbasis semantic similarity
- Preprocessing teks Arab dan Indonesia
- Embedding menggunakan model SentenceTransformers atau OpenAI
- Vector database dengan ChromaDB (default) atau FAISS
- Integrasi dengan Google Gemini untuk generasi jawaban
- API endpoint untuk tanya jawab hadits
- Caching embedding untuk performa lebih baik

## 🛠️ Teknologi

- FastAPI - Web framework
- SentenceTransformers - Model embedding (default: all-MiniLM-L6-v2)
- ChromaDB/FAISS - Vector database
- Google Gemini - Large Language Model
- Pandas - Data processing
- Pydantic - Data validation

## 📋 Prasyarat

- Python 3.8+
- Google Gemini API key
- (Opsional) OpenAI API key jika menggunakan embedding OpenAI

## 🚀 Instalasi

1. Clone repository:
```bash
git clone https://github.com/yourusername/hadits-ai.git
cd hadits-ai
```

2. Buat virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Setup environment variables:
```bash
cp .env.example .env
# Edit .env dengan konfigurasi Anda
```

## 💾 Dataset

Dataset input harus dalam format CSV dengan kolom berikut:
- `id`: ID unik hadits
- `kitab`: Nama kitab hadits
- `arab`: Teks hadits dalam bahasa Arab
- `terjemah`: Terjemahan hadits dalam bahasa Indonesia

Contoh format:
```csv
id,kitab,arab,terjemah
1,Shahih Bukhari,إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ,Sesungguhnya setiap perbuatan tergantung niatnya
```

## 🏃‍♂️ Menjalankan Aplikasi

1. Pastikan virtual environment aktif dan `.env` sudah dikonfigurasi

2. Jalankan aplikasi:
```bash
python main.py
```

3. Akses API di http://localhost:8000
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 🔄 Alur Sistem

1. **Preprocessing Data**
   - Normalisasi teks Arab (hapus harakat)
   - Pembersihan teks Indonesia
   - Konversi ke format JSON dokumen

2. **Embedding & Indexing**
   - Generate embedding dari teks terjemahan
   - Simpan dalam vector database
   - Index otomatis dibuat saat startup jika belum ada

3. **Pencarian & Generasi**
   - Query → embedding → semantic search
   - Retrieve top-k hadits relevan
   - Generate jawaban dengan Gemini LLM

## 📡 API Endpoints

### GET /api/v1/ask
```
/api/v1/ask?q=Apa pentingnya niat dalam Islam?
```

Response:
```json
{
  "query": "Apa pentingnya niat dalam Islam?",
  "answer": "...",
  "retrieved_hadits": [
    {
      "id": "1",
      "kitab": "Shahih Bukhari",
      "arab": "...",
      "terjemah": "...",
      "score": 0.95
    }
  ],
  "total_results": 1,
  "processing_time_ms": 150.5
}
```

### POST /api/v1/index
Rebuild vector index dari dataset

### GET /api/v1/info
Informasi dataset dan index

### GET /api/v1/health
Health check endpoint

## ⚙️ Konfigurasi

Semua konfigurasi dapat diatur melalui environment variables (`.env`):

- `GEMINI_API_KEY`: API key untuk Google Gemini
- `EMBEDDING_MODEL`: Model embedding yang digunakan
- `VECTOR_DB`: Pilihan vector database (chroma/faiss)
- `MAX_RETRIEVAL_RESULTS`: Jumlah maksimum hadits yang di-retrieve
- `SCORE_THRESHOLD`: Threshold skor relevansi minimum

## 🔍 Troubleshooting

1. **Embedding Error**
   - Pastikan model tersedia dan terinstall
   - Cek koneksi internet untuk download model
   - Gunakan model alternatif di `.env`

2. **Vector DB Error**
   - Cek permission direktori penyimpanan
   - Hapus dan rebuild index jika corrupt
   - Switch ke ChromaDB jika FAISS bermasalah

3. **LLM Error**
   - Validasi API key Gemini
   - Cek format prompt dan response
   - Periksa log untuk detail error

## 📝 License

MIT License - lihat file [LICENSE](LICENSE) untuk detail 