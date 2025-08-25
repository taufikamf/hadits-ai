# Hadits AI

Sistem pencarian dan tanya jawab berbasis hadits dengan pendekatan RAG (Retrieval Augmented Generation).

## 🚀 Panduan Penggunaan

### Prasyarat

- Python 3.8+
- Pip
- Virtual Environment (opsional tapi direkomendasikan)

### Instalasi

1. Clone repository ini
2. Buat dan aktifkan virtual environment (opsional)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # atau
   venv\Scripts\activate  # Windows
   ```
3. Install dependensi
   ```bash
   pip install -r requirements.txt
   ```

### Menyiapkan Dataset

1. Pastikan semua file CSV hadits berada di folder `data/csv/`:
   - shahih_bukhari.csv
   - shahih_muslim.csv
   - sunan_abu_daud.csv
   - sunan_tirmidzi.csv
   - sunan_nasai.csv
   - sunan_ibnu_majah.csv

2. Format CSV harus memiliki kolom: `id`, `kitab`, `arab`, `terjemah`

### Memproses Dataset

Jalankan langkah-langkah berikut secara berurutan:

1. **Preprocessing Data**
   ```bash
   python preprocessing/normalize.py
   ```
   Ini akan membaca semua file CSV dan menghasilkan file `data/processed/hadits_docs.json`

2. **Ekstraksi Keyword**
   ```bash
   python utils/improved_keyword_extractor.py
   ```
   Ini akan menghasilkan file `data/processed/keywords_map_grouped.json`

3. **Pembersihan Keyword** (opsional)
   ```bash
   python utils/clean_keywords_map.py
   ```
   Ini akan menghasilkan file `data/processed/keywords_map_grouped_cleaned.json`

4. **Embedding Dokumen**
   ```bash
   python embedding/embed_model.py
   ```
   Ini akan menghasilkan file `data/processed/hadits_embeddings.pkl`

5. **Pengindeksan**
   ```bash
   python indexing/build_index.py
   ```
   Ini akan membuat database vektor di `db/hadits_index/`

### Menjalankan Server

```bash
python main.py
```

Server akan berjalan di `http://localhost:8000`

### Mengakses API

- **Dokumentasi API**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Endpoint Utama**: `http://localhost:8000/ask` (POST)

### Contoh Penggunaan API

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "hukum shalat jumat"}'
```

## 📋 Struktur Proyek

- **api/**: Definisi endpoint API
- **data/**: Dataset dan file hasil pemrosesan
  - **csv/**: File CSV hadits
  - **processed/**: File hasil pemrosesan
  - **raw/**: File mentah
  - **sessions/**: Data sesi chat
- **db/**: Database vektor ChromaDB
- **embedding/**: Kode untuk membuat embedding dokumen
- **indexing/**: Kode untuk mengindeks dokumen
- **llm/**: Integrasi dengan model bahasa
- **preprocessing/**: Kode untuk preprocessing data
- **retriever/**: Kode untuk pencarian dokumen
- **utils/**: Utilitas dan alat bantu

## 🔧 Konfigurasi

Konfigurasi utama dapat diatur melalui variabel lingkungan atau file `.env`:

```
GEMINI_API_KEY=your_api_key
CHROMA_DB_PATH=./db/hadits_index
```

## 📚 Informasi Tambahan

Untuk detail lebih lanjut tentang ekstraksi keyword dan komponen sistem, lihat [utils/README.md](utils/README.md).
