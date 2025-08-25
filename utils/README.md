# Hadits AI System

Sistem pencarian dan tanya jawab berbasis hadits dengan pendekatan RAG (Retrieval Augmented Generation).

## 🚀 Panduan Lengkap

### 1. Persiapan Dataset

Sistem ini menggunakan 6 koleksi hadits dalam format CSV:
- Shahih Bukhari
- Shahih Muslim
- Sunan Abu Daud
- Sunan Tirmidzi
- Sunan Nasai
- Sunan Ibnu Majah

Semua file CSV harus berada di direktori `data/csv/` dengan struktur kolom berikut:
```
id,kitab,arab,terjemah
```

### 2. Preprocessing Data

Preprocessing mengubah data mentah menjadi format yang siap diproses:

```bash
# Modifikasi file normalize.py untuk memproses semua dataset CSV
python preprocessing/normalize.py
```

### 3. Ekstraksi Keyword

Ekstraksi keyword mengidentifikasi istilah penting dalam hadits:

```bash
# Menggunakan ekstraksi keyword yang ditingkatkan
python utils/improved_keyword_extractor.py
```

### 4. Embedding Dokumen

Mengubah teks hadits menjadi vektor numerik:

```bash
# Membuat embedding untuk semua dokumen hadits
python embedding/embed_model.py
```

### 5. Pengindeksan

Menyimpan embedding dalam database vektor:

```bash
# Mengindeks dokumen ke FAISS
python indexing/build_index.py
```

### 6. Menjalankan Layanan

Menjalankan API backend:

```bash
# Menjalankan server FastAPI
python main.py
```

## 📁 Struktur Sistem

### Preprocessing

1. **`preprocessing/normalize.py`** - Normalisasi teks Arab dan Indonesia
   - Membersihkan teks dari karakter khusus
   - Menyeragamkan format

### Ekstraksi Keyword

1. **`advanced_keyword_extractor.py`** - Ekstraksi keyword komprehensif
   - Memproses semua file CSV di `data/csv/`
   - Menghasilkan n-gram 1-3 kata
   - Mengelompokkan istilah serupa
   - Output: `data/processed/keywords_map_grouped.json`

2. **`improved_keyword_extractor.py`** - Versi yang ditingkatkan
   - Filter stopword yang lebih baik
   - Normalisasi frasa Islam
   - Kategorisasi semantik
   - Output dengan struktur metadata

3. **`run_keyword_extraction.py`** - Runner dengan parameter yang dapat dikonfigurasi

## 📊 Alur Kerja Lengkap

### 1. Persiapan Dataset

1. Letakkan semua 6 file CSV di direktori `data/csv/`
2. Pastikan setiap file memiliki format yang benar (id, kitab, arab, terjemah)

### 2. Preprocessing Data

Untuk memproses semua 6 dataset, modifikasi file `preprocessing/normalize.py` sebagai berikut:

```python
if __name__ == "__main__":
    # Daftar semua file CSV yang akan diproses
    csv_files = [
        "shahih_bukhari.csv",
        "shahih_muslim.csv",
        "sunan_abu_daud.csv",
        "sunan_tirmidzi.csv",
        "sunan_nasai.csv",
        "sunan_ibnu_majah.csv"
    ]
    
    # Inisialisasi list untuk menyimpan semua dokumen
    all_documents = []
    
    # Proses setiap file CSV
    for csv_file in csv_files:
        csv_path = f"data/csv/{csv_file}"
        print(f"[~] Memproses {csv_path}...")
        
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            doc = {
                "id": f"{row['kitab']}_{row['id']}",
                "kitab": row["kitab"],
                "arab_asli": row["arab"],
                "arab_bersih": normalize_arabic(row["arab"]),
                "terjemah": normalize_indo(row["terjemah"]),
            }
            all_documents.append(doc)
    
    # Simpan semua dokumen ke file JSON
    output_path = "data/processed/hadits_docs.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, ensure_ascii=False, indent=2)
    
    print(f"[✓] Preprocessed {len(all_documents)} documents saved to {output_path}")
```

### 3. Ekstraksi Keyword

```bash
# Ekstraksi keyword dasar
python utils/advanced_keyword_extractor.py

# Ekstraksi dengan parameter kustom
python utils/run_keyword_extraction.py --min-freq 20 --max-ngram 3

# Pembersihan keyword map (opsional)
python utils/clean_keywords_map.py
```

### 4. Embedding Dokumen

```bash
# Buat embedding untuk semua dokumen
python embedding/embed_model.py
```

### 5. Pengindeksan

```bash
# Indeks dokumen ke FAISS
python indexing/build_index.py
```

### 6. Menjalankan Layanan

```bash
# Jalankan server FastAPI
python main.py
```

## 📊 Format Data

### CSV File Structure
Sistem mengharapkan file CSV di `data/csv/` dengan struktur berikut:
```csv
id,kitab,arab,terjemah
1,shahih_bukhari,"Teks Arab di sini","Terjemahan Indonesia di sini"
```

Kolom yang diperlukan: `terjemah` (terjemahan Indonesia)

### Koleksi yang Didukung
- Shahih Bukhari
- Shahih Muslim  
- Sunan Abu Daud
- Sunan Tirmidzi
- Sunan Nasai
- Sunan Ibnu Majah

## 🔧 Konfigurasi

### Parameter Utama

- **`MIN_FREQUENCY`**: Minimum kemunculan istilah (default: 20)
- **`MAX_NGRAM`**: Ukuran n-gram maksimum (default: 3)
- **`CSV_DIR`**: Direktori berisi file CSV (default: "data/csv")
- **`OUTPUT_PATH`**: Path file JSON output
- **`MODEL_NAME`**: Model embedding yang digunakan (default: "intfloat/e5-small-v2")
- **`FAISS_INDEX_PATH`**: Path untuk FAISS index (default: "./db/hadits_index")
- **`METADATA_PATH`**: Path untuk metadata JSON (default: "./db/hadits_metadata.json")

### Stopwords

Sistem menyertakan stopwords bahasa Indonesia dan istilah khusus hadits:
- Kata umum bahasa Indonesia (`yang`, `dan`, `di`, dll.)
- Istilah perawi hadits (`bin`, `abu`, `ibnu`, dll.)
- Frasa Islam (`shallallahu`, `alaihi`, `wasallam`, dll.)

## 📈 Output Format

### Advanced Extractor Output
```json
{
  "shalat": [
    "ada shalat",
    "beliau shalat", 
    "mengerjakan shalat",
    "shalat maghrib",
    "waktu shalat"
  ],
  "zakat": [
    "membayar zakat",
    "menunaikan zakat", 
    "zakat fitrah"
  ]
}
```

### Improved Extractor Output
```json
{
  "metadata": {
    "description": "Islamic keywords extracted from hadits collections",
    "min_frequency": 20,
    "max_ngram": 3,
    "total_groups": 150,
    "extraction_method": "improved_semantic_grouping"
  },
  "keywords": {
    "shalat": ["shalat subuh", "mengerjakan shalat", "waktu shalat"],
    "puasa": ["puasa ramadhan", "berbuka puasa", "menjalankan puasa"]
  }
}
```

## 🕌 Islamic Semantic Categories

The improved extractor recognizes these Islamic concepts:

### Worship & Rituals
- **shalat**: Prayer and worship
- **puasa**: Fasting  
- **zakat**: Obligatory charity
- **haji**: Pilgrimage to Mecca
- **umrah**: Lesser pilgrimage

### Legal Concepts
- **halal**: Permissible in Islam
- **haram**: Forbidden in Islam
- **riba**: Usury/interest
- **zina**: Adultery/fornication

### Spiritual Concepts
- **iman**: Faith
- **ikhlas**: Sincerity
- **sabar**: Patience
- **taubat**: Repentance
- **jihad**: Struggle in the path of Allah

### Character & Morality
- **akhlak**: Character/morality
- **adab**: Etiquette
- **amanah**: Trust

## 📋 Processing Pipeline

1. **Text Loading**: Read all CSV files with `terjemah` column
2. **Normalization**: Clean and standardize text
   - Lowercase conversion
   - Remove excessive punctuation
   - Normalize Islamic phrases
   - Handle narrator chains
3. **N-gram Generation**: Create 1-3 word phrases
4. **Frequency Filtering**: Keep terms appearing ≥ min_frequency times
5. **Semantic Grouping**: Group related terms by:
   - Predefined Islamic categories
   - String similarity
   - Shared root words
6. **Output Generation**: Save structured JSON with metadata

## 🔍 Features

### Advanced Extractor
- ✅ Comprehensive n-gram generation
- ✅ String similarity grouping
- ✅ High recall (captures many variations)
- ⚠️ May include some noise from narrator chains

### Improved Extractor  
- ✅ Islamic terminology focused
- ✅ Better noise filtering
- ✅ Semantic categorization
- ✅ Metadata inclusion
- ✅ Clean, focused output

## 📊 Performance

**Sample Results** (30,845 hadits from 6 collections):
- **Processing time**: ~5-10 minutes
- **N-grams generated**: ~7.6M total
- **Frequent terms**: ~33K (frequency ≥ 20)
- **Final groups**: ~2.5K (advanced) / ~150 (improved)

## 🔍 Komponen Utama Sistem

### 1. Preprocessing (`preprocessing/`)
- **normalize.py**: Membersihkan dan menyeragamkan teks Arab dan Indonesia
- **normalize_comment.py**: Membersihkan komentar dan anotasi

### 2. Ekstraksi Keyword (`utils/`)
- **advanced_keyword_extractor.py**: Ekstraksi n-gram dan pengelompokan
- **improved_keyword_extractor.py**: Ekstraksi dengan pemahaman terminologi Islam
- **clean_keywords_map.py**: Membersihkan peta keyword

### 3. Embedding (`embedding/`)
- **embed_model.py**: Mengubah teks menjadi vektor dengan model bahasa

### 4. Pengindeksan (`indexing/`)
- **build_index.py**: Menyimpan embedding ke database vektor FAISS

### 5. Pencarian (`retriever/`)
- **query_runner.py**: Melakukan pencarian semantik dan filtering
- **evaluator.py**: Mengevaluasi kualitas hasil pencarian

### 6. API (`main.py`)
- Server FastAPI untuk layanan pencarian dan tanya jawab
- Integrasi dengan model LLM (Gemini)

## 🛠️ Penggunaan dalam Sistem Hadits AI

Peta keyword yang dihasilkan digunakan untuk:

1. **Optimasi Query**: Meningkatkan query pencarian dengan istilah terkait
2. **Pengindeksan Semantik**: Meningkatkan akurasi pencarian
3. **Klasifikasi Topik**: Mengkategorikan hadits berdasarkan tema
4. **Pencarian Kontekstual**: Menemukan konsep terkait

### Contoh Integrasi

```python
from utils.query_optimizer import optimize_query

# Query asli
query = "hukum riba"

# Ditingkatkan dengan keyword  
optimized = optimize_query(query)
# Output: "passage: hukum riba. Kata kunci penting: riba bertransaksi riba mengambil riba"
```

## 🔄 Pembaruan Rutin

Untuk menjaga peta keyword tetap aktual:

1. **Tambahkan file CSV baru** ke `data/csv/`
2. **Jalankan preprocessing** untuk semua koleksi
3. **Jalankan ekstraksi** dengan koleksi yang diperbarui
4. **Buat embedding** untuk dokumen yang diperbarui
5. **Perbarui indeks** dengan embedding baru

## 📝 Catatan

- Memproses koleksi besar (30K+ hadits) memerlukan memori yang signifikan
- Kualitas hasil bergantung pada konsistensi terjemahan
- Peninjauan manual direkomendasikan untuk aplikasi penting
- Pertimbangkan untuk menyesuaikan `MIN_FREQUENCY` berdasarkan ukuran koleksi

## 🤝 Kontribusi

Untuk meningkatkan sistem:

1. **Tambahkan kelompok semantik** untuk konsep Islam baru
2. **Tingkatkan daftar stopword** dengan istilah domain-spesifik
3. **Tingkatkan normalisasi** untuk pembersihan teks yang lebih baik
4. **Tambahkan validasi** untuk penilaian kualitas output