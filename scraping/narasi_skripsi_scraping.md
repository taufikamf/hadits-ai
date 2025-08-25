# NARASI MODUL SCRAPING HADITS UNTUK PENULISAN SKRIPSI

## 1. PENDAHULUAN MODUL

Modul scraping hadits merupakan komponen fundamental dalam sistem pengembangan AI untuk analisis hadits. Modul ini dirancang untuk melakukan ekstraksi data hadits secara otomatis dari situs web hadits.id, yang merupakan sumber data hadits terpercaya dalam bahasa Indonesia. Tujuan utama modul ini adalah mengumpulkan dataset hadits yang terstruktur untuk keperluan training dan testing sistem AI.

### 1.1 Tujuan Pengembangan
- Mengumpulkan dataset hadits yang komprehensif dari berbagai koleksi
- Memastikan kualitas data melalui validasi struktur HTML
- Menyediakan data dalam format yang siap untuk preprocessing
- Mendukung skalabilitas pengumpulan data untuk koleksi besar

### 1.2 Dependencies dan Teknologi
Modul ini menggunakan beberapa library Python yang telah teruji:
- **requests**: Untuk melakukan HTTP request ke server hadits.id
- **BeautifulSoup**: Untuk parsing HTML dan ekstraksi konten
- **pandas**: Untuk manipulasi data dan export ke format CSV

## 2. ARSITEKTUR MODUL

### 2.1 Struktur Fungsi Utama

#### Fungsi `scrape_hadith(collection, number)`
Fungsi ini merupakan jantung dari modul scraping yang melakukan ekstraksi data hadits individual. Fungsi ini menerima dua parameter:
- `collection`: String yang menentukan koleksi hadits (contoh: 'bukhari', 'muslim')
- `number`: Integer yang menentukan nomor urut hadits dalam koleksi

**Return Value:**
- Dictionary berisi data hadits dengan struktur: `{'id': number, 'kitab': collection, 'arab': arabic_text, 'terjemah': translation_text}`
- `None` jika terjadi error atau hadits tidak ditemukan

### 2.2 Alur Proses Ekstraksi

#### Tahap 1: URL Construction
```python
url = f"https://www.hadits.id/hadits/{collection}/{number}"
```
Membangun URL target berdasarkan koleksi dan nomor hadits. URL ini mengikuti pola yang konsisten dari situs hadits.id.

#### Tahap 2: HTTP Request dan Error Handling
```python
try:
    response = requests.get(url)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    return None
```
Melakukan HTTP GET request dengan implementasi robust error handling untuk menangani berbagai kemungkinan error seperti network issues, HTTP error codes, dan timeout.

#### Tahap 3: HTML Parsing dan Content Location
```python
soup = BeautifulSoup(response.text, 'html.parser')
hadith_content = soup.find('article', class_='hadits-content')
```
Menggunakan BeautifulSoup untuk parse HTML response dan mencari elemen yang mengandung konten hadits berdasarkan struktur HTML yang telah dianalisis.

#### Tahap 4: Ekstraksi Teks Arab
```python
arabic_text_tag = hadith_content.find('p', class_='rtl')
arabic_text = arabic_text_tag.get_text(strip=True) if arabic_text_tag else "Teks Arab Tidak Ditemukan"
```
Mengekstrak teks Arab hadits dari elemen dengan class 'rtl' (right-to-left). Teks Arab ditampilkan dengan font khusus dan alignment kanan-kiri.

#### Tahap 5: Ekstraksi Terjemahan
```python
terjemah_text_tag = arabic_text_tag.find_next_sibling('p')
terjemah_text = terjemah_text_tag.get_text(strip=True) if terjemah_text_tag else "Terjemahan Tidak Ditemukan"
```
Mengekstrak terjemahan Indonesia yang biasanya berada di elemen `<p>` yang mengikuti teks Arab menggunakan `find_next_sibling()`.

#### Tahap 6: Struktur Data Output
```python
hadith_data = {
    'id': number,
    'kitab': collection,
    'arab': arabic_text,
    'terjemah': terjemah_text,
}
```
Menyusun data yang telah diekstrak dalam format dictionary yang konsisten untuk memudahkan processing lanjutan.

## 3. BATCH PROCESSING DAN DATA MANAGEMENT

### 3.1 Konfigurasi Parameter
```python
koleksi_hadits = 'bukhari'
max_hadith_number = 50
```
Parameter yang dapat disesuaikan untuk mengontrol proses scraping:
- `koleksi_hadits`: Menentukan koleksi hadits yang akan di-scrape
- `max_hadith_number`: Menentukan jumlah maksimal hadits yang akan diambil

### 3.2 Loop Processing dan Data Collection
```python
hadits_list = []
for i in range(1, max_hadith_number + 1):
    hadith = scrape_hadith(koleksi_hadits, i)
    if hadith:
        hadits_list.append(hadith)
    else:
        break
```
Melakukan iterasi untuk setiap nomor hadits dalam range yang ditentukan. Implementasi early stopping jika hadits tidak ditemukan untuk menghemat waktu dan resources.

### 3.3 Output Processing dan File Export
```python
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

df = pd.DataFrame(hadits_list)
output_csv_path = os.path.join(output_folder, f"{koleksi_hadits}_hadits.csv")
df.to_csv(output_csv_path, index=False)
```
Menyimpan hasil scraping ke file CSV untuk analisis lanjutan dengan implementasi automatic directory creation.

## 4. FITUR KEAMANAN DAN ROBUSTNESS

### 4.1 Error Handling
Modul ini mengimplementasikan comprehensive error handling untuk menangani:
- Network connectivity issues
- HTTP error responses (404, 500, dll)
- Timeout issues
- HTML structure changes
- Missing content elements

### 4.2 Data Validation
- Validasi keberadaan container utama konten hadits
- Validasi keberadaan teks Arab dan terjemahan
- Fallback values untuk konten yang tidak ditemukan

### 4.3 Scalability Features
- Configurable parameters untuk jumlah hadits
- Early stopping mechanism
- Batch processing capability
- Automatic file management

## 5. INTEGRASI DENGAN SISTEM AI

### 5.1 Data Format Standardization
Data yang dihasilkan modul ini telah distandarisasi dalam format yang konsisten:
- `id`: Identifier unik untuk setiap hadits
- `kitab`: Klasifikasi koleksi hadits
- `arab`: Teks Arab asli hadits
- `terjemah`: Terjemahan Indonesia

### 5.2 Preprocessing Readiness
Data yang dihasilkan siap untuk preprocessing lanjutan:
- Text normalization
- Tokenization
- Embedding generation
- Feature extraction

## 6. KESIMPULAN

Modul scraping hadits merupakan komponen penting dalam pipeline pengembangan AI untuk analisis hadits. Modul ini berhasil mengatasi tantangan teknis dalam ekstraksi data dari website dinamis dengan implementasi robust error handling dan data validation. Hasil output yang terstruktur memungkinkan integrasi yang seamless dengan komponen AI lainnya dalam sistem.

### 6.1 Kontribusi terhadap Sistem AI
- Menyediakan dataset berkualitas tinggi untuk training
- Memastikan konsistensi format data
- Mendukung skalabilitas pengumpulan data
- Memungkinkan analisis multi-koleksi hadits

### 6.2 Potensi Pengembangan
- Implementasi rate limiting untuk menghormati server
- Penambahan metadata hadits (perawi, sanad, dll)
- Integrasi dengan database untuk caching
- Implementasi parallel processing untuk efisiensi 