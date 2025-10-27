# Enhanced Islamic Keyword Extractor - Technical Flow Documentation

## 🏗️ Architecture Overview

The Enhanced Islamic Keyword Extractor follows a multi-stage pipeline architecture designed specifically for processing Islamic hadith texts. The system combines statistical analysis with rule-based Islamic knowledge to produce high-quality keyword extractions.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Input     │───▶│  Text Processing │───▶│ Keyword Extract │
│   (Hadith Data) │    │  & Normalization │    │ (Stat + Rules)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  JSON Output    │◀───│ Result Packaging │◀───│ Semantic Group  │
│  (Keywords Map) │    │   & Metadata     │    │ & Categorization│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📊 Processing Pipeline

### Stage 1: Data Loading & Preprocessing

```python
def load_csv_texts(csv_dir: Path) -> List[str]
```

**Input**: CSV files in `data/csv/` directory
**Process**:
1. **File Discovery**: Scan for all `.csv` files
2. **Column Validation**: Ensure `terjemah` column exists  
3. **Data Cleaning**: Remove null/empty entries
4. **Text Aggregation**: Combine all hadith translations

**Output**: List of raw hadith texts

### Stage 2: Advanced Text Normalization

```python
def normalize_text(self, text: str) -> str
```

**Multi-layer Cleaning Process**:

#### Layer 1: Basic Normalization
```python
text = text.lower().strip()
text = re.sub(r'[""\"\'\'``]', '', text)  # Remove quotes
text = re.sub(r'\s*[-–—]\s*', ' ', text)  # Normalize dashes
```

#### Layer 2: Islamic Honorifics Removal  
```python
text = re.sub(r'shallallahu\s+[\'\"]*alaihi\s+wa?\s*sallam', '', text)
text = re.sub(r'radhi\s*allahu\s+(anhu|anha|anhum)', '', text)
text = re.sub(r'rahimahullah\s*', '', text)
```

#### Layer 3: Sanad (Chain) Removal
```python
# Remove transmission chain indicators
text = re.sub(r'\b(telah\s+)?(menceritakan|mengabarkan|memberitahukan|bercerita)\s+(kepada\s+)?(kami|ku|nya)\b', '', text)
text = re.sub(r'\b(dari|kepada)\s+([a-z]+\s+)?(bin|abu|ibnu)\s+[a-z]+\b', '', text)
```

#### Layer 4: Arabic Script & Final Cleanup
```python
text = re.sub(r'[ء-ي]+', ' ', text)  # Remove Arabic characters
text = re.sub(r'[^\w\s\'-]', ' ', text)  # Clean punctuation
text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
```

### Stage 3: Enhanced N-gram Generation

```python
def generate_enhanced_ngrams(self, text: str) -> List[str]
```

**Process Flow**:
```
Input Text ─┐
           ├──▶ Word Tokenization ──▶ N-gram Generation (1-3) ──▶ Meaningfulness Filter ──▶ Output
           └──▶ Islamic Context Check ───────────────────────────────────────────────────────┘
```

**Filtering Criteria**:
1. **Stopword Filter**: Remove common Indonesian stopwords
2. **Noise Pattern Filter**: Apply 15+ regex patterns for narrator chains
3. **Islamic Term Boost**: Prioritize terms containing Islamic concepts
4. **Length Filter**: Remove very short terms (< 3 chars)

### Stage 4: Dual Extraction Methods

#### Statistical Extraction
```python
def extract_statistical_keywords(self, texts: List[str]) -> Counter
```

**Algorithm**:
```
For each text in corpus:
    1. Normalize text using enhanced normalization
    2. Generate n-grams with Islamic context awareness
    3. Count frequencies across entire corpus
    4. Apply minimum frequency threshold (default: 15)
    5. Return Counter with term frequencies
```

**Optimization**:
- Batch processing with progress logging
- Memory-efficient streaming for large datasets
- Early termination for very frequent terms

#### Rule-based Extraction  
```python
def extract_rule_based_keywords(self, texts: List[str]) -> Set[str]
```

**Islamic Terms Database** (145+ terms):
```python
islamic_terms = {
    # Worship practices
    'shalat', 'puasa', 'zakat', 'haji', 'umrah', 'doa', 'dzikir',
    
    # Purification  
    'wudhu', 'tayammum', 'ghusl', 'bersuci', 'najis', 'suci',
    
    # Legal concepts
    'halal', 'haram', 'wajib', 'sunnah', 'makruh', 'mubah',
    
    # Spiritual concepts
    'iman', 'islam', 'tauhid', 'taqwa', 'sabar', 'syukur',
    
    # ... and more categories
}
```

**Pattern Matching**:
```
For each text in corpus:
    1. Normalize text
    2. Convert to lowercase for matching
    3. Check presence of each Islamic term
    4. Collect all found terms
    5. Return unique set
```

### Stage 5: Semantic Grouping & Categorization

```python
def create_enhanced_keywords_map(self, texts: List[str]) -> Dict[str, List[str]]
```

**Two-Phase Grouping**:

#### Phase 1: Islamic Category Mapping
```python
islamic_categories = {
    'worship': ['shalat', 'puasa', 'zakat', 'haji', 'umrah'],
    'purification': ['wudhu', 'tayammum', 'ghusl', 'bersuci'],
    'ethics': ['akhlak', 'adab', 'sabar', 'syukur', 'ikhlas'],
    'legal': ['halal', 'haram', 'wajib', 'sunnah', 'makruh'],
    'belief': ['iman', 'islam', 'tauhid', 'syirik', 'kufur'],
    'social': ['nikah', 'jual_beli', 'sedekah', 'silaturahmi'],
    'eschatology': ['akhirat', 'surga', 'neraka', 'kiamat']
}
```

**Grouping Algorithm**:
```
For each Islamic category:
    For each term in all_extracted_terms:
        For each category_keyword:
            If category_keyword in term.lower():
                Add term to category group
                Mark term as categorized
```

#### Phase 2: High-Frequency Standalone Terms
```python
remaining_terms = [term for term in statistical_terms 
                  if term not in categorized]
remaining_terms.sort(key=lambda x: frequency[x], reverse=True)

# Add top frequency terms as standalone groups
for term in remaining_terms[:50]:
    if frequency[term] >= min_frequency * 2:
        grouped[term] = [term]
```

### Stage 6: Result Packaging & Output

```python
def save_results(grouped_terms, output_path, min_freq, max_ngram)
```

**Output Structure**:
```json
{
  "metadata": {
    "description": "Enhanced Islamic keywords extracted from hadits collections",
    "extraction_version": "enhanced_v2.0",
    "min_frequency": 15,
    "max_ngram": 3,
    "total_groups": 150,
    "extraction_method": "enhanced_islamic_semantic_grouping",
    "features": [
      "sanad_noise_removal",
      "islamic_terminology_detection", 
      "semantic_grouping",
      "literal_overlap_boosting"
    ]
  },
  "keywords": {
    "group_name": ["term1", "term2", "term3", ...]
  }
}
```

## 🧮 Algorithm Complexity

### Time Complexity
- **Text Loading**: O(n) where n = total characters
- **Normalization**: O(m) where m = number of texts
- **N-gram Generation**: O(k × w²) where k = texts, w = average words
- **Statistical Extraction**: O(t) where t = total n-grams
- **Rule-based Extraction**: O(m × i) where i = Islamic terms count
- **Semantic Grouping**: O(g × c) where g = groups, c = categories

**Overall Complexity**: O(k × w² + t + m × i + g × c)

### Space Complexity
- **Text Storage**: O(n) for raw texts
- **N-gram Storage**: O(t) for all n-grams  
- **Counter Storage**: O(u) where u = unique terms
- **Final Output**: O(g × a) where a = average group size

**Peak Memory**: O(n + t + u + g × a)

## 🚀 Performance Optimizations

### 1. Streaming Processing
```python
# Process texts in chunks to manage memory
for i, text in enumerate(texts):
    if i % 1000 == 0:
        logger.info(f"Processing text {i+1}/{len(texts)}")
    # Process individual text
```

### 2. Early Termination
```python
# Stop processing when enough candidates found
if len(candidates) >= top_k * 2:
    break
```

### 3. Efficient Data Structures
- **Counter**: For frequency counting with O(1) access
- **Set**: For fast membership testing of Islamic terms
- **defaultdict**: For automatic group initialization

### 4. Regex Compilation
```python
# Pre-compile regex patterns for better performance
self.compiled_patterns = [re.compile(pattern) for pattern in noise_patterns]
```

## 🔧 Configuration Parameters

### Tunable Parameters
```python
class EnhancedIslamicKeywordExtractor:
    def __init__(self, 
                 min_frequency: int = 15,      # Frequency threshold
                 max_ngram: int = 3,           # N-gram size limit
                 islamic_terms_path: str = None # Custom Islamic terms
                ):
```

### Performance vs Quality Trade-offs

| Parameter | Low Value | High Value | Impact |
|-----------|-----------|------------|---------|
| `min_frequency` | More terms, more noise | Fewer terms, higher quality | Quality vs Coverage |
| `max_ngram` | Faster processing | Better context capture | Speed vs Accuracy |
| Batch size | Lower memory usage | Faster processing | Memory vs Speed |

## 🔍 Quality Assurance

### Validation Pipeline
1. **Syntax Validation**: Ensure code compiles correctly
2. **Unit Testing**: Test individual components
3. **Integration Testing**: Test full pipeline with sample data
4. **Performance Testing**: Benchmark against original extractor
5. **Quality Validation**: Compare output with manually cleaned data

### Metrics Tracking
```python
def log_query_results(query, keywords, results, processing_time):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "keywords": keywords,
        "results_count": len(results),
        "processing_time_ms": processing_time
    }
```

## 🔄 Integration Points

### 1. Query Runner Integration
```python
# In retriever/query_runner.py
from utils.improved_keyword_extractor import EnhancedIslamicKeywordExtractor

def query_hadits_return(raw_query, ...):
    if not required_keywords and hasattr(query_optimizer, 'optimize_query'):
        # Use enhanced extractor for keyword extraction
        extractor = EnhancedIslamicKeywordExtractor(min_frequency=1)
        extracted_keywords = extractor.extract_rule_based_keywords([raw_query])
```

### 2. Embedding Model Integration
```python
# Can be used to improve embedding quality by filtering noise
normalized_text = extractor.normalize_text(hadith_text)
embedding = model.encode(normalized_text)
```

### 3. Data Preprocessing Pipeline
```python
# Preprocess CSV data before indexing
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['terjemah_normalized'] = df['terjemah'].apply(extractor.normalize_text)
```

## 📊 Monitoring & Debugging

### Logging Levels
```python
# INFO: General processing information
# DEBUG: Detailed step-by-step processing  
# WARNING: Non-critical issues (missing dependencies)
# ERROR: Critical failures requiring attention
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed processing information
extractor = EnhancedIslamicKeywordExtractor()
```

### Performance Profiling
```python
import time
start_time = time.time()
# ... processing ...
processing_time = (time.time() - start_time) * 1000
logger.info(f"Processing completed in {processing_time:.2f}ms")
```

---

**Next**: See `API_REFERENCE.md` for detailed method documentation and parameters.