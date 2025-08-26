# Enhanced Islamic Keyword Extractor - API Reference

## 📚 Table of Contents

- [Core Classes](#core-classes)
- [Main Functions](#main-functions)
- [Utility Functions](#utility-functions)
- [Configuration](#configuration)
- [Error Handling](#error-handling)
- [Examples](#examples)

## 🏛️ Core Classes

### EnhancedIslamicKeywordExtractor

The main class for enhanced Islamic keyword extraction with advanced noise filtering and semantic grouping.

#### Constructor

```python
class EnhancedIslamicKeywordExtractor:
    def __init__(self, 
                 min_frequency: int = 15,
                 max_ngram: int = 3,
                 islamic_terms_path: str = None)
```

**Parameters:**
- `min_frequency` (int): Minimum frequency threshold for statistical extraction. Default: 15
- `max_ngram` (int): Maximum n-gram length. Default: 3 (1, 2, 3-grams)
- `islamic_terms_path` (str, optional): Path to custom Islamic terms JSON file

**Returns:** EnhancedIslamicKeywordExtractor instance

**Example:**
```python
# Basic usage
extractor = EnhancedIslamicKeywordExtractor()

# Custom configuration
extractor = EnhancedIslamicKeywordExtractor(
    min_frequency=10,
    max_ngram=2,
    islamic_terms_path="custom_islamic_terms.json"
)
```

---

### Core Methods

#### normalize_text()

Advanced text normalization specifically designed for hadith texts.

```python
def normalize_text(self, text: str) -> str
```

**Parameters:**
- `text` (str): Raw hadith text to normalize

**Returns:** str - Normalized text with noise removed

**Process:**
1. Convert to lowercase and strip whitespace
2. Remove quotes, normalize punctuation
3. Remove Islamic honorifics (saw, ra, rah, etc.)
4. Remove Arabic script characters
5. Remove transmission chain indicators (sanad)
6. Remove narrator introduction patterns
7. Clean excessive punctuation

**Example:**
```python
raw_text = "Diriwayatkan dari Abu Hurairah ra bahwa Rasulullah saw bersabda..."
normalized = extractor.normalize_text(raw_text)
# Output: "diriwayatkan ra bahwa rasulullah bersabda..."
```

---

#### is_meaningful_term()

Check if a term is meaningful for Islamic keyword extraction.

```python
def is_meaningful_term(self, term: str) -> bool
```

**Parameters:**
- `term` (str): Term to evaluate

**Returns:** bool - True if term is meaningful, False otherwise

**Criteria:**
- Not all stopwords
- Longer than 2 characters  
- Not pure numbers
- Doesn't match noise patterns
- Gets boosted if contains Islamic terms

**Example:**
```python
extractor.is_meaningful_term("shalat berjamaah")  # True
extractor.is_meaningful_term("abu bin")           # False
extractor.is_meaningful_term("dari kepada")      # False
```

---

#### generate_enhanced_ngrams()

Generate meaningful n-grams with Islamic context awareness.

```python
def generate_enhanced_ngrams(self, text: str) -> List[str]
```

**Parameters:**
- `text` (str): Input text (should be normalized)

**Returns:** List[str] - List of meaningful n-grams

**Process:**
1. Tokenize text into words
2. Generate all n-grams (1 to max_ngram length)
3. Filter using `is_meaningful_term()`
4. Return list of valid n-grams

**Example:**
```python
text = "shalat berjamaah lebih baik"
ngrams = extractor.generate_enhanced_ngrams(text)
# Output: ["shalat", "berjamaah", "lebih", "baik", "shalat berjamaah", ...]
```

---

#### extract_statistical_keywords()

Extract keywords using statistical frequency analysis.

```python
def extract_statistical_keywords(self, texts: List[str]) -> Counter
```

**Parameters:**
- `texts` (List[str]): List of hadith texts to process

**Returns:** Counter - Terms with their frequencies

**Process:**
1. Process each text with progress logging
2. Normalize text using `normalize_text()`
3. Generate n-grams using `generate_enhanced_ngrams()`
4. Count frequencies across all texts
5. Filter by minimum frequency threshold

**Example:**
```python
texts = ["Text 1...", "Text 2...", "Text 3..."]
frequent_terms = extractor.extract_statistical_keywords(texts)
# Output: Counter({'shalat': 45, 'puasa': 32, 'zakat': 28, ...})
```

---

#### extract_rule_based_keywords()

Extract keywords using Islamic terminology pattern matching.

```python
def extract_rule_based_keywords(self, texts: List[str]) -> Set[str]
```

**Parameters:**
- `texts` (List[str]): List of hadith texts to scan

**Returns:** Set[str] - Set of found Islamic terms

**Process:**
1. Iterate through all texts with progress logging
2. Normalize each text
3. Check presence of each Islamic term from database
4. Collect all found terms in a set

**Example:**
```python
texts = ["Shalat adalah rukun islam", "Berpuasa di bulan ramadan"]
islamic_terms = extractor.extract_rule_based_keywords(texts)
# Output: {'shalat', 'islam', 'puasa', 'ramadan'}
```

---

#### enhanced_extract()

Combined extraction using both statistical and rule-based methods.

```python
def enhanced_extract(self, texts: List[str]) -> Tuple[Counter, Set[str]]
```

**Parameters:**
- `texts` (List[str]): List of hadith texts

**Returns:** Tuple[Counter, Set[str]] - (statistical_terms, rule_based_terms)

**Example:**
```python
texts = ["Hadith texts..."]
statistical_terms, rule_based_terms = extractor.enhanced_extract(texts)
```

---

#### create_enhanced_keywords_map()

Main method to create complete keywords map with semantic grouping.

```python
def create_enhanced_keywords_map(self, texts: List[str]) -> Dict[str, List[str]]
```

**Parameters:**
- `texts` (List[str]): List of hadith texts to process

**Returns:** Dict[str, List[str]] - Semantic groups mapping

**Process:**
1. Extract keywords using both methods
2. Combine statistical and rule-based results
3. Group by Islamic categories
4. Add high-frequency standalone terms
5. Clean and sort groups

**Example:**
```python
texts = load_csv_texts("data/csv")
keywords_map = extractor.create_enhanced_keywords_map(texts)
# Output: {
#   "shalat": ["shalat", "sholat", "shalat berjamaah", ...],
#   "puasa": ["puasa", "berpuasa", "shaum", ...],
#   ...
# }
```

---

## 🔧 Main Functions

### main()

Main extraction pipeline with full processing workflow.

```python
def main() -> None
```

**Process:**
1. Initialize EnhancedIslamicKeywordExtractor
2. Load CSV texts from `data/csv/`
3. Create enhanced keywords map
4. Save results to `data/processed/enhanced_keywords_map.json`
5. Log summary statistics

**Example:**
```python
from utils.improved_keyword_extractor import main
main()
```

---

### test_extractor()

Test function with sample hadith data for validation.

```python
def test_extractor() -> bool
```

**Returns:** bool - True if test passes, False otherwise

**Process:**
1. Create sample hadith texts
2. Test text normalization
3. Test keyword extraction
4. Log results and validate output

**Example:**
```python
from utils.improved_keyword_extractor import test_extractor
success = test_extractor()
```

---

## 🛠️ Utility Functions

### load_csv_texts()

Load all terjemah texts from CSV files in specified directory.

```python
def load_csv_texts(csv_dir: Path) -> List[str]
```

**Parameters:**
- `csv_dir` (Path): Directory containing CSV files

**Returns:** List[str] - All hadith texts combined

**Requirements:**
- CSV files must have `terjemah` column
- Files should be in UTF-8 encoding

**Example:**
```python
from pathlib import Path
texts = load_csv_texts(Path("data/csv"))
```

---

### save_results()

Save extraction results to JSON file with metadata.

```python
def save_results(grouped_terms: Dict[str, List[str]], 
                 output_path: Path,
                 min_freq: int = 15,
                 max_ngram: int = 3) -> None
```

**Parameters:**
- `grouped_terms` (Dict[str, List[str]]): Keywords map to save
- `output_path` (Path): Output file path
- `min_freq` (int): Minimum frequency used
- `max_ngram` (int): Maximum n-gram size used

**Output Format:**
```json
{
  "metadata": {
    "description": "Enhanced Islamic keywords...",
    "extraction_version": "enhanced_v2.0",
    "min_frequency": 15,
    "max_ngram": 3,
    "total_groups": 150,
    "features": ["sanad_noise_removal", ...]
  },
  "keywords": {
    "group_name": ["term1", "term2", ...]
  }
}
```

---

## 📋 Configuration Constants

### Global Settings

```python
# File paths
CSV_DIR = Path("data/csv")
OUTPUT_PATH = Path("data/processed/enhanced_keywords_map.json")
TERJEMAH_FIELD = "terjemah"

# Processing parameters  
MIN_FREQUENCY = 15      # Default minimum frequency
MAX_NGRAM = 3          # Default maximum n-gram length
```

### Islamic Terms Database

```python
islamic_terms = {
    # Core worship practices (10+ terms)
    'shalat', 'puasa', 'zakat', 'haji', 'umrah', ...
    
    # Purification (5+ terms)
    'wudhu', 'tayammum', 'ghusl', ...
    
    # Legal concepts (15+ terms)
    'halal', 'haram', 'wajib', 'sunnah', ...
    
    # Spiritual concepts (15+ terms) 
    'iman', 'islam', 'tauhid', 'taqwa', ...
    
    # Ethics and behavior (10+ terms)
    'akhlak', 'adab', 'birrul walidain', ...
    
    # And more categories...
}
```

### Stopwords Set

```python
stopwords = {
    # Common Indonesian stopwords
    'yang', 'dan', 'di', 'ke', 'dari', ...
    
    # Hadith transmission terms
    'menceritakan', 'bercerita', 'mengabarkan', ...
    
    # Honorific terms
    'saw', 'ra', 'rah', 'radhiyallahu', ...
    
    # Narrator introduction patterns
    'hadits', 'riwayat', 'diriwayatkan', ...
}
```

### Noise Patterns

```python
noise_patterns = [
    # Arabic name patterns
    r'.*bin.*', r'.*abu.*', r'.*ibnu.*', ...
    
    # Transmission chain patterns  
    r'.*menceritakan.*', r'.*bercerita.*', ...
    
    # Pure honorific patterns
    r'^saw$', r'^ra$', r'^rah$', ...
]
```

---

## ⚠️ Error Handling

### Exception Types

#### ImportError
```python
try:
    import pandas as pd
except ImportError:
    logger.error("pandas not available - install with: pip install pandas")
```

#### FileNotFoundError
```python
try:
    texts = load_csv_texts(csv_dir)
except FileNotFoundError:
    logger.error(f"CSV directory not found: {csv_dir}")
```

#### ValueError
```python
if min_frequency < 1:
    raise ValueError("min_frequency must be >= 1")
```

### Best Practices

#### Parameter Validation
```python
def __init__(self, min_frequency: int = 15, ...):
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")
    if max_ngram < 1 or max_ngram > 5:
        raise ValueError("max_ngram must be between 1-5")
```

#### Graceful Degradation
```python
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Advanced clustering disabled.")
```

---

## 💡 Usage Examples

### Basic Extraction

```python
from utils.improved_keyword_extractor import EnhancedIslamicKeywordExtractor
from pathlib import Path

# Initialize extractor
extractor = EnhancedIslamicKeywordExtractor()

# Load texts
texts = ["Hadith text 1", "Hadith text 2", ...]

# Extract keywords
keywords_map = extractor.create_enhanced_keywords_map(texts)

# Access results
for group, terms in keywords_map.items():
    print(f"{group}: {terms}")
```

### Custom Configuration

```python
# High precision configuration
extractor = EnhancedIslamicKeywordExtractor(
    min_frequency=25,  # Higher threshold
    max_ngram=2        # Shorter n-grams
)

# Low precision, high recall configuration
extractor = EnhancedIslamicKeywordExtractor(
    min_frequency=5,   # Lower threshold
    max_ngram=4        # Longer n-grams
)
```

### Batch Processing

```python
import pandas as pd

# Process multiple CSV files
all_texts = []
for csv_file in Path("data/csv").glob("*.csv"):
    df = pd.read_csv(csv_file)
    if 'terjemah' in df.columns:
        texts = df['terjemah'].dropna().tolist()
        all_texts.extend(texts)

# Extract keywords from combined corpus
extractor = EnhancedIslamicKeywordExtractor()
keywords_map = extractor.create_enhanced_keywords_map(all_texts)
```

### Integration with Query System

```python
from utils.improved_keyword_extractor import EnhancedIslamicKeywordExtractor

def enhanced_query_processing(query: str):
    # Extract Islamic keywords from query
    extractor = EnhancedIslamicKeywordExtractor(min_frequency=1)
    query_keywords = extractor.extract_rule_based_keywords([query])
    
    # Use for filtering FAISS results
    return list(query_keywords)
```

---

## 📊 Performance Considerations

### Memory Usage
- **Small Dataset** (< 1K hadith): ~10MB RAM
- **Medium Dataset** (1K-10K hadith): ~50MB RAM  
- **Large Dataset** (10K+ hadith): ~200MB+ RAM

### Processing Time
- **Text Normalization**: ~0.1ms per hadith
- **N-gram Generation**: ~1ms per hadith
- **Statistical Extraction**: ~10s per 1K hadith
- **Rule-based Extraction**: ~1s per 1K hadith
- **Semantic Grouping**: ~100ms per 1K unique terms

### Optimization Tips
```python
# For large datasets, process in chunks
chunk_size = 1000
for i in range(0, len(texts), chunk_size):
    chunk = texts[i:i + chunk_size]
    partial_results = extractor.extract_statistical_keywords(chunk)
    # Combine results...
```

---

**Related Documentation:**
- [User Guide](ENHANCED_KEYWORD_EXTRACTOR_GUIDE.md) - Getting started and usage examples
- [Technical Flow](TECHNICAL_FLOW.md) - Architecture and implementation details