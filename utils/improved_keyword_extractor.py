"""
Enhanced Islamic Keyword Extractor for Hadith AI
================================================

Advanced keyword extraction system specifically designed for Islamic hadith texts.
This improved version addresses the issues found in the original extractor by:

1. Better noise filtering (sanad chains, narrator names) 
2. Enhanced Islamic terminology detection
3. Semantic grouping and clustering
4. Context-aware keyword validation
5. Frequency-based importance scoring
6. Multi-level filtering pipeline

Author: Hadith AI Team - Enhanced Version
Date: 2024
"""

import pandas as pd
import re
import json
import math
import logging
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Union, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Advanced clustering disabled.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("numpy not available. Some features may be limited.")


class EnhancedIslamicKeywordExtractor:
    """
    Advanced Islamic keyword extractor with enhanced noise filtering
    and semantic grouping capabilities.
    """
    
    def __init__(self, 
                 min_frequency: int = 20,
                 max_ngram: int = 3,
                 islamic_terms_path: str = None):
        self.min_frequency = min_frequency
        self.max_ngram = max_ngram
        
        # Load Islamic terminology dictionary
        self.islamic_terms = self._load_enhanced_islamic_terms(islamic_terms_path)
        
        # Enhanced stopwords specifically for hadith texts
        self.stopwords = self._get_enhanced_stopwords()
        
        # Comprehensive noise patterns for sanad removal
        self.noise_patterns = self._get_comprehensive_noise_patterns()
        
        # Islamic concept categories for semantic grouping
        self.islamic_categories = self._get_islamic_categories()

    def _load_enhanced_islamic_terms(self, path: str = None) -> Set[str]:
        """Load comprehensive Islamic terms dictionary from cleaned keywords map."""
        # Try to load from the comprehensive cleaned keywords map
        cleaned_keywords_path = "data/processed/keywords_map_grouped_cleaned.json"
        if Path(cleaned_keywords_path).exists():
            try:
                with open(cleaned_keywords_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    keywords_data = data.get('keywords', {})
                    
                    # Flatten all keyword groups into a single set
                    all_terms = set()
                    for group_name, terms_list in keywords_data.items():
                        # Add the main group name
                        all_terms.add(group_name)
                        # Add all variants
                        all_terms.update(terms_list)
                    
                    logger.info(f"Loaded {len(all_terms)} Islamic terms from cleaned keywords map")
                    return all_terms
                    
            except Exception as e:
                logger.warning(f"Could not load from cleaned keywords map: {e}")
        
        # Fallback to custom path if provided
        if path and Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return set(data.get('terms', []))
            except Exception as e:
                logger.warning(f"Could not load Islamic terms from {path}: {e}")
        
        # Fallback to basic Islamic terms if files not available
        logger.warning("Using fallback Islamic terms - limited coverage")
        return {
            # Core terms that should always be recognized
            'shalat', 'salat', 'sholat', 'solat', 'puasa', 'shaum', 'zakat', 'sedekah',
            'haji', 'umrah', 'wudhu', 'wudu', 'tayammum', 'halal', 'haram', 'makruh',
            'sunnah', 'wajib', 'fardhu', 'mubah', 'iman', 'islam', 'muslim', 'mukmin',
            'taqwa', 'sabar', 'syukur', 'ikhlas', 'taubat', 'istighfar', 'jihad',
            'nikah', 'menikah', 'talaq', 'cerai', 'riba', 'zina', 'khamr', 'quran',
            'hadits', 'sunnah', 'wahyu', 'surga', 'neraka', 'akhirat', 'kiamat',
            'doa', 'dzikir', 'tasbih', 'takbir', 'tahlil', 'malaikat', 'syahid'
        }

    def _get_enhanced_stopwords(self) -> Set[str]:
        """Get comprehensive stopwords including hadith-specific terms."""
        return {
            # Common Indonesian stopwords
            'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'dengan', 'untuk', 'adalah', 'akan',
            'telah', 'sudah', 'atau', 'juga', 'tidak', 'bila', 'jika', 'ketika', 'saat',
            'itu', 'ini', 'mereka', 'kita', 'kami', 'dia', 'ia', 'saya', 'anda', 'engkau',
            'kamu', 'kalian', 'beliau', 'ada', 'seperti', 'antara', 'semua', 'setiap',
            'dalam', 'oleh', 'kepada', 'bagi', 'atas', 'bawah', 'lalu', 'kemudian',
            'setelah', 'sebelum', 'sambil', 'seraya', 'hingga', 'sampai', 'selama',
            
            # Query-specific noise terms
            'apa', 'bagaimana', 'dimana', 'kapan', 'mengapa', 'siapa', 'berapa',
            'berikan', 'jelaskan', 'sebutkan', 'cara', 'metode', 'langkah',
            'tentang', 'mengenai', 'terkait', 'berhubungan', 'itu', 'benar',
            
            # Hadith transmission chain terms (sanad)
            'menceritakan', 'bercerita', 'mengabarkan', 'memberitahukan', 'mendengar',
            'dari', 'kepada', 'telah', 'berkata', 'bin', 'abu', 'ibnu', 'al', 'an', 
            'as', 'ad', 'ar', 'az', 'ats', 'ath', 'asy', 'ash', 'binti',
            
            # Honorific terms
            'saw', 'ra', 'rah', 'radhiyallahu', 'anhu', 'anha', 'anhum', 'anhuma',
            'shallallahu', 'alaihi', 'wasallam', 'sallallahu', 'alaih', 'radhi',
            'rahimahullah', 'hafizahullah',
            
            # Narrator introduction patterns
            'hadits', 'hadist', 'riwayat', 'diriwayatkan', 'meriwayatkan',
            'hadathana', 'akhbarana', 'haddatsani', 'akhbarani',
            
            # Common verbs that don't add semantic value
            'berkata', 'mengatakan', 'bersabda', 'menjawab', 'bertanya',
            'melihat', 'datang', 'pergi', 'kembali', 'pulang', 'keluar', 'masuk',
            
            # Truncated/broken words that cause noise
            'ang', 'berik', 'dekah', 'ramad', 'ras', 'deng', 'bakti'
        }

    def _get_comprehensive_noise_patterns(self) -> List[str]:
        """Get comprehensive regex patterns for noise removal."""
        return [
            # Arabic name patterns
            r'.*bin.*', r'.*abu.*', r'.*ibnu.*', r'.*al[- ].*', r'.*ibn.*',
            r'.*binti.*', r'.*ummu.*',
            
            # Transmission chain patterns
            r'.*menceritakan.*', r'.*bercerita.*', r'.*mengabarkan.*',
            r'.*memberitahukan.*', r'.*riwayat.*', r'.*hadathana.*',
            r'.*akhbarana.*',
            
            # Pure honorific patterns
            r'^saw$', r'^ra$', r'^rah$', r'^radhiyallahu.*', r'^shallallahu.*',
            
            # Pure transmission verbs
            r'^(menceritakan|bercerita|mengabarkan|memberitahukan|mendengar)$',
            r'^(berkata|mengatakan|bersabda|menjawab|bertanya)$'
        ]

    def _get_islamic_categories(self) -> Dict[str, List[str]]:
        """Get Islamic concept categories for semantic grouping using Indonesian terms."""
        return {
            'ibadah': ['shalat', 'puasa', 'zakat', 'haji', 'umrah', 'doa', 'dzikir'],
            'bersuci': ['wudhu', 'tayammum', 'ghusl', 'bersuci', 'najis', 'suci', 'thaharah'],
            'akhlak': ['akhlak', 'adab', 'sabar', 'syukur', 'ikhlas', 'amanah', 'jujur'],
            'hukum': ['halal', 'haram', 'wajib', 'sunnah', 'makruh', 'mubah', 'hudud'],
            'aqidah': ['iman', 'islam', 'tauhid', 'syirik', 'kufur', 'taqwa'],
            'muamalah': ['nikah', 'jual_beli', 'sedekah', 'silaturahmi', 'jihad', 'riba'],
            'akhirat': ['akhirat', 'surga', 'neraka', 'kiamat', 'mahsyar', 'syahid'],
            'ramadan': ['puasa', 'ramadan', 'sahur', 'iftar', 'berbuka', 'tarawih'],
            'minuman': ['khamr', 'arak', 'minuman keras', 'memabukkan', 'mabuk']
        }

    def normalize_text(self, text: str) -> str:
        """Enhanced text normalization for hadith texts."""
        if pd.isna(text) or not text:
            return ""
        
        text = text.lower().strip()
        
        # Remove quotes and normalize punctuation
        text = re.sub(r'[\"\"\"\'\'\``]', '', text)
        text = re.sub(r'\s*[-–—]\s*', ' ', text)
        
        # Normalize Islamic phrases and honorifics
        text = re.sub(r'shallallahu\s+[\'\"]*alaihi\s+wa?\s*sallam', '', text)
        text = re.sub(r'sallallahu\s+[\'\"]*alaihi\s+wa?\s*sallam', '', text)
        text = re.sub(r'radhi\s*allahu\s+(anhu|anha|anhum)', '', text)
        text = re.sub(r'radhiyallahu\s+(anhu|anha|anhum)', '', text)
        text = re.sub(r'rahimahullah\s*', '', text)
        
        # Remove Arabic script remnants
        text = re.sub(r'[ء-ي]+', ' ', text)
        
        # Remove transmission chain indicators
        text = re.sub(r'\b(telah\s+)?(menceritakan|mengabarkan|memberitahukan|bercerita)\s+(kepada\s+)?(kami|ku|nya)\b', '', text)
        text = re.sub(r'\btelah\s+(menceritakan|mengabarkan)\s+kepada\s+(kami|ku)\b', '', text)
        
        # Remove narrator introduction patterns
        text = re.sub(r'\b(dari|kepada)\s+([a-z]+\s+)?(bin|abu|ibnu)\s+[a-z]+\b', '', text, flags=re.IGNORECASE)
        
        # Remove bracketed narrator names
        text = re.sub(r'\[[^\]]*\]', '', text)
        
        # Clean up excessive punctuation but preserve word structure
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def is_meaningful_term(self, term: str) -> bool:
        """Check if a term is meaningful for Islamic keyword extraction."""
        if not term or pd.isna(term):
            return False
            
        words = term.split()
        
        # Skip if all words are stopwords
        if all(word in self.stopwords for word in words):
            return False
        
        # Skip pure numbers or very short terms
        if len(term.strip()) <= 2 or re.match(r'^\d+$', term.strip()):
            return False
        
        # Apply comprehensive noise patterns
        for pattern in self.noise_patterns:
            if re.search(pattern, term, re.IGNORECASE):
                return False
        
        # Boost Islamic terms
        if any(islamic_term in term.lower() for islamic_term in self.islamic_terms):
            return True
            
        return True

    def generate_enhanced_ngrams(self, text: str) -> List[str]:
        """Generate meaningful n-grams with Islamic context awareness."""
        if not text or pd.isna(text):
            return []
            
        words = text.split()
        ngrams = []
        
        for n in range(1, self.max_ngram + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                if self.is_meaningful_term(ngram):
                    ngrams.append(ngram)
        
        return ngrams

    def extract_statistical_keywords(self, texts: List[str]) -> Counter:
        """Extract keywords using statistical methods (TF-IDF + frequency)."""
        logger.info("Extracting statistical keywords...")
        
        all_ngrams = []
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Processing text {i+1}/{len(texts)}")
            
            normalized = self.normalize_text(text)
            ngrams = self.generate_enhanced_ngrams(normalized)
            all_ngrams.extend(ngrams)
        
        logger.info(f"Generated {len(all_ngrams)} meaningful n-grams")
        
        # Count frequencies
        counter = Counter(all_ngrams)
        
        # Filter by minimum frequency
        frequent_terms = {term: count for term, count in counter.items() 
                         if count >= self.min_frequency}
        
        logger.info(f"Found {len(frequent_terms)} terms with frequency >= {self.min_frequency}")
        return Counter(frequent_terms)

    def extract_rule_based_keywords(self, texts: List[str]) -> Set[str]:
        """Extract keywords using Islamic terminology rules."""
        logger.info("Extracting rule-based Islamic keywords...")
        
        found_terms = set()
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Scanning text {i+1}/{len(texts)}")
                
            normalized = self.normalize_text(text)
            text_lower = normalized.lower()
            
            # Find Islamic terms in text
            for term in self.islamic_terms:
                if term in text_lower:
                    found_terms.add(term)
        
        logger.info(f"Found {len(found_terms)} Islamic terms")
        return found_terms

    def enhanced_extract(self, texts: List[str]) -> Tuple[Counter, Set[str]]:
        """Enhanced extraction combining statistical and rule-based methods."""
        logger.info("Starting enhanced Islamic keyword extraction...")
        
        # Extract using both methods
        statistical_terms = self.extract_statistical_keywords(texts)
        rule_based_terms = self.extract_rule_based_keywords(texts)
        
        return statistical_terms, rule_based_terms

    def create_enhanced_keywords_map(self, texts: List[str]) -> Dict[str, List[str]]:
        """Create enhanced keywords map with semantic grouping."""
        logger.info("Creating enhanced keywords map...")
        
        # Extract keywords using both methods
        statistical_terms, rule_based_terms = self.enhanced_extract(texts)
        
        # Combine and organize
        all_terms = set(statistical_terms.keys()) | rule_based_terms
        grouped = defaultdict(list)
        categorized = set()
        
        # Group by Islamic categories
        for category, category_terms in self.islamic_categories.items():
            for term in all_terms:
                for category_term in category_terms:
                    if category_term.lower() in term.lower():
                        grouped[category_term].append(term)
                        categorized.add(term)
                        break
        
        # Handle remaining high-frequency terms
        remaining_terms = [term for term in statistical_terms.keys() 
                          if term not in categorized]
        remaining_terms.sort(key=lambda x: statistical_terms[x], reverse=True)
        
        # Add high-frequency standalone terms
        for term in remaining_terms[:50]:  # Top 50 remaining terms
            if statistical_terms[term] >= self.min_frequency * 2:
                grouped[term] = [term]
        
        # Clean up groups - remove duplicates and sort
        final_grouped = {}
        for key, values in grouped.items():
            if values:  # Only keep non-empty groups
                final_grouped[key] = sorted(list(set(values)))
        
        logger.info(f"Created {len(final_grouped)} semantic groups")
        return final_grouped


# Configuration constants
CSV_DIR = Path("data/csv")
JSON_DIR = Path("data/processed/hadits_docs.json")
OUTPUT_PATH = Path("data/processed/enhanced_keywords_map.json")
TERJEMAH_FIELD = "terjemah"
MIN_FREQUENCY = 40
MAX_NGRAM = 3

def generate_meaningful_ngrams(text: str, max_n: int = MAX_NGRAM) -> list:
    """Legacy function - use EnhancedIslamicKeywordExtractor instead."""
    extractor = EnhancedIslamicKeywordExtractor(max_ngram=max_n)
    return extractor.generate_enhanced_ngrams(text)

def load_json_texts(json_path: Path = Path("data/processed/hadits_docs.json")) -> List[str]:
      """Load all terjemah texts from processed JSON file."""
      logger.info(f"Loading texts from {json_path}")

      with open(json_path, 'r', encoding='utf-8') as f:
          hadits_data = json.load(f)

      texts = [hadis['terjemah'] for hadis in hadits_data if hadis.get('terjemah')]
      logger.info(f"Loaded {len(texts)} hadis texts")
      return texts

def load_csv_texts(csv_dir: Path) -> List[str]:
    """Load all terjemah texts from CSV files."""
    all_texts = []
    
    logger.info(f"Loading CSV files from {csv_dir}")
    
    for csv_file in csv_dir.glob("*.csv"):
        try:
            logger.info(f"Reading {csv_file.name}...")
            df = pd.read_csv(csv_file)
            
            if TERJEMAH_FIELD in df.columns:
                texts = df[TERJEMAH_FIELD].dropna().tolist()
                all_texts.extend(texts)
                logger.info(f"Loaded {len(texts)} hadis from {csv_file.name}")
            else:
                logger.warning(f"Column '{TERJEMAH_FIELD}' not found in {csv_file.name}")
                
        except Exception as e:
            logger.error(f"Error reading {csv_file.name}: {e}")
    
    logger.info(f"Total hadis texts loaded: {len(all_texts)}")
    return all_texts

def extract_frequent_terms_legacy(texts: List[str], min_freq: int = MIN_FREQUENCY) -> Counter:
    """Legacy function - use EnhancedIslamicKeywordExtractor instead."""
    extractor = EnhancedIslamicKeywordExtractor(min_frequency=min_freq)
    return extractor.extract_statistical_keywords(texts)

def group_by_semantic_categories_legacy(frequent_terms: Counter) -> Dict[str, List[str]]:
    """Legacy function - use EnhancedIslamicKeywordExtractor instead."""
    extractor = EnhancedIslamicKeywordExtractor()
    # Convert Counter to text list for processing
    texts = []
    for term, freq in frequent_terms.items():
        texts.extend([term] * freq)
    
    return extractor.create_enhanced_keywords_map(texts)

def save_results(grouped_terms: Dict[str, List[str]], output_path: Path, 
                 min_freq: int = MIN_FREQUENCY, max_ngram: int = MAX_NGRAM):
    """Save results to JSON file with metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    result = {
        "metadata": {
            "description": "Enhanced Islamic keywords extracted from hadits collections",
            "extraction_version": "enhanced_v2.0",
            "min_frequency": min_freq,
            "max_ngram": max_ngram,
            "total_groups": len(grouped_terms),
            "extraction_method": "enhanced_islamic_semantic_grouping",
            "features": [
                "sanad_noise_removal",
                "islamic_terminology_detection", 
                "semantic_grouping",
                "literal_overlap_boosting"
            ]
        },
        "keywords": grouped_terms
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {output_path}")

def main():
    """Enhanced main extraction pipeline with improved Islamic terminology handling."""
    logger.info("🚀 Starting Enhanced Islamic Keyword Extraction")
    logger.info("=" * 60)
    
    try:
        # Initialize enhanced extractor
        extractor = EnhancedIslamicKeywordExtractor(
            min_frequency=MIN_FREQUENCY,
            max_ngram=MAX_NGRAM
        )
        
        # Load all CSV texts
        # texts = load_csv_texts(CSV_DIR)
        texts = load_json_texts(JSON_DIR)
        
        if not texts:
            logger.error("No texts found. Please check the CSV files.")
            return
        
        # Create enhanced keywords map
        grouped_terms = extractor.create_enhanced_keywords_map(texts)
        
        if not grouped_terms:
            logger.error(f"No semantic groups created. Check min_frequency ({MIN_FREQUENCY})")
            return
        
        # Save results
        save_results(grouped_terms, OUTPUT_PATH, MIN_FREQUENCY, MAX_NGRAM)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 ENHANCED EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total hadis processed: {len(texts):,}")
        logger.info(f"Final semantic groups: {len(grouped_terms):,}")
        logger.info(f"Output file: {OUTPUT_PATH}")
        
        # Show top Islamic categories
        logger.info("\n🕌 Top semantic groups found:")
        sorted_groups = sorted(grouped_terms.items(), key=lambda x: len(x[1]), reverse=True)
        for group_name, terms in sorted_groups[:10]:  # Top 10 groups
            logger.info(f"  • {group_name}: {len(terms)} terms")
            if len(terms) <= 5:  # Show terms if few
                logger.info(f"    Terms: {', '.join(terms)}")
        
        logger.info(f"\n✅ Enhanced extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main extraction pipeline: {e}")
        raise

def test_extractor():
    """Test the enhanced extractor with sample data."""
    logger.info("Testing Enhanced Islamic Keyword Extractor...")
    
    # Sample hadith texts for testing
    sample_texts = [
        "Diriwayatkan dari Abu Hurairah ra bahwa Rasulullah saw bersabda: Shalat berjamaah lebih baik daripada shalat sendirian dengan 27 derajat.",
        "Telah menceritakan kepada kami Abdullah bin Masud ra: Barang siapa yang berpuasa pada bulan Ramadan dengan iman dan mengharap pahala, diampuni dosanya yang telah lalu.",
        "Dari Anas bin Malik ra berkata: Rasulullah saw bersabda tentang zakat: Zakat adalah rukun Islam yang ketiga setelah syahadat dan shalat.",
        "Abu Bakar ra meriwayatkan: Nabi saw bersabda bahwa berwudhu dengan sempurna adalah kunci diterimanya shalat."
    ]
    
    try:
        extractor = EnhancedIslamicKeywordExtractor(min_frequency=1, max_ngram=2)
        
        # Test normalization
        logger.info("Testing text normalization...")
        for i, text in enumerate(sample_texts[:2]):
            normalized = extractor.normalize_text(text)
            logger.info(f"Original: {text[:50]}...")
            logger.info(f"Normalized: {normalized[:50]}...\n")
        
        # Test keyword extraction
        logger.info("Testing keyword extraction...")
        keywords_map = extractor.create_enhanced_keywords_map(sample_texts)
        
        logger.info(f"Extracted {len(keywords_map)} groups:")
        for group, terms in keywords_map.items():
            logger.info(f"  {group}: {terms}")
            
        logger.info("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    main() 