"""
Enhanced Keyword Extractor - Fixed V1
====================================

Combines the best features from all extraction systems:
- Hybrid TF-IDF + rule-based + cleaned keywords integration
- Conservative noise filtering with Indonesian Islamic terms
- Semantic grouping with phrase component extraction
- Statistical scoring with Islamic term prioritization

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

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
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn/numpy not available. Using fallback methods.")


class EnhancedKeywordExtractor:
    """
    Next-generation keyword extractor combining all best practices.
    """
    
    def __init__(self, 
                 min_frequency: int = 40,
                 max_ngram: int = 3,
                 cleaned_keywords_path: str = "../data/processed/keywords_map_grouped_cleaned.json"):
        
        self.min_frequency = min_frequency
        self.max_ngram = max_ngram
        
        # Load comprehensive Islamic terms from cleaned keywords map
        self.islamic_terms, self.cleaned_keywords_map = self._load_enhanced_islamic_terms(cleaned_keywords_path)
        
        # Enhanced stopwords with broken word fragments
        self.stopwords = self._get_comprehensive_stopwords()
        
        # Comprehensive noise patterns for sanad removal  
        self.noise_patterns = self._get_noise_patterns()
        
        # Indonesian Islamic categories for semantic grouping
        self.islamic_categories = self._get_indonesian_islamic_categories()
    
    def _load_enhanced_islamic_terms(self, path: str) -> Tuple[Set[str], Dict]:
        """Load comprehensive Islamic terms from cleaned keywords map."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                keywords_data = data.get('keywords', {})
                
                # Flatten all keyword groups into a single set
                all_terms = set()
                for group_name, terms_list in keywords_data.items():
                    all_terms.add(group_name)
                    all_terms.update(terms_list)
                
                logger.info(f"Loaded {len(all_terms)} Islamic terms from cleaned keywords map")
                return all_terms, keywords_data
                
        except Exception as e:
            logger.warning(f"Could not load from cleaned keywords map: {e}")
            # Fallback to basic Islamic terms
            basic_terms = {
                'shalat', 'puasa', 'zakat', 'haji', 'wudhu', 'halal', 'haram',
                'islam', 'iman', 'muslim', 'quran', 'hadits', 'rasul', 'allah'
            }
            return basic_terms, {}
    
    def _get_comprehensive_stopwords(self) -> Set[str]:
        """Get comprehensive stopwords including hadith-specific terms."""
        return {
            # Indonesian stopwords
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
            
            # Truncated/broken words that cause noise (fixed from previous issues)
            'ang', 'berik', 'dekah', 'ramad', 'ras', 'deng', 'bakti'
        }
    
    def _get_noise_patterns(self) -> List[str]:
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
    
    def _get_indonesian_islamic_categories(self) -> Dict[str, List[str]]:
        """Indonesian Islamic concept categories for semantic grouping."""
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
        if not text or not isinstance(text, str):
            return ""
        
        # Unicode normalization
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove quotes and special characters but keep apostrophes
        text = re.sub(r'["""\'`""''«»]', '', text)
        
        # Normalize Islamic phrases
        text = re.sub(r'shallallahu\s+[\'"]?alaihi\s+wa?\s*sallam', 'saw', text)
        text = re.sub(r'sallallahu\s+[\'"]?alaihi\s+wa?\s*sallam', 'saw', text)
        text = re.sub(r'radhi\s*allahu\s+(anhu|anha|anhum)', 'ra', text)
        text = re.sub(r'radhiyallahu\s+(anhu|anha|anhum)', 'ra', text)
        
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
        text = re.sub(r'\s*-\s*', ' ', text)  # Clean up dashes
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def is_meaningful_term(self, term: str) -> bool:
        """Check if a term is meaningful for Islamic keyword extraction."""
        if not term or len(term.strip()) <= 2:
            return False
            
        words = term.split()
        
        # Skip if all words are stopwords
        if all(word in self.stopwords for word in words):
            return False
        
        # Skip pure numbers or very short terms
        if re.match(r'^\d+$', term.strip()):
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
        if not text:
            return []
            
        words = self.normalize_text(text).split()
        ngrams = []
        
        for n in range(1, self.max_ngram + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                if self.is_meaningful_term(ngram):
                    ngrams.append(ngram)
        
        return ngrams
    
    def extract_statistical_keywords(self, texts: List[str]) -> Dict[str, float]:
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
        
        # Count frequencies with progress monitoring
        logger.info("Counting n-gram frequencies...")
        counter = Counter()
        
        # Process in batches to avoid memory issues
        batch_size = 100000
        for i in range(0, len(all_ngrams), batch_size):
            batch = all_ngrams[i:i + batch_size]
            counter.update(batch)
            if i % (batch_size * 10) == 0:
                logger.info(f"Processed {i:,}/{len(all_ngrams):,} n-grams")
        
        logger.info(f"Frequency counting completed. Found {len(counter)} unique terms")
        
        # Filter by minimum frequency
        frequent_terms = {term: count for term, count in counter.items() 
                         if count >= self.min_frequency}
        
        logger.info(f"After frequency filter (>={self.min_frequency}): {len(frequent_terms)} terms")
        
        # Simple TF-IDF approximation if sklearn available
        if SKLEARN_AVAILABLE:
            logger.info("Calculating TF-IDF scores...")
            total_docs = len(texts)
            tfidf_scores = {}
            
            # Pre-normalize texts for faster lookup
            normalized_texts = [self.normalize_text(text) for text in texts]
            
            for i, (ngram, freq) in enumerate(frequent_terms.items()):
                if i % 100 == 0:
                    logger.info(f"TF-IDF progress: {i}/{len(frequent_terms)} terms")
                
                # Calculate document frequency (faster with pre-normalized texts)
                df = sum(1 for norm_text in normalized_texts if ngram in norm_text)
                
                # Calculate TF-IDF
                tf = freq / len(all_ngrams)
                idf = math.log(total_docs / (df + 1))
                tfidf_scores[ngram] = tf * idf
            
            logger.info(f"Found {len(tfidf_scores)} terms with TF-IDF scores")
            return tfidf_scores
        else:
            # Normalize frequency scores
            total_count = sum(frequent_terms.values())
            normalized_scores = {
                term: count / total_count for term, count in frequent_terms.items()
            }
            logger.info(f"Found {len(normalized_scores)} terms with frequency scores")
            return normalized_scores
    
    def extract_rule_based_keywords(self, texts: List[str]) -> Dict[str, float]:
        """Extract keywords using Islamic terminology rules."""
        logger.info("Extracting rule-based Islamic keywords...")
        
        found_terms = {}
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Scanning text {i+1}/{len(texts)}")
                
            normalized = self.normalize_text(text)
            text_lower = normalized.lower()
            
            # Find Islamic terms in text
            for term in self.islamic_terms:
                if re.search(rf'\b{re.escape(term)}\b', text_lower):
                    found_terms[term] = found_terms.get(term, 0) + 1
        
        # Normalize scores
        total_count = sum(found_terms.values())
        if total_count > 0:
            normalized_scores = {
                term: count / total_count for term, count in found_terms.items()
            }
        else:
            normalized_scores = {}
        
        logger.info(f"Found {len(normalized_scores)} Islamic terms")
        return normalized_scores
    
    def extract_phrase_components(self, phrases: List[str]) -> Dict[str, List[str]]:
        """Extract phrase components for better matching."""
        phrase_components = {}
        
        for phrase in phrases:
            if ' ' in phrase:  # Multi-word phrase
                components = phrase.split()
                meaningful_components = [
                    comp for comp in components 
                    if self.is_meaningful_term(comp)
                ]
                if meaningful_components:
                    phrase_components[phrase] = meaningful_components
            else:
                phrase_components[phrase] = [phrase]
        
        return phrase_components
    
    def hybrid_extract(self, texts: List[str]) -> Dict[str, Dict]:
        """Enhanced extraction combining statistical and rule-based methods."""
        logger.info("Starting enhanced hybrid keyword extraction...")
        
        # Extract using both methods
        statistical_terms = self.extract_statistical_keywords(texts)
        rule_based_terms = self.extract_rule_based_keywords(texts)
        
        # Combine results
        all_keywords = set(statistical_terms.keys()) | set(rule_based_terms.keys())
        combined_results = {}
        
        for keyword in all_keywords:
            stat_score = statistical_terms.get(keyword, 0)
            rule_score = rule_based_terms.get(keyword, 0)
            
            # Weighted combination (Islamic terms get higher rule-based weight)
            if keyword in self.islamic_terms:
                combined_score = 0.3 * stat_score + 0.7 * rule_score
            else:
                combined_score = 0.7 * stat_score + 0.3 * rule_score
            
            combined_results[keyword] = {
                'score': combined_score,
                'statistical_score': stat_score,
                'rule_based_score': rule_score,
                'is_islamic_term': keyword in self.islamic_terms,
                'length': len(keyword.split())
            }
        
        # Extract phrase components
        logger.info("Extracting phrase components...")
        phrases = list(combined_results.keys())
        phrase_components = self.extract_phrase_components(phrases)
        
        # Add component information
        for phrase, components in phrase_components.items():
            if phrase in combined_results:
                combined_results[phrase]['components'] = components
        
        logger.info(f"Extracted {len(combined_results)} keywords with hybrid scoring")
        return combined_results
    
    def create_enhanced_keywords_map(self, texts: List[str]) -> Dict[str, List[str]]:
        """Create enhanced keywords map with semantic grouping."""
        logger.info("Creating enhanced keywords map...")
        
        # Extract keywords using hybrid method
        extraction_results = self.hybrid_extract(texts)
        
        # Group by Islamic categories first (using cleaned keywords structure)
        grouped = defaultdict(list)
        categorized = set()
        
        # Use cleaned keywords map structure if available
        if self.cleaned_keywords_map:
            for category, terms_list in self.cleaned_keywords_map.items():
                found_variants = []
                for term in extraction_results.keys():
                    if term in terms_list or any(variant in term.lower() for variant in terms_list):
                        found_variants.append(term)
                        categorized.add(term)
                
                if found_variants:
                    grouped[category] = sorted(found_variants)
        
        # Group by Indonesian Islamic categories for remaining terms
        remaining_terms = [term for term in extraction_results.keys() if term not in categorized]
        for category, category_terms in self.islamic_categories.items():
            for term in remaining_terms:
                for category_term in category_terms:
                    if category_term.lower() in term.lower():
                        grouped[category_term].append(term)
                        categorized.add(term)
                        break
        
        # Add high-scoring standalone terms
        high_scoring_terms = [
            term for term, info in extraction_results.items() 
            if info['score'] >= 0.01 and term not in categorized
        ]
        high_scoring_terms.sort(key=lambda x: extraction_results[x]['score'], reverse=True)
        
        for term in high_scoring_terms[:50]:  # Top 50 remaining terms
            if extraction_results[term]['score'] >= self.min_frequency * 0.001:
                grouped[term] = [term]
        
        # Clean up groups - remove duplicates and sort
        final_grouped = {}
        for key, values in grouped.items():
            if values:  # Only keep non-empty groups
                final_grouped[key] = sorted(list(set(values)))
        
        logger.info(f"Created {len(final_grouped)} enhanced semantic groups")
        return final_grouped


def main():
    """Enhanced main extraction pipeline."""
    logger.info("🚀 Starting Enhanced Keyword Extraction - Fixed V1")
    logger.info("=" * 60)
    
    # Configuration
    JSON_DIR = Path("../data/processed/hadits_docs.json")
    OUTPUT_PATH = Path("../data/enhanced_index_v1/enhanced_keywords_map_v1.json")
    MIN_FREQUENCY = 80  # Increase to reduce candidates
    MAX_NGRAM = 2       # Reduce to speed up processing
    
    try:
        # Initialize enhanced extractor
        extractor = EnhancedKeywordExtractor(
            min_frequency=MIN_FREQUENCY,
            max_ngram=MAX_NGRAM
        )
        
        # Load texts from JSON
        logger.info(f"Loading texts from {JSON_DIR}")
        with open(JSON_DIR, 'r', encoding='utf-8') as f:
            hadits_data = json.load(f)
        
        texts = [hadis['terjemah'] for hadis in hadits_data if hadis.get('terjemah')]
        logger.info(f"Loaded {len(texts)} hadis texts")
        
        if not texts:
            logger.error("No texts found. Please check the JSON file.")
            return
        
        # Create enhanced keywords map
        grouped_terms = extractor.create_enhanced_keywords_map(texts)
        
        if not grouped_terms:
            logger.error(f"No semantic groups created. Check min_frequency ({MIN_FREQUENCY})")
            return
        
        # Save results with metadata
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "metadata": {
                "description": "Enhanced Islamic keywords extracted from hadits collections - Fixed V1",
                "extraction_version": "enhanced_v1.0",
                "min_frequency": MIN_FREQUENCY,
                "max_ngram": MAX_NGRAM,
                "total_groups": len(grouped_terms),
                "total_texts": len(texts),
                "extraction_method": "hybrid_enhanced_islamic_semantic_grouping",
                "features": [
                    "cleaned_keywords_integration",
                    "indonesian_islamic_categories", 
                    "conservative_noise_filtering",
                    "hybrid_statistical_rule_based",
                    "phrase_component_extraction",
                    "islamic_term_prioritization"
                ]
            },
            "keywords": grouped_terms
        }
        
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 ENHANCED EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total hadis processed: {len(texts):,}")
        logger.info(f"Final semantic groups: {len(grouped_terms):,}")
        logger.info(f"Output file: {OUTPUT_PATH}")
        
        # Show top semantic groups
        logger.info("\n🕌 Top semantic groups found:")
        sorted_groups = sorted(grouped_terms.items(), key=lambda x: len(x[1]), reverse=True)
        for group_name, terms in sorted_groups[:10]:
            logger.info(f"  • {group_name}: {len(terms)} terms")
            if len(terms) <= 5:
                logger.info(f"    Terms: {', '.join(terms)}")
        
        logger.info(f"\n✅ Enhanced extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in enhanced extraction pipeline: {e}")
        raise


if __name__ == "__main__":
    main()