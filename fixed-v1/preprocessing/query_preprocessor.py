"""
Enhanced Query Preprocessor - Fixed V1
=====================================

Advanced query preprocessing with conservative lemmatization and
comprehensive Islamic term standardization.

Features:
- Conservative Indonesian lemmatization (prevents word truncation)
- Islamic term standardization and spelling variations
- Enhanced stopword filtering with context awareness
- Query intent preservation
- Multi-level text normalization

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import re
import unicodedata
import json
from pathlib import Path
from typing import Set, Dict, List, Optional


class EnhancedQueryPreprocessor:
    """
    Advanced query preprocessor optimized for Islamic hadith queries.
    Combines conservative lemmatization with comprehensive Islamic term mapping.
    """
    
    def __init__(self, keywords_map_path: str = "../../data/enhanced_index_v1/enhanced_keywords_map_v1.json"):
        """
        Initialize the enhanced query preprocessor.
        
        Args:
            keywords_map_path (str): Path to enhanced keywords map for term standardization
        """
        self.keywords_map_path = keywords_map_path
        self.keywords_map = self._load_keywords_map()
        
        # Enhanced Indonesian stopwords with query-specific terms
        self.indonesian_stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'dengan', 'untuk', 'adalah', 'akan',
            'telah', 'sudah', 'atau', 'juga', 'tidak', 'bila', 'jika', 'ketika', 'saat',
            'itu', 'ini', 'mereka', 'kita', 'kami', 'dia', 'ia', 'saya', 'anda', 'engkau',
            'kamu', 'kalian', 'beliau', 'ada', 'seperti', 'antara', 'semua', 'setiap',
            'bagi', 'oleh', 'karena', 'sebab', 'supaya', 'agar', 'hingga', 'sampai',
            'maka', 'lalu', 'kemudian', 'setelah', 'sebelum', 'selama', 'sambil',
            'dalam', 'atas', 'bawah', 'luar', 'depan', 'belakang', 'kiri', 'kanan',
            
            # Query-specific stopwords (keep some for intent preservation)
            'apa', 'bagaimana', 'dimana', 'kapan', 'mengapa', 'siapa', 'berapa',
            'tentang', 'mengenai', 'terkait', 'berhubungan', 'cara', 'metode', 'langkah'
        }
        
        # Arabic transliteration stopwords
        self.arabic_stopwords = {
            'al', 'an', 'fi', 'min', 'ila', 'wa', 'la', 'ma', 'li', 'bi',
            'qala', 'qaala', 'hadathana', 'akhbarana', 'anna', 'alladhii',
            'bin', 'abu', 'ibnu', 'ibn', 'as', 'ad', 'ar', 'az', 'ats', 'ath'
        }
        
        # Hadith transmission chain terms (sanad noise)
        self.hadith_stopwords = {
            'saw', 'ra', 'rah', 'radhiyallahu', 'anhu', 'anha', 'anhum', 'anhuma',
            'shallallahu', 'alaihi', 'wasallam', 'sallallahu', 'alaih', 'alayhi',
            'hadits', 'hadist', 'riwayat', 'diriwayatkan', 'menceritakan',
            'telah', 'bercerita', 'kepada', 'kami', 'dari', 'meriwayatkan'
        }
        
        # Combine all stopwords
        self.all_stopwords = (
            self.indonesian_stopwords | 
            self.arabic_stopwords | 
            self.hadith_stopwords
        )
        
        # Enhanced conservative lemmatization rules
        self.conservative_lemmatization_rules = self._build_conservative_rules()
        
        # Term standardization map (from keywords map)
        self.term_standardization = self._build_standardization_map()
    
    def _load_keywords_map(self) -> Dict:
        """Load enhanced keywords map for term standardization."""
        try:
            with open(self.keywords_map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('keywords', {})
        except Exception as e:
            print(f"Warning: Could not load keywords map from {self.keywords_map_path}: {e}")
            return {}
    
    def _build_conservative_rules(self) -> Dict[str, str]:
        """Build conservative lemmatization rules that preserve meaning."""
        return {
            # Worship practices - precise mapping
            'berwudhu': 'wudhu',
            'berwudu': 'wudhu', 
            'bertayammum': 'tayammum',
            'bershalat': 'shalat',
            'menshalatkan': 'shalat',
            'menshalati': 'shalat',
            'menyalatkan': 'shalat',
            'berpuasa': 'puasa',
            'berzakat': 'zakat',
            'bersedekah': 'sedekah',
            'berdzikir': 'dzikir',
            'berdoa': 'doa',
            'berhaji': 'haji',
            
            # IMPORTANT: Preserve query intent words
            'berikan': 'berikan',
            'jelaskan': 'jelaskan',
            'sebutkan': 'sebutkan',
            'tunjukkan': 'tunjukkan',
            'terangkan': 'terangkan',
            
            # Action words that preserve semantic meaning
            'mengeluarkan': 'keluarkan',
            'mengerjakan': 'kerjakan',
            'melaksanakan': 'laksanakan',
            'melakukan': 'lakukan',
            'mengamalkan': 'amalkan',
            'menjalankan': 'jalankan',
            'memenuhi': 'penuhi',
            
            # Permissibility terms - crucial for Islamic queries
            'mengharamkan': 'haram',
            'menghalalkan': 'halal',
            'memperbolehkan': 'boleh',
            'diperbolehkan': 'boleh',
            'dilarang': 'larang',
            'diharamkan': 'haram',
            'dihalalkan': 'halal',
            'dimakruhkan': 'makruh',
            'disunnahkan': 'sunnah',
            'diwajibkan': 'wajib',
            'diperintahkan': 'perintah',
            'dianjurkan': 'anjur',
            'disunahkan': 'sunnah',
            
            # Possessive forms - only for core Islamic terms
            'shalatnya': 'shalat',
            'shalatmu': 'shalat',
            'shalatku': 'shalat',
            'puasanya': 'puasa',
            'puasamu': 'puasa',
            'puasaku': 'puasa',
            'zakatnya': 'zakat',
            'zakatmu': 'zakat',
            'zakatku': 'zakat',
            'wudhunya': 'wudhu',
            'wudhumu': 'wudhu',
            'wudhuku': 'wudhu',
            
            # Spelling variations and standardization
            'salat': 'shalat',
            'sholat': 'shalat',
            'solat': 'shalat',
            'wudu': 'wudhu',
            'shaum': 'puasa',
            'shiyam': 'puasa',
            'shadaqah': 'sedekah',
            'sodaqoh': 'sedekah',
            'shodaqoh': 'sedekah',
            'dzuhur': 'dhuhur',
            'zhuhur': 'dhuhur',
            'dluhur': 'dhuhur',
            'ashar': 'asar',
            'maghrib': 'magrib',
            'ramadhan': 'ramadan',
            'ramadlan': 'ramadan',
            
            # Time and context-specific terms
            'berbuka': 'buka',  # For iftar context
            'khamr': 'khamr',   # Keep Arabic terms intact
            'jihad': 'jihad',
            'syahid': 'syahid',
            'perang': 'perang',
            'nikah': 'nikah',
            'talak': 'talak',
            'cerai': 'cerai',
            
            # Important compound concepts - preserve meaning
            'minuman keras': 'minuman keras',
            'orang tua': 'orang tua',
            'anak yatim': 'anak yatim',
            'fakir miskin': 'fakir miskin'
        }
    
    def _build_standardization_map(self) -> Dict[str, str]:
        """Build term standardization map from keywords map."""
        standardization = {}
        
        for canonical_term, variants in self.keywords_map.items():
            # Map all variants to canonical term
            for variant in variants:
                if variant != canonical_term:
                    standardization[variant.lower()] = canonical_term.lower()
        
        return standardization
    
    def normalize_text(self, text: str) -> str:
        """
        Comprehensive text normalization for hadith queries.
        
        Args:
            text (str): Input text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove quotes and special characters but keep apostrophes and hyphens
        text = re.sub(r'["""\'`""''«»]', '', text)
        
        # Normalize Islamic honorific phrases (remove but mark for context)
        text = re.sub(r'shallallahu\s+[\'\'"]?alaihi\s+wa?\s*sallam', '', text)
        text = re.sub(r'sallallahu\s+[\'\'"]?alaihi\s+wa?\s*sallam', '', text)
        text = re.sub(r'radhi\s*allahu\s+(anhu|anha|anhum)', '', text)
        text = re.sub(r'radhiyallahu\s+(anhu|anha|anhum)', '', text)
        text = re.sub(r'rahimahullah\s*', '', text)
        
        # Remove Arabic script remnants but preserve transliteration
        text = re.sub(r'[ء-ي]+', ' ', text)
        
        # Clean up excessive punctuation but preserve sentence structure
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        text = re.sub(r'\s*-\s*', ' ', text)  # Clean up dashes
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def apply_conservative_lemmatization(self, text: str) -> str:
        """
        Apply conservative lemmatization that preserves query intent and Islamic terminology.
        
        Args:
            text (str): Text to lemmatize
            
        Returns:
            str: Lemmatized text
        """
        words = text.split()
        lemmatized_words = []
        
        for word in words:
            # Check exact matches first (highest priority)
            if word in self.conservative_lemmatization_rules:
                lemmatized_words.append(self.conservative_lemmatization_rules[word])
            # Check term standardization from keywords map
            elif word in self.term_standardization:
                lemmatized_words.append(self.term_standardization[word])
            else:
                # Apply very conservative suffix/prefix rules
                lemmatized_word = self._conservative_suffix_removal(word)
                lemmatized_word = self._conservative_prefix_removal(lemmatized_word)
                lemmatized_words.append(lemmatized_word)
        
        return ' '.join(lemmatized_words)
    
    def _conservative_suffix_removal(self, word: str) -> str:
        """
        Apply conservative suffix removal rules.
        Only removes suffixes if the result is semantically meaningful.
        """
        # Skip very short words
        if len(word) <= 4:
            return word
        
        # Only remove possessive suffixes for very long words (>6 chars)
        if len(word) > 6:
            if word.endswith('nya'):
                base = word[:-3]
                # Only if base is meaningful
                if len(base) >= 4 and not base.endswith(('ng', 'ng')):
                    return base
            elif word.endswith(('mu', 'ku')) and len(word) > 5:
                return word[:-2]
        
        # Very conservative -an suffix removal 
        if word.endswith('an') and len(word) > 6:
            base = word[:-2]
            # Only remove if it results in a known meaningful root
            meaningful_roots = {
                'minum', 'makan', 'shalat', 'puasa', 'zakat', 'nikah',
                'daging', 'hewan', 'makanan', 'pernik', 'buat', 'laku',
                'ajar', 'kerja', 'laksana', 'main', 'baca', 'tulis'
            }
            if base in meaningful_roots:
                return base
        
        return word
    
    def _conservative_prefix_removal(self, word: str) -> str:
        """
        Apply very conservative prefix removal.
        Preserves query intent words and important Islamic terms.
        """
        # Skip if word should remain intact
        keep_intact = {
            'berikan', 'jelaskan', 'sebutkan', 'bagaimana', 'dimana',
            'mengapa', 'perang', 'sedekah', 'ramadan', 'minuman',
            'tunjukkan', 'terangkan', 'beritahu'
        }
        
        if word.lower() in keep_intact:
            return word
        
        # Only apply prefix removal for specific known mappings
        specific_mappings = {
            'berwudhu': 'wudhu',
            'berpuasa': 'puasa', 
            'berzakat': 'zakat',
            'bershalat': 'shalat',
            'mengharamkan': 'haram',
            'menghalalkan': 'halal',
            'memperbolehkan': 'boleh',
            'melaksanakan': 'laksana'
        }
        
        if word in specific_mappings:
            return specific_mappings[word]
        
        return word
    
    def remove_stopwords(self, text: str, preserve_query_intent: bool = True) -> str:
        """
        Remove stopwords while preserving Islamic terms and query intent.
        
        Args:
            text (str): Input text
            preserve_query_intent (bool): Whether to preserve query intent words
            
        Returns:
            str: Text with stopwords removed
        """
        words = text.split()
        filtered_words = []
        
        # Words to preserve even if they're in stopwords
        preserve_words = {
            'berikan', 'jelaskan', 'sebutkan', 'bagaimana', 'apa',
            'dimana', 'kapan', 'mengapa', 'siapa', 'berapa'
        } if preserve_query_intent else set()
        
        for word in words:
            # Keep if not in stopwords, or if it's a preserved query word, or if very short
            if (word not in self.all_stopwords or 
                word in preserve_words or 
                len(word) <= 2):
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def preprocess_query(self, query: str, remove_stopwords: bool = True) -> str:
        """
        Complete enhanced query preprocessing pipeline.
        
        Args:
            query (str): Raw query from user
            remove_stopwords (bool): Whether to remove stopwords
            
        Returns:
            str: Preprocessed query
        """
        if not query:
            return ""
        
        # Step 1: Normalize text
        processed = self.normalize_text(query)
        
        # Step 2: Apply conservative lemmatization with standardization
        processed = self.apply_conservative_lemmatization(processed)
        
        # Step 3: Remove stopwords while preserving query intent
        if remove_stopwords:
            processed = self.remove_stopwords(processed, preserve_query_intent=True)
        
        # Step 4: Final cleanup
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key terms from query for enhanced keyword matching.
        
        Args:
            query (str): Input query
            
        Returns:
            List[str]: List of key terms sorted by importance
        """
        # Preprocess but keep some stopwords for context
        processed = self.preprocess_query(query, remove_stopwords=False)
        
        # Split into terms
        terms = processed.split()
        
        # Filter and rank terms
        key_terms = []
        
        for term in terms:
            # Skip pure stopwords and very short terms
            if term not in self.all_stopwords and len(term) >= 3:
                key_terms.append(term)
        
        # Prioritize Islamic terms and important concepts
        islamic_priority = []
        other_terms = []
        
        for term in key_terms:
            # Check if term is in keywords map (Islamic term)
            is_islamic = any(term in variants for variants in self.keywords_map.values())
            
            if is_islamic or term in self.conservative_lemmatization_rules.values():
                islamic_priority.append(term)
            else:
                other_terms.append(term)
        
        # Return Islamic terms first, then others
        return islamic_priority + other_terms
    
    def analyze_query_intent(self, query: str) -> Dict[str, any]:
        """
        Analyze query to understand user intent and extract metadata.
        
        Args:
            query (str): Input query
            
        Returns:
            Dict: Analysis results including intent, key terms, and context
        """
        original_lower = query.lower()
        processed = self.preprocess_query(query)
        key_terms = self.extract_key_terms(query)
        
        # Detect query type
        question_words = ['apa', 'bagaimana', 'dimana', 'kapan', 'mengapa', 'siapa', 'berapa']
        has_question = any(qw in original_lower for qw in question_words)
        
        # Detect action intent
        action_words = ['berikan', 'jelaskan', 'sebutkan', 'tunjukkan', 'terangkan']
        has_action = any(aw in original_lower for aw in action_words)
        
        # Detect Islamic context strength
        islamic_terms_found = [term for term in key_terms 
                              if any(term in variants for variants in self.keywords_map.values())]
        
        return {
            'original_query': query,
            'processed_query': processed,
            'key_terms': key_terms,
            'islamic_terms': islamic_terms_found,
            'has_question': has_question,
            'has_action_intent': has_action,
            'islamic_context_strength': len(islamic_terms_found) / max(len(key_terms), 1),
            'query_length': len(query.split()),
            'processed_length': len(processed.split())
        }


# Global instance for easy import
enhanced_preprocessor = EnhancedQueryPreprocessor()


def preprocess_query(query: str, remove_stopwords: bool = True) -> str:
    """
    Convenience function for enhanced query preprocessing.
    
    Args:
        query (str): Raw query from user
        remove_stopwords (bool): Whether to remove stopwords
        
    Returns:
        str: Preprocessed query
    """
    return enhanced_preprocessor.preprocess_query(query, remove_stopwords)


def extract_key_terms(query: str) -> List[str]:
    """
    Convenience function for enhanced key term extraction.
    
    Args:
        query (str): Input query
        
    Returns:
        List[str]: List of key terms
    """
    return enhanced_preprocessor.extract_key_terms(query)


def analyze_query_intent(query: str) -> Dict[str, any]:
    """
    Convenience function for query intent analysis.
    
    Args:
        query (str): Input query
        
    Returns:
        Dict: Analysis results
    """
    return enhanced_preprocessor.analyze_query_intent(query)


# Test function
if __name__ == "__main__":
    print("=== Enhanced Query Preprocessor - Fixed V1 Test ===")
    
    test_queries = [
        "Apa hukum shalat jum'at bagi wanita?",
        "Bagaimana cara berwudhu yang benar menurut hadits?", 
        "Berapa kali melakukan shalat dalam sehari?",
        "Apakah puasa wajib bagi semua muslim?",
        "Bagaimana hukum mengharamkan makanan halal?",
        "Jelaskan tentang zakat fitrah dan zakat mal",
        "Hukum minuman keras dalam Islam"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i} ---")
        processed = preprocess_query(query)
        key_terms = extract_key_terms(query)
        analysis = analyze_query_intent(query)
        
        print(f"Original: {query}")
        print(f"Processed: {processed}")
        print(f"Key terms: {key_terms}")
        print(f"Islamic terms: {analysis['islamic_terms']}")
        print(f"Context strength: {analysis['islamic_context_strength']:.2f}")
        print(f"Has question: {analysis['has_question']}")
        print(f"Has action: {analysis['has_action_intent']}")