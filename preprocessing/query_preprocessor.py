"""
Query Preprocessor Module for Hadith AI
========================================

This module provides enhanced query preprocessing capabilities including:
- Text normalization (lowercase, punctuation removal, stopword filtering)
- Light lemmatization for Indonesian terms
- Islamic terms standardization
- Query sanitization

Author: Hadith AI Team
Date: 2024
"""

import re
import unicodedata
from typing import Set, Dict, List


class QueryPreprocessor:
    """
    Advanced query preprocessor for hadith queries with Indonesian and Arabic text support.
    """
    
    def __init__(self):
        # Enhanced Indonesian stopwords with query-specific terms
        self.indonesian_stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'dengan', 'untuk', 'adalah', 'akan',
            'telah', 'sudah', 'atau', 'juga', 'tidak', 'bila', 'jika', 'ketika', 'saat',
            'itu', 'ini', 'mereka', 'kita', 'kami', 'dia', 'ia', 'saya', 'anda', 'engkau',
            'kamu', 'kalian', 'beliau', 'ada', 'seperti', 'antara', 'semua', 'setiap',
            'bagi', 'oleh', 'karena', 'sebab', 'supaya', 'agar', 'hingga', 'sampai',
            'maka', 'lalu', 'kemudian', 'setelah', 'sebelum', 'selama', 'sambil',
            
            # Query-specific stopwords
            'apa', 'bagaimana', 'dimana', 'kapan', 'mengapa', 'siapa', 'berapa',
            'apa itu', 'berikan', 'jelaskan', 'sebutkan', 'tentang', 'mengenai',
            'terkait', 'berhubungan', 'cara', 'metode', 'langkah'
        }
        
        # Arabic transliteration stopwords
        self.arabic_stopwords = {
            'al', 'an', 'fi', 'min', 'ila', 'wa', 'la', 'ma', 'li', 'bi',
            'qala', 'qaala', 'hadathana', 'akhbarana', 'anna', 'alladhii',
            'bin', 'abu', 'ibnu', 'ibn', 'as', 'ad', 'ar', 'az', 'ats', 'ath'
        }
        
        # Hadith-specific stopwords (narrator chains, common phrases)
        self.hadith_stopwords = {
            'saw', 'ra', 'rah', 'radhiyallahu', 'anhu', 'anha', 'anhum', 'anhuma',
            'shallallahu', 'alaihi', 'wasallam', 'sallallahu', 'alaih', 'alayhi',
            'hadits', 'hadist', 'riwayat', 'diriwayatkan', 'menceritakan',
            'telah', 'bercerita', 'kepada', 'kami', 'dari'
        }
        
        # Combine all stopwords
        self.all_stopwords = (
            self.indonesian_stopwords | 
            self.arabic_stopwords | 
            self.hadith_stopwords
        )
        
        # Enhanced lemmatization rules for Indonesian Islamic terms
        self.lemmatization_rules = {
            # Worship practices - improved mapping
            'berwudhu': 'wudhu',
            'berwudu': 'wudhu', 
            'bertayammum': 'tayammum',
            'bershalat': 'shalat',
            'menshalatkan': 'shalat',
            'menshalati': 'shalat',
            'menyalatkan': 'shalat',
            'berpuasa': 'puasa',
            'berzakat': 'zakat',
            
            # Action words that should retain meaning
            'berikan': 'berikan',  # Keep this for query intent
            'jelaskan': 'jelaskan',  # Keep this for query intent  
            'sebutkan': 'sebutkan',  # Keep this for query intent
            'mengeluarkan': 'keluarkan',  # Better than 'keluar'
            'mengerjakan': 'kerjakan',    # Better than 'kerja'
            'melaksanakan': 'laksanakan', # Better than 'laksana'
            'melakukan': 'lakukan',       # Better than 'laku'
            
            # Permissibility terms
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
            
            # Possessive forms
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
            
            # Spelling variations
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
            
            # Time-specific terms
            'minuman': 'minum',  # Keep root for better matching
            'keras': 'keras',    # Important modifier
            'perang': 'perang',  # Keep full word
            'jihad': 'jihad',    # Keep full word
            'syahid': 'syahid',  # Keep full word
            'berbakti': 'bakti', # Keep root
            'orang': 'orang',    # Keep full word
            'tua': 'tua'         # Keep full word
        }
    
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
        
        # Remove quotes and special characters but keep apostrophes
        text = re.sub(r'["""\'`""''«»]', '', text)
        
        # Normalize Islamic phrases
        text = re.sub(r'shallallahu\s+[\'\'"]?alaihi\s+wa?\s*sallam', 'saw', text)
        text = re.sub(r'sallallahu\s+[\'\'"]?alaihi\s+wa?\s*sallam', 'saw', text)
        text = re.sub(r'radhi\s*allahu\s+(anhu|anha|anhum)', 'ra', text)
        text = re.sub(r'radhiyallahu\s+(anhu|anha|anhum)', 'ra', text)
        
        # Remove excessive punctuation but preserve sentence structure
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        text = re.sub(r'\s*-\s*', ' ', text)  # Clean up dashes
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def apply_lemmatization(self, text: str) -> str:
        """
        Apply light lemmatization rules to Indonesian text.
        
        Args:
            text (str): Text to lemmatize
            
        Returns:
            str: Lemmatized text
        """
        words = text.split()
        lemmatized_words = []
        
        for word in words:
            # Check exact matches first
            if word in self.lemmatization_rules:
                lemmatized_words.append(self.lemmatization_rules[word])
            else:
                # Check for common suffixes and prefixes
                lemmatized_word = self._apply_suffix_rules(word)
                lemmatized_word = self._apply_prefix_rules(lemmatized_word)
                lemmatized_words.append(lemmatized_word)
        
        return ' '.join(lemmatized_words)
    
    def _apply_suffix_rules(self, word: str) -> str:
        """Apply conservative suffix removal rules for queries."""
        # Be more conservative with suffix removal for queries
        
        # Only remove possessive suffixes for very long words
        if word.endswith(('nya', 'mu', 'ku')) and len(word) > 6:
            if word.endswith('nya'):
                return word[:-3]
            elif word.endswith(('mu', 'ku')):
                return word[:-2]
        
        # More conservative with -an suffix removal
        if word.endswith('an') and len(word) > 6:
            base = word[:-2]
            # Only remove if it results in a known Islamic term or common word
            common_roots = {
                'minum', 'makan', 'shalat', 'puasa', 'zakat', 'nikah',
                'daging', 'hewan', 'makanan', 'pernik', 'buat', 'laku'
            }
            if base in common_roots:
                return base
        
        return word
    
    def _apply_prefix_rules(self, word: str) -> str:
        """Apply conservative prefix removal rules for queries."""
        # Be very conservative with prefix removal in queries
        # Only remove prefixes if the result is a known meaningful word
        
        # Skip prefix removal for important query words
        keep_intact = {
            'berikan', 'jelaskan', 'sebutkan', 'bagaimana', 'dimana',
            'mengapa', 'perang', 'sedekah', 'ramadan', 'minuman'
        }
        
        if word.lower() in keep_intact:
            return word
            
        # Very conservative prefix removal
        conservative_prefixes = {
            'ber': {'berwudhu': 'wudhu', 'berpuasa': 'puasa', 'berzakat': 'zakat'},
            'men': {'mengharamkan': 'haram', 'menghalalkan': 'halal'},
            'mem': {'memperbolehkan': 'boleh'}
        }
        
        for prefix, mappings in conservative_prefixes.items():
            if word in mappings:
                return mappings[word]
        
        return word
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text while preserving Islamic terms.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with stopwords removed
        """
        words = text.split()
        filtered_words = []
        
        for word in words:
            # Keep Islamic terms and non-stopwords
            if word not in self.all_stopwords or len(word) <= 2:
                filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def preprocess_query(self, query: str, remove_stopwords: bool = True) -> str:
        """
        Complete query preprocessing pipeline.
        
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
        
        # Step 2: Apply lemmatization
        processed = self.apply_lemmatization(processed)
        
        # Step 3: Remove stopwords (optional)
        if remove_stopwords:
            processed = self.remove_stopwords(processed)
        
        # Step 4: Final cleanup
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def extract_key_terms(self, query: str) -> List[str]:
        """
        Extract key terms from query for keyword matching.
        
        Args:
            query (str): Input query
            
        Returns:
            List[str]: List of key terms
        """
        processed = self.preprocess_query(query, remove_stopwords=True)
        terms = processed.split()
        
        # Filter out very short terms
        key_terms = [term for term in terms if len(term) >= 3]
        
        return key_terms


# Global instance for easy import
preprocessor = QueryPreprocessor()


def preprocess_query(query: str, remove_stopwords: bool = True) -> str:
    """
    Convenience function for query preprocessing.
    
    Args:
        query (str): Raw query from user
        remove_stopwords (bool): Whether to remove stopwords
        
    Returns:
        str: Preprocessed query
    """
    return preprocessor.preprocess_query(query, remove_stopwords)


def extract_key_terms(query: str) -> List[str]:
    """
    Convenience function for key term extraction.
    
    Args:
        query (str): Input query
        
    Returns:
        List[str]: List of key terms
    """
    return preprocessor.extract_key_terms(query)


# Test function
if __name__ == "__main__":
    # Test the preprocessor
    test_queries = [
        "Apa hukum shalat jum'at bagi wanita?",
        "Bagaimana cara berwudhu yang benar?", 
        "Berapa kali melakukan shalat dalam sehari?",
        "Apakah puasa wajib bagi semua muslim?",
        "Bagaimana hukum mengharamkan makanan halal?"
    ]
    
    print("=== Query Preprocessor Test ===")
    for query in test_queries:
        processed = preprocess_query(query)
        key_terms = extract_key_terms(query)
        
        print(f"\nOriginal: {query}")
        print(f"Processed: {processed}")
        print(f"Key terms: {key_terms}")