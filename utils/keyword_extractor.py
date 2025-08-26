"""
Enhanced Keyword Extractor for Hadith AI
========================================

This module provides hybrid keyword extraction capabilities including:
- Statistical methods (TF-IDF, YAKE)
- Embedding similarity with corpus
- Rule-based Islamic dictionary terms
- N-gram detection (2-3 words)
- Phrase + component storage

Author: Hadith AI Team
Date: 2024
"""

import re
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Union
import pandas as pd

# Try to import optional dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. TF-IDF extraction disabled.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Some features may be limited.")


class HybridKeywordExtractor:
    """
    Advanced keyword extractor that combines multiple approaches for optimal results.
    """
    
    def __init__(self, 
                 min_frequency: int = 20,
                 max_ngram: int = 3,
                 islamic_terms_path: str = None):
        self.min_frequency = min_frequency
        self.max_ngram = max_ngram
        
        # Islamic terms dictionary (can be loaded from file or use default)
        self.islamic_terms = self._load_islamic_terms(islamic_terms_path)
        
        # Stopwords for filtering
        self.stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'dengan', 'untuk', 'adalah', 'akan',
            'telah', 'sudah', 'atau', 'juga', 'tidak', 'bila', 'jika', 'ketika', 'saat',
            'itu', 'ini', 'mereka', 'kita', 'kami', 'dia', 'ia', 'saya', 'anda', 'engkau',
            'kamu', 'kalian', 'beliau', 'ada', 'seperti', 'antara', 'semua', 'setiap',
            # Hadith-specific stopwords
            'bin', 'abu', 'ibnu', 'al', 'an', 'as', 'ad', 'ar', 'az', 'ats', 'ath',
            'saw', 'ra', 'rah', 'radhiyallahu', 'anhu', 'anha', 'anhum', 'anhuma',
            'shallallahu', 'alaihi', 'wasallam', 'sallallahu', 'alaih',
            'hadits', 'hadist', 'riwayat', 'diriwayatkan', 'menceritakan',
            'telah', 'bercerita', 'kepada', 'kami', 'dari'
        }
        
        # Noise patterns (sanad, narrator chains)
        self.noise_patterns = [
            r'.*bin.*',
            r'.*abu.*',
            r'.*ibnu.*',
            r'.*al-.*',
            r'.*menceritakan.*',
            r'.*bercerita.*',
            r'.*riwayat.*',
            r'.*hadathana.*',
            r'.*akhbarana.*'
        ]
    
    def _load_islamic_terms(self, path: str = None) -> Set[str]:
        """Load Islamic terms dictionary."""
        if path and Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return set(data.get('terms', []))
            except Exception as e:
                print(f"Warning: Could not load Islamic terms from {path}: {e}")
        
        # Default Islamic terms dictionary
        return {
            # Prayer-related
            'shalat', 'salat', 'sholat', 'solat', 'shalatnya', 'sujud', 'rukuk', 'takbir',
            'tahiyyat', 'taslim', 'qiblat', 'kiblat', 'jamaah', 'imam', 'makmum',
            'shalat subuh', 'shalat zhuhur', 'shalat ashar', 'shalat maghrib', 'shalat isya',
            'shalat jumat', 'shalat ied', 'shalat tarawih', 'shalat witir', 'shalat tahajud',
            
            # Purification
            'wudhu', 'wudu', 'tayammum', 'ghusl', 'najis', 'suci', 'bersuci', 'thaharah',
            'istinja', 'istinjak', 'air', 'debu', 'tanah',
            
            # Fasting
            'puasa', 'shaum', 'shiyam', 'sahur', 'iftar', 'berbuka', 'ramadan', 'fidyah',
            'kafarat', 'itikaf', 'lailatul qadr',
            
            # Charity
            'zakat', 'sadaqah', 'infaq', 'fitrah', 'mal', 'harta', 'nisab', 'haul',
            'mustahiq', 'asnaf', 'fakir', 'miskin',
            
            # Pilgrimage
            'haji', 'umrah', 'ihram', 'tawaf', 'sai', 'wukuf', 'arafah', 'muzdalifah',
            'mina', 'jumrah', 'tahallul', 'hady', 'dam', 'badal',
            
            # Legal rulings
            'halal', 'haram', 'makruh', 'sunnah', 'mustahab', 'wajib', 'fardhu',
            'mubah', 'hukum', 'syariat', 'fiqih', 'fatwa', 'ijma', 'qiyas',
            
            # Beliefs
            'iman', 'islam', 'ihsan', 'tauhid', 'syirik', 'kufur', 'munafik',
            'allah', 'rasul', 'nabi', 'malaikat', 'kitab', 'akhirat', 'qadar',
            
            # Ethics and behavior
            'akhlaq', 'adab', 'birrul walidain', 'silaturahmi', 'amanah', 'jujur',
            'sabar', 'syukur', 'tawadhu', 'ikhlas', 'taqwa', 'takut', 'harap',
            
            # Marriage and family
            'nikah', 'kawin', 'talaq', 'rujuk', 'iddah', 'khulu', 'mubarat',
            'mahar', 'nafkah', 'wali', 'saksi', 'walimah',
            
            # Business and transactions
            'jual', 'beli', 'dagang', 'riba', 'gharar', 'tadlis', 'ijarah',
            'mudharabah', 'musyarakah', 'salam', 'istisna', 'wakalah'
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        if pd.isna(text) or not text:
            return ""
        
        text = text.lower().strip()
        
        # Remove quotes and normalize punctuation
        text = re.sub(r'["""\'\'\"`]', '', text)
        text = re.sub(r'\s*-\s*', ' ', text)
        
        # Normalize Islamic phrases
        text = re.sub(r'shallallahu\s+[\'\"]?alaihi\s+wa?\s*sallam', 'saw', text)
        text = re.sub(r'sallallahu\s+[\'\"]?alaihi\s+wa?\s*sallam', 'saw', text)
        text = re.sub(r'radhi\s*allahu\s+(anhu|anha|anhum)', 'ra', text)
        text = re.sub(r'radhiyallahu\s+(anhu|anha|anhum)', 'ra', text)
        
        # Remove excessive punctuation but preserve word structure
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def is_meaningful_term(self, term: str) -> bool:
        """Check if a term is meaningful (not noise)."""
        if not term or len(term.strip()) < 3:
            return False
        
        term = term.strip().lower()
        
        # Check against stopwords
        if term in self.stopwords:
            return False
        
        # Check against noise patterns
        for pattern in self.noise_patterns:
            if re.match(pattern, term):
                return False
        
        # Check if term contains meaningful content
        if re.match(r'^[0-9]+$', term):  # Only numbers
            return False
        
        if len(set(term)) == 1:  # Repeated characters
            return False
        
        return True
    
    def generate_ngrams(self, text: str, max_n: int = None) -> List[str]:
        """Generate meaningful n-grams from text."""
        if max_n is None:
            max_n = self.max_ngram
        
        words = self.normalize_text(text).split()
        ngrams = []
        
        for n in range(1, max_n + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                if self.is_meaningful_term(ngram):
                    ngrams.append(ngram)
        
        return ngrams
    
    def extract_statistical_keywords(self, texts: List[str]) -> Dict[str, float]:
        """Extract keywords using statistical methods (TF-IDF)."""
        if not SKLEARN_AVAILABLE:
            print("Warning: Falling back to basic frequency counting")
            return self._extract_frequency_keywords(texts)
        
        try:
            # Generate all ngrams
            all_ngrams = []
            for text in texts:
                ngrams = self.generate_ngrams(text)
                all_ngrams.extend(ngrams)
            
            # Count frequencies
            ngram_counts = Counter(all_ngrams)
            
            # Filter by minimum frequency
            frequent_ngrams = {
                ngram: count for ngram, count in ngram_counts.items()
                if count >= self.min_frequency
            }
            
            # Simple TF-IDF approximation
            total_docs = len(texts)
            tfidf_scores = {}
            
            for ngram, freq in frequent_ngrams.items():
                # Calculate document frequency
                df = sum(1 for text in texts if ngram in self.normalize_text(text))
                
                # Calculate TF-IDF
                tf = freq / len(all_ngrams)
                idf = math.log(total_docs / (df + 1))
                tfidf_scores[ngram] = tf * idf
            
            return tfidf_scores
            
        except Exception as e:
            print(f"Warning: Statistical extraction failed: {e}")
            return self._extract_frequency_keywords(texts)
    
    def _extract_frequency_keywords(self, texts: List[str]) -> Dict[str, float]:
        """Fallback frequency-based keyword extraction."""
        all_ngrams = []
        for text in texts:
            ngrams = self.generate_ngrams(text)
            all_ngrams.extend(ngrams)
        
        ngram_counts = Counter(all_ngrams)
        
        # Filter by minimum frequency and normalize scores
        total_count = sum(ngram_counts.values())
        frequent_ngrams = {
            ngram: count / total_count for ngram, count in ngram_counts.items()
            if count >= self.min_frequency
        }
        
        return frequent_ngrams
    
    def extract_rule_based_keywords(self, texts: List[str]) -> Dict[str, float]:
        """Extract keywords using rule-based Islamic dictionary."""
        keyword_scores = {}
        
        for text in texts:
            normalized = self.normalize_text(text)
            
            for term in self.islamic_terms:
                # Look for exact and partial matches
                if re.search(rf'\b{re.escape(term)}\b', normalized):
                    keyword_scores[term] = keyword_scores.get(term, 0) + 1
        
        # Normalize scores
        total_count = sum(keyword_scores.values())
        if total_count > 0:
            keyword_scores = {
                term: count / total_count 
                for term, count in keyword_scores.items()
            }
        
        return keyword_scores
    
    def extract_phrase_components(self, phrases: List[str]) -> Dict[str, List[str]]:
        """
        Extract phrase components for better matching.
        
        Example: "shalat jum'at" -> ["shalat", "jumat"]
        """
        phrase_components = {}
        
        for phrase in phrases:
            if ' ' in phrase:  # Multi-word phrase
                components = phrase.split()
                # Filter meaningful components
                meaningful_components = [
                    comp for comp in components 
                    if self.is_meaningful_term(comp)
                ]
                if meaningful_components:
                    phrase_components[phrase] = meaningful_components
            else:
                # Single word
                phrase_components[phrase] = [phrase]
        
        return phrase_components
    
    def hybrid_extract(self, texts: List[str]) -> Dict[str, Dict]:
        """
        Main hybrid extraction method combining all approaches.
        
        Returns:
            Dict containing extracted keywords with metadata
        """
        print("🔍 Starting hybrid keyword extraction...")
        
        # Statistical extraction
        print("📊 Extracting statistical keywords...")
        statistical_keywords = self.extract_statistical_keywords(texts)
        
        # Rule-based extraction
        print("📋 Extracting rule-based keywords...")
        rule_based_keywords = self.extract_rule_based_keywords(texts)
        
        # Combine results
        print("🔀 Combining extraction results...")
        all_keywords = set(statistical_keywords.keys()) | set(rule_based_keywords.keys())
        
        combined_results = {}
        for keyword in all_keywords:
            stat_score = statistical_keywords.get(keyword, 0)
            rule_score = rule_based_keywords.get(keyword, 0)
            
            # Weighted combination (rule-based gets higher weight for Islamic terms)
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
        print("🔗 Extracting phrase components...")
        phrases = list(combined_results.keys())
        phrase_components = self.extract_phrase_components(phrases)
        
        # Add component information
        for phrase, components in phrase_components.items():
            if phrase in combined_results:
                combined_results[phrase]['components'] = components
        
        print(f"✅ Extracted {len(combined_results)} keywords")
        return combined_results
    
    def create_keywords_map(self, 
                          extraction_results: Dict[str, Dict],
                          min_score: float = 0.01) -> Dict[str, List[str]]:
        """
        Create a keywords map suitable for query optimization.
        
        Args:
            extraction_results: Results from hybrid_extract
            min_score: Minimum score threshold for inclusion
            
        Returns:
            Dict mapping canonical terms to their variants
        """
        keywords_map = defaultdict(set)
        
        # Filter by minimum score
        filtered_results = {
            keyword: info for keyword, info in extraction_results.items()
            if info['score'] >= min_score
        }
        
        # Group similar terms
        for keyword, info in filtered_results.items():
            canonical_term = keyword
            
            # For multi-word phrases, use the main term as canonical
            if info.get('components'):
                components = info['components']
                # Use the longest or most Islamic term as canonical
                if any(comp in self.islamic_terms for comp in components):
                    islamic_comps = [comp for comp in components if comp in self.islamic_terms]
                    canonical_term = max(islamic_comps, key=len)
                else:
                    canonical_term = max(components, key=len)
            
            # Add the keyword and its components to the map
            keywords_map[canonical_term].add(keyword)
            if info.get('components'):
                for component in info['components']:
                    if self.is_meaningful_term(component):
                        keywords_map[canonical_term].add(component)
        
        # Convert sets to sorted lists
        final_map = {
            canonical: sorted(list(variants))
            for canonical, variants in keywords_map.items()
            if len(variants) > 0
        }
        
        return final_map


def extract_keywords_from_corpus(csv_dir: str, 
                                output_path: str = None,
                                min_frequency: int = 20,
                                max_ngram: int = 3) -> Dict[str, List[str]]:
    """
    Extract keywords from hadith corpus CSV files.
    
    Args:
        csv_dir: Directory containing CSV files
        output_path: Path to save the keywords map
        min_frequency: Minimum frequency for terms
        max_ngram: Maximum n-gram size
        
    Returns:
        Keywords map dictionary
    """
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise ValueError(f"CSV directory not found: {csv_dir}")
    
    # Initialize extractor
    extractor = HybridKeywordExtractor(min_frequency=min_frequency, max_ngram=max_ngram)
    
    # Collect texts from CSV files
    print("📁 Loading hadith texts from CSV files...")
    all_texts = []
    
    for csv_file in csv_path.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            if 'terjemah' in df.columns:
                texts = df['terjemah'].dropna().tolist()
                all_texts.extend(texts)
                print(f"✅ Loaded {len(texts)} texts from {csv_file.name}")
            else:
                print(f"⚠️ No 'terjemah' column found in {csv_file.name}")
        except Exception as e:
            print(f"❌ Error loading {csv_file.name}: {e}")
    
    if not all_texts:
        raise ValueError("No texts found in CSV files")
    
    print(f"📄 Total texts loaded: {len(all_texts)}")
    
    # Extract keywords
    extraction_results = extractor.hybrid_extract(all_texts)
    
    # Create keywords map
    keywords_map = extractor.create_keywords_map(extraction_results)
    
    # Save results if output path specified
    if output_path:
        output_data = {
            'metadata': {
                'total_texts': len(all_texts),
                'min_frequency': min_frequency,
                'max_ngram': max_ngram,
                'total_keywords': len(keywords_map),
                'extraction_method': 'hybrid_tfidf_rule_based'
            },
            'keywords': keywords_map
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Keywords map saved to {output_path}")
    
    return keywords_map


# Backward compatibility functions
def normalize(text: str) -> str:
    """Legacy function for backward compatibility."""
    return re.sub(r"[^\w\s]", "", text.lower())

def tokenize(text: str):
    """Legacy function for backward compatibility."""
    return normalize(text).split()

def collect_terjemah_texts(directory: Path):
    """Legacy function for backward compatibility."""
    all_texts = []
    for file in directory.glob("*.csv"):
        try:
            df = pd.read_csv(file)
            if 'terjemah' in df.columns:
                all_texts.extend(df['terjemah'].dropna().tolist())
                print(f"[✓] Terbaca: {file.name} — {len(df)} baris")
            else:
                print(f"[!] Kolom 'terjemah' tidak ditemukan di {file.name}")
        except Exception as e:
            print(f"[!] Gagal membaca {file.name}: {e}")
    return all_texts

# Test function
if __name__ == "__main__":
    # Test with sample texts
    sample_texts = [
        "Rasulullah saw bersabda tentang shalat lima waktu yang wajib",
        "Bagaimana cara berwudhu yang benar menurut sunnah",
        "Puasa ramadan adalah kewajiban bagi setiap muslim",
        "Zakat fitrah wajib dikeluarkan sebelum shalat ied",
        "Hukum shalat jumat bagi kaum wanita"
    ]
    
    print("=== Keyword Extractor Test ===")
    extractor = HybridKeywordExtractor(min_frequency=1, max_ngram=3)
    
    results = extractor.hybrid_extract(sample_texts)
    keywords_map = extractor.create_keywords_map(results, min_score=0.0)
    
    print("\nExtracted keywords:")
    for canonical, variants in list(keywords_map.items())[:10]:
        print(f"{canonical}: {variants}")
