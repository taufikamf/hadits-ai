"""
Enhanced Query Optimizer for Hadith AI
======================================

This module provides enhanced query optimization with:
- Keyword map integration (canonical + synonyms) 
- Synonym expansion (e.g., "haram" → "mengharamkan")
- Phrase + component retention for better matching
- Integration with query preprocessor
- Semantic rule-based expansion

Author: Hadith AI Team
Date: 2024
"""

from preprocessing.query_preprocessor import preprocess_query, extract_key_terms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import re
import logging
from typing import List, Tuple, Set, Dict


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedQueryOptimizer:
    """Enhanced query optimizer with comprehensive keyword mapping and expansion."""
    
    def __init__(self, keywords_map_path: str = "data/processed/keywords_map_grouped.json"):
        """
        Initialize the enhanced query optimizer.
        
        Args:
            keywords_map_path: Path to the keywords map JSON file
        """
        self.keywords_map_path = keywords_map_path
        self.keywords_map = self._load_keywords_map()
        self.reverse_map = self._build_reverse_map()
        self.sorted_phrases = self._build_sorted_phrases()
        
        # Enhanced semantic expansion rules
        self.semantic_expansion_rules = {
            'hukum': [
                'halal', 'haram', 'makruh', 'sunnah', 'wajib', 'mubah',
                'mengharamkan', 'menghalalkan', 'memakruhkan', 'menyunnahkan',
                'mewajibkan', 'membolehkan', 'dilarang', 'dibolehkan', 
                'diperbolehkan', 'diwajibkan', 'disunnahkan', 'dimakruhkan',
                'diharamkan', 'dihalalkan'
            ],
            'shalat': [
                'shalat', 'salat', 'sholat', 'solat', 'shalatnya', 'menshalatkan',
                'bershalat', 'mengerjakan shalat', 'melaksanakan shalat',
                'mendirikan shalat', 'sujud', 'rukuk', 'takbir', 'tahiyyat'
            ],
            'puasa': [
                'puasa', 'shaum', 'shiyam', 'berpuasa', 'menjalankan puasa',
                'melaksanakan puasa', 'sahur', 'iftar', 'berbuka', 'buka puasa'
            ],
            'zakat': [
                'zakat', 'berzakat', 'mengeluarkan zakat', 'membayar zakat',
                'menunaikan zakat', 'sadaqah', 'infaq', 'fitrah'
            ],
            'wudhu': [
                'wudhu', 'wudu', 'berwudhu', 'berwudu', 'mengambil wudhu',
                'bersuci', 'tayammum', 'bertayammum', 'ghusl', 'mandi besar'
            ],
            'nikah': [
                'nikah', 'kawin', 'menikah', 'bernikah', 'kawin', 'perkawinan',
                'pernikahan', 'talaq', 'rujuk', 'bercerai', 'perceraian'
            ],
            'jual': [
                'jual', 'beli', 'jual beli', 'berjual', 'membeli', 'menjual',
                'perdagangan', 'dagang', 'berdagang', 'berniaga', 'transaksi'
            ]
        }
        
        # Synonym mappings for better expansion
        self.synonym_mappings = {
            'haram': ['haram', 'mengharamkan', 'diharamkan', 'keharaman', 'terlarang'],
            'halal': ['halal', 'menghalalkan', 'dihalalkan', 'kehalalan', 'boleh'],
            'wajib': ['wajib', 'mewajibkan', 'diwajibkan', 'kewajiban', 'fardhu'],
            'sunnah': ['sunnah', 'mustahab', 'disunnahkan', 'menyunnahkan'],
            'makruh': ['makruh', 'dimakruhkan', 'memakruhkan', 'kemakruhan'],
            'rasul': ['rasul', 'rasulullah', 'nabi', 'nabi muhammad'],
            'allah': ['allah', 'tuhan', 'sang pencipta', 'rabb'],
            'iman': ['iman', 'beriman', 'keimanan', 'percaya'],
            'islam': ['islam', 'muslim', 'islami', 'keislaman'],
            'doa': ['doa', 'berdoa', 'memohon', 'permohonan', 'dzikir'],
            'surga': ['surga', 'jannah', 'paradise', 'akhirat'],
            'neraka': ['neraka', 'jahannam', 'azab', 'siksa']
        }
    
    def _load_keywords_map(self) -> Dict[str, List[str]]:
        """Load keywords map from JSON file."""
        try:
            with open(self.keywords_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both old and new format
                if isinstance(data, dict) and 'keywords' in data:
                    return data["keywords"]
                else:
                    return data
        except FileNotFoundError:
            logger.warning(f"Keywords map not found at {self.keywords_map_path}. Using empty map.")
            return {}
        except Exception as e:
            logger.error(f"Error loading keywords map: {e}")
            return {}
    
    def _build_reverse_map(self) -> Dict[str, str]:
        """Build reverse lookup: variant -> canonical."""
        reverse_map = {}
        for canonical, variants in self.keywords_map.items():
            for variant in variants:
                reverse_map[variant.lower()] = canonical
        return reverse_map
    
    def _build_sorted_phrases(self) -> List[str]:
        """Build sorted phrases prioritizing longer phrases."""
        phrases = list(self.reverse_map.keys())
        return sorted(phrases, key=lambda x: (-len(x), x))
    
    def extract_keywords_from_query(self, query: str) -> Tuple[Set[str], Set[str]]:
        """
        Extract keywords from query using enhanced preprocessing and mapping.
        
        Args:
            query: Raw query string
            
        Returns:
            Tuple of (matched_phrases, canonical_keywords)
        """
        # Preprocess query
        processed_query = preprocess_query(query, remove_stopwords=False)
        key_terms = extract_key_terms(query)
        
        matched_phrases = set()
        canonical_keywords = set()
        
        # Search for keyword matches in processed query
        for phrase in self.sorted_phrases:
            # Use word boundary matching for better accuracy
            pattern = rf'\\b{re.escape(phrase)}\\b'
            if re.search(pattern, processed_query):
                matched_phrases.add(phrase)
                canonical = self.reverse_map[phrase]
                canonical_keywords.add(canonical)
        
        # Also check key terms directly
        for term in key_terms:
            term_lower = term.lower()
            if term_lower in self.reverse_map:
                matched_phrases.add(term_lower)
                canonical = self.reverse_map[term_lower]
                canonical_keywords.add(canonical)
        
        return matched_phrases, canonical_keywords
    
    def apply_semantic_expansion(self, query: str, matched_phrases: Set[str]) -> Set[str]:
        """Apply semantic rule-based expansion."""
        expanded_terms = set(matched_phrases)
        query_lower = query.lower()
        
        # Apply semantic expansion rules
        for trigger_word, expansions in self.semantic_expansion_rules.items():
            if re.search(rf'\\b{re.escape(trigger_word)}\\b', query_lower):
                expanded_terms.update(expansions)
                logger.info(f"Applied semantic expansion for '{trigger_word}': {expansions[:5]}...")
        
        return expanded_terms
    
    def apply_synonym_expansion(self, matched_phrases: Set[str]) -> Set[str]:
        """Apply synonym expansion for matched phrases."""
        expanded_terms = set(matched_phrases)
        
        for phrase in matched_phrases:
            # Check if phrase has synonym mappings
            if phrase in self.synonym_mappings:
                synonyms = self.synonym_mappings[phrase]
                expanded_terms.update(synonyms)
                logger.debug(f"Added synonyms for '{phrase}': {synonyms}")
            
            # Check canonical terms for synonyms
            if phrase in self.reverse_map:
                canonical = self.reverse_map[phrase]
                if canonical in self.synonym_mappings:
                    synonyms = self.synonym_mappings[canonical]
                    expanded_terms.update(synonyms)
                    logger.debug(f"Added synonyms for canonical '{canonical}': {synonyms}")
        
        return expanded_terms
    
    def retain_phrase_components(self, query: str, matched_phrases: Set[str]) -> Set[str]:
        """
        Retain both phrases and their components for better matching.
        
        Example: "shalat jum'at" → retain both "shalat jum'at", "shalat", "jumat"
        """
        enhanced_terms = set(matched_phrases)
        
        for phrase in matched_phrases:
            # For multi-word phrases, add individual components
            if ' ' in phrase:
                components = phrase.split()
                for component in components:
                    if len(component) >= 3:  # Only meaningful components
                        enhanced_terms.add(component)
        
        # Also extract key terms from original query
        key_terms = extract_key_terms(query)
        enhanced_terms.update(key_terms)
        
        return enhanced_terms
    
    def optimize_query(self, query: str, return_keywords: bool = False) -> Union[str, Tuple[str, List[str]]]:
        """
        Enhanced query optimization with comprehensive keyword processing.
        
        Args:
            query: Raw query from user
            return_keywords: Whether to return extracted keywords
            
        Returns:
            Optimized query string, or tuple of (optimized_query, keywords) if return_keywords=True
        """
        if not query or not query.strip():
            if return_keywords:
                return "", []
            return ""
        
        original_query = query.strip()
        
        # Step 1: Extract keywords using enhanced preprocessing
        matched_phrases, canonical_keywords = self.extract_keywords_from_query(query)
        
        # Step 2: Apply semantic expansion
        expanded_phrases = self.apply_semantic_expansion(query, matched_phrases)
        
        # Step 3: Apply synonym expansion  
        synonym_expanded = self.apply_synonym_expansion(expanded_phrases)
        
        # Step 4: Retain phrase components
        final_keywords = self.retain_phrase_components(query, synonym_expanded)
        
        # Step 5: Add canonical keywords to final set
        final_keywords.update(canonical_keywords)
        
        # Remove empty and very short terms
        final_keywords = {kw for kw in final_keywords if kw and len(kw.strip()) >= 3}
        
        # Log results
        if final_keywords:
            logger.info(f"Query optimization results for '{original_query}':")
            logger.info(f"  Matched phrases: {len(matched_phrases)}")
            logger.info(f"  Canonical keywords: {len(canonical_keywords)}")
            logger.info(f"  Final keywords: {len(final_keywords)}")
            logger.debug(f"  Keywords: {sorted(list(final_keywords))[:10]}")
        else:
            logger.info(f"No keywords found for query: '{original_query}'")
        
        # Format optimized query
        base = f"passage: {original_query}"
        if final_keywords:
            # Sort keywords for consistent output
            sorted_keywords = sorted(list(final_keywords))
            enriched = base + ". Kata kunci penting: " + " ".join(sorted_keywords)
        else:
            enriched = base
        
        # Return based on parameter
        if return_keywords:
            return enriched, sorted(list(final_keywords))
        return enriched


# Global instance for backward compatibility
keywords_map_path = "data/processed/keywords_map_grouped.json"
if not os.path.exists(keywords_map_path):
    # Fallback to old path
    keywords_map_path = "data/keywords_map.json"

try:
    query_optimizer = EnhancedQueryOptimizer(keywords_map_path)
except Exception as e:
    logger.warning(f"Failed to initialize enhanced optimizer: {e}")
    # Fallback to basic implementation
    query_optimizer = None

# Backward compatibility functions
def optimize_query(query: str, return_keywords: bool = False):
    """
    Backward compatible optimize_query function.
    
    Args:
        query: Query string to optimize
        return_keywords: Whether to return keywords list
        
    Returns:
        Optimized query string or tuple of (query, keywords)
    """
    if query_optimizer:
        return query_optimizer.optimize_query(query, return_keywords)
    else:
        # Fallback implementation
        from preprocessing.normalize import normalize_text
        
        normalized = normalize_text(query.lower())
        base = f"passage: {query.strip()}"
        
        if return_keywords:
            # Extract basic keywords
            words = normalized.split()
            keywords = [w for w in words if len(w) >= 3]
            return base, keywords
        return base


# Test function
if __name__ == "__main__":
    print("=== Enhanced Query Optimizer Test ===")
    
    test_queries = [
        "Apa hukum shalat jum'at bagi wanita?",
        "Bagaimana cara berwudhu yang benar?",
        "Apakah puasa wajib bagi semua muslim?",
        "Hukum jual beli dalam Islam",
        "Tata cara nikah menurut syariat"
    ]
    
    if query_optimizer:
        for query in test_queries:
            optimized, keywords = query_optimizer.optimize_query(query, return_keywords=True)
            
            print(f"\\nQuery: {query}")
            print(f"Keywords: {keywords}")
            print(f"Optimized: {optimized}")
    else:
        print("Query optimizer not available - missing dependencies")
