"""
Text processing utilities for hadits content.
Based on Dify's CleanProcessor but specialized for Arabic-Indonesian hadits.
"""
import re
import json
from typing import Dict, Any
import unicodedata
import logging

logger = logging.getLogger(__name__)


class ArabicTextProcessor:
    """Processor for Arabic text normalization"""
    
    # Arabic diacritics (harakat) to remove
    ARABIC_DIACRITICS = [
        '\u064B',  # Fathatan
        '\u064C',  # Dammatan  
        '\u064D',  # Kasratan
        '\u064E',  # Fatha
        '\u064F',  # Damma
        '\u0650',  # Kasra
        '\u0651',  # Shadda
        '\u0652',  # Sukun
        '\u0653',  # Maddah
        '\u0654',  # Hamza above
        '\u0655',  # Hamza below
        '\u0656',  # Subscript alef
        '\u0657',  # Inverted damma
        '\u0658',  # Mark noon ghunna
        '\u0670',  # Superscript alef
        # Additional diacritics
        '\u0610', '\u0611', '\u0612', '\u0613', '\u0614', '\u0615', '\u0616', '\u0617', '\u0618', '\u0619', '\u061A',
        '\u06D6', '\u06D7', '\u06D8', '\u06D9', '\u06DA', '\u06DB', '\u06DC',
        '\u06DF', '\u06E0', '\u06E1', '\u06E2', '\u06E3', '\u06E4',
        '\u06E7', '\u06E8', '\u06EA', '\u06EB', '\u06EC', '\u06ED'
    ]
    
    # Arabic punctuation to normalize
    ARABIC_PUNCTUATION = {
        '،': ',',  # Arabic comma
        '؛': ';',  # Arabic semicolon
        '؟': '?',  # Arabic question mark
        '«': '"',  # Arabic opening quote
        '»': '"',  # Arabic closing quote
    }
    
    @classmethod
    def remove_diacritics(cls, text: str) -> str:
        """Remove Arabic diacritics (harakat) from text"""
        if not text:
            return ""
            
        for diacritic in cls.ARABIC_DIACRITICS:
            text = text.replace(diacritic, '')
        return text
    
    @classmethod
    def normalize_punctuation(cls, text: str) -> str:
        """Normalize Arabic punctuation to Latin equivalents"""
        if not text:
            return ""
            
        for ar_punct, en_punct in cls.ARABIC_PUNCTUATION.items():
            text = text.replace(ar_punct, en_punct)
        return text
    
    @classmethod
    def normalize_arabic(cls, text: str) -> str:
        """Normalize Arabic text by removing diacritics and standardizing punctuation"""
        if not text:
            return ""
            
        try:
            # Remove diacritics
            text = cls.remove_diacritics(text)
            
            # Normalize punctuation
            text = cls.normalize_punctuation(text)
            
            # Normalize Unicode (NFD -> NFC)
            text = unicodedata.normalize('NFC', text)
            
            # Remove extra whitespaces
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error normalizing Arabic text: {e}")
            return text  # Return original text if normalization fails


class IndonesianTextProcessor:
    """Processor for Indonesian text cleaning"""
    
    # Common patterns to clean
    PATTERNS = {
        'html_tags': r'<[^>]+>',
        'square_brackets': r'\[[^\]]*\]',
        'parentheses': r'\([^\)]*\)',
        'multiple_spaces': r'\s+',
        'special_chars': r'[^\w\s\.\,\?\!\;\:\-\/]'
    }
    
    # Words to standardize
    WORD_STANDARDIZATION = {
        'saw\.': 'shallallahu alaihi wasallam',
        'swt\.': 'subhanahu wa taala',
        'ra\.': 'radhiallahu anhu',
        'hr\.': 'hadits riwayat',
        'saws': 'shallallahu alaihi wasallam',
        'SAW': 'shallallahu alaihi wasallam',
        'SWT': 'subhanahu wa taala',
        'RA': 'radhiallahu anhu'
    }
    
    @classmethod
    def standardize_words(cls, text: str) -> str:
        """Standardize common Islamic terms"""
        if not text:
            return ""
            
        for abbr, full in cls.WORD_STANDARDIZATION.items():
            text = re.sub(rf'\b{abbr}\b', full, text)
        return text
    
    @classmethod
    def clean_translation(cls, text: str) -> str:
        """Clean Indonesian translation text"""
        if not text:
            return ""
            
        try:
            # Remove HTML entities and tags
            text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
            text = re.sub(cls.PATTERNS['html_tags'], '', text)
            
            # Remove square brackets and parentheses content
            text = re.sub(cls.PATTERNS['square_brackets'], '', text)
            text = re.sub(cls.PATTERNS['parentheses'], '', text)
            
            # Remove quotes at the beginning and end
            text = text.strip('"\'""''')
            
            # Standardize Islamic terms
            text = cls.standardize_words(text)
            
            # Remove extra newlines and whitespaces
            text = re.sub(r'\n+', ' ', text)
            text = re.sub(cls.PATTERNS['multiple_spaces'], ' ', text)
            
            # Remove special symbols but keep basic punctuation
            text = re.sub(cls.PATTERNS['special_chars'], ' ', text)
            text = re.sub(cls.PATTERNS['multiple_spaces'], ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning Indonesian text: {e}")
            return text  # Return original text if cleaning fails


class HaditsDocumentProcessor:
    """Main processor for hadits documents"""
    
    def __init__(self):
        self.arabic_processor = ArabicTextProcessor()
        self.indonesian_processor = IndonesianTextProcessor()
    
    def process_hadits_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single hadits row from CSV.
        
        Args:
            row: Dictionary with keys: id, kitab, arab, terjemah
            
        Returns:
            Processed document with original, cleaned text, and metadata
        """
        try:
            # Extract and validate data
            hadits_id = str(row.get('id', '')).strip()
            kitab = str(row.get('kitab', '')).strip()
            arab_asli = str(row.get('arab', '')).strip()
            terjemah_asli = str(row.get('terjemah', '')).strip()
            
            if not all([hadits_id, kitab, arab_asli, terjemah_asli]):
                logger.warning(f"Missing required fields in row: {hadits_id}")
                return None
            
            # Process Arabic text
            arab_bersih = self.arabic_processor.normalize_arabic(arab_asli)
            
            # Process Indonesian translation
            terjemah_bersih = self.indonesian_processor.clean_translation(terjemah_asli)
            
            # Create processed document
            processed_doc = {
                'id': hadits_id,
                'kitab': kitab,
                'arab_asli': arab_asli,
                'arab_bersih': arab_bersih,
                'terjemah_asli': terjemah_asli,
                'terjemah_bersih': terjemah_bersih,
                'content_for_embedding': self.create_document_content({
                    'kitab': kitab,
                    'terjemah_bersih': terjemah_bersih
                }),
                'metadata': {
                    'id': hadits_id,
                    'kitab': kitab,
                    'source': 'hadits_dataset',
                    'has_arabic': bool(arab_bersih),
                    'has_translation': bool(terjemah_bersih)
                }
            }
            
            return processed_doc
            
        except Exception as e:
            logger.error(f"Error processing hadits row {row.get('id', 'unknown')}: {e}")
            return None
    
    def create_document_content(self, processed_doc: Dict[str, Any]) -> str:
        """
        Create final content for vector storage.
        This will be used for embedding and retrieval.
        """
        content_parts = []
        
        # Add kitab information
        if processed_doc.get('kitab'):
            content_parts.append(f"Kitab: {processed_doc['kitab']}")
        
        # Add Indonesian translation (main content for search)
        if processed_doc.get('terjemah_bersih'):
            content_parts.append(f"Terjemahan: {processed_doc['terjemah_bersih']}")
        
        # Join with newlines
        return '\n'.join(content_parts)


# Example usage and testing
if __name__ == "__main__":
    processor = HaditsDocumentProcessor()
    
    # Test data
    test_row = {
        'id': 1,
        'kitab': 'shahih_bukhari',
        'arab': 'حَدَّثَنَا الْحُمَيْدِيُّ عَبْدُ اللَّهِ بْنُ الزُّبَيْرِ',
        'terjemah': 'Telah menceritakan kepada kami [Al Humaidi Abdullah bin Az Zubair] (ra.)'
    }
    
    result = processor.process_hadits_row(test_row)
    print("Processed result:")
    print(json.dumps(result, indent=2, ensure_ascii=False)) 