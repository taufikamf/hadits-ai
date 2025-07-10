"""
Text processing utilities for hadits content.
Based on Dify's CleanProcessor but specialized for Arabic-Indonesian hadits.
"""
import re
import json
from typing import Dict, Any
import unicodedata


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
    ]
    
    @classmethod
    def remove_diacritics(cls, text: str) -> str:
        """Remove Arabic diacritics (harakat) from text"""
        for diacritic in cls.ARABIC_DIACRITICS:
            text = text.replace(diacritic, '')
        return text
    
    @classmethod
    def normalize_arabic(cls, text: str) -> str:
        """Normalize Arabic text by removing diacritics and extra spaces"""
        # Remove diacritics
        text = cls.remove_diacritics(text)
        
        # Normalize Unicode (NFD -> NFC)
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class IndonesianTextProcessor:
    """Processor for Indonesian text cleaning"""
    
    @classmethod
    def clean_translation(cls, text: str) -> str:
        """Clean Indonesian translation text"""
        # Remove HTML entities and tags
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove quotes at the beginning and end
        text = text.strip('"\'""''')
        
        # Remove extra newlines and whitespaces
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special symbols but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\?\!\;\:\-\(\)\[\]\/]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


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
        # Extract data
        hadits_id = str(row.get('id', ''))
        kitab = str(row.get('kitab', '')).strip()
        arab_asli = str(row.get('arab', '')).strip()
        terjemah_asli = str(row.get('terjemah', '')).strip()
        
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
            'content_for_embedding': terjemah_bersih,  # Use clean Indonesian for embedding
            'metadata': {
                'id': hadits_id,
                'kitab': kitab,
                'source': 'hadits_dataset'
            }
        }
        
        return processed_doc
    
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
        'terjemah': 'Telah menceritakan kepada kami [Al Humaidi Abdullah bin Az Zubair]...'
    }
    
    result = processor.process_hadits_row(test_row)
    print("Processed result:")
    print(json.dumps(result, indent=2, ensure_ascii=False)) 