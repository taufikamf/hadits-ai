"""
Data loader service for hadits CSV ingestion.
Inspired by Dify's ExtractProcessor pattern.
"""
import logging
import pandas as pd
from typing import List, Dict, Any, Iterator, Optional
import os

from config import settings
from utils.text_processor import HaditsDocumentProcessor

logger = logging.getLogger(__name__)


class HaditsDataLoader:
    """
    Data loader for hadits CSV files.
    Inspired by Dify's CSV extractor implementation.
    """
    
    def __init__(self):
        self.processor = HaditsDocumentProcessor()
    
    def load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and process hadits from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of processed hadits documents
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            # Read CSV file
            logger.info(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_columns = ['id', 'kitab', 'arab', 'terjemah']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Process each row
            processed_docs = []
            for index, row in df.iterrows():
                try:
                    # Process single hadits row
                    processed_doc = self.processor.process_hadits_row(row.to_dict())
                    processed_docs.append(processed_doc)
                    
                except Exception as e:
                    logger.warning(f"Failed to process row {index}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(processed_docs)} hadits documents")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Failed to load CSV file: {e}")
            raise
    
    def load_default_dataset(self) -> List[Dict[str, Any]]:
        """Load the default hadits dataset"""
        return self.load_csv(settings.dataset_path)
    
    def validate_document(self, doc: Dict[str, Any]) -> bool:
        """
        Validate that a document has required fields.
        
        Args:
            doc: Document to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['id', 'kitab', 'arab_asli', 'terjemah_bersih', 'content_for_embedding']
        
        for field in required_fields:
            if field not in doc or not doc[field]:
                return False
        
        return True
    
    def get_dataset_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Args:
            documents: List of processed documents
            
        Returns:
            Dictionary with dataset statistics
        """
        if not documents:
            return {'total': 0, 'kitab_distribution': {}}
        
        # Count by kitab
        kitab_counts = {}
        total_chars = 0
        total_words = 0
        
        for doc in documents:
            kitab = doc.get('kitab', 'Unknown')
            kitab_counts[kitab] = kitab_counts.get(kitab, 0) + 1
            
            # Count characters and words in terjemah
            terjemah = doc.get('terjemah_bersih', '')
            total_chars += len(terjemah)
            total_words += len(terjemah.split())
        
        return {
            'total': len(documents),
            'kitab_distribution': kitab_counts,
            'avg_chars_per_hadits': total_chars / len(documents) if documents else 0,
            'avg_words_per_hadits': total_words / len(documents) if documents else 0,
            'total_chars': total_chars,
            'total_words': total_words
        }


# Global data loader instance
_data_loader: Optional[HaditsDataLoader] = None


def get_data_loader() -> HaditsDataLoader:
    """Get global data loader instance"""
    global _data_loader
    if _data_loader is None:
        _data_loader = HaditsDataLoader()
    return _data_loader


# Example usage
if __name__ == "__main__":
    # Test the data loader
    loader = get_data_loader()
    
    # Load dataset
    documents = loader.load_default_dataset()
    
    # Get statistics
    stats = loader.get_dataset_stats(documents)
    print(f"Dataset stats: {stats}")
    
    # Show first document
    if documents:
        print(f"\nFirst document:")
        first_doc = documents[0]
        print(f"ID: {first_doc['id']}")
        print(f"Kitab: {first_doc['kitab']}")
        print(f"Arab (clean): {first_doc['arab_bersih'][:100]}...")
        print(f"Terjemah (clean): {first_doc['terjemah_bersih'][:100]}...") 