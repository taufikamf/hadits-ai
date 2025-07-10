"""
Data loader for hadits-ai.
Handles loading and preprocessing of hadits datasets.
"""
import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm

from config import settings

logger = logging.getLogger(__name__)

class HaditsDataLoader:
    """
    Data loader for hadits datasets.
    Handles loading and preprocessing of multiple hadits CSV files.
    """
    
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__))
        
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = str(text).strip()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        return text
    
    def _process_single_document(self, row: pd.Series, source_file: str) -> Optional[Dict[str, Any]]:
        """Process a single document row"""
        try:
            # Extract base filename without extension as kitab name
            kitab = os.path.splitext(os.path.basename(source_file))[0]
            
            # Clean text fields
            arab_text = self._preprocess_text(row.get('arab', ''))
            terjemah_text = self._preprocess_text(row.get('terjemah', ''))
            
            if not arab_text or not terjemah_text:
                logger.warning(f"Skipping document from {kitab} due to missing content")
                return None
            
            # Create document with clean text
            document = {
                'id': str(row.get('id', '')),
                'kitab': kitab,
                'arab_asli': arab_text,
                'terjemah_bersih': terjemah_text,
                'metadata': {
                    'source': source_file,
                    'row_index': row.name
                }
            }
            
            # Create optimized text for embedding
            # Combine Arabic and Indonesian for better semantic search
            document['content_for_embedding'] = f"{arab_text}\n{terjemah_text}"
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return None
    
    def load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and process a single dataset file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of processed documents
        """
        try:
            logger.info(f"Loading dataset from {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Process documents
            documents = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {os.path.basename(file_path)}"):
                doc = self._process_single_document(row, file_path)
                if doc:
                    documents.append(doc)
            
            logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load dataset {file_path}: {e}")
            raise
    
    def load_all_datasets(self) -> List[Dict[str, Any]]:
        """
        Load and process all CSV files in the data directory.
        
        Returns:
            List of processed documents from all datasets
        """
        all_documents = []
        
        try:
            # Get all CSV files in data directory
            csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            
            if not csv_files:
                logger.warning("No CSV files found in data directory")
                return []
            
            logger.info(f"Found {len(csv_files)} dataset files")
            
            # Process each CSV file
            for csv_file in csv_files:
                file_path = os.path.join(self.data_dir, csv_file)
                try:
                    documents = self.load_dataset(file_path)
                    all_documents.extend(documents)
                    logger.info(f"Added {len(documents)} documents from {csv_file}")
                except Exception as e:
                    logger.error(f"Failed to load {csv_file}: {e}")
                    continue
            
            logger.info(f"Successfully loaded total of {len(all_documents)} documents from all datasets")
            return all_documents
            
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise
    
    def get_dataset_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the loaded datasets.
        
        Args:
            documents: List of processed documents
            
        Returns:
            Dictionary with dataset statistics
        """
        try:
            # Count documents per kitab
            kitab_distribution = {}
            for doc in documents:
                kitab = doc.get('kitab', 'unknown')
                kitab_distribution[kitab] = kitab_distribution.get(kitab, 0) + 1
            
            # Get unique sources
            sources = set()
            for doc in documents:
                source = doc.get('metadata', {}).get('source', '')
                if source:
                    sources.add(source)
            
            stats = {
                'total_documents': len(documents),
                'kitab_distribution': kitab_distribution,
                'unique_sources': list(sources)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get dataset stats: {e}")
            return {
                'total_documents': 0,
                'kitab_distribution': {},
                'unique_sources': []
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
    documents = loader.load_all_datasets()
    
    # Get statistics
    stats = loader.get_dataset_stats(documents)
    print(f"Dataset stats: {stats}")
    
    # Show first document
    if documents:
        print(f"\nFirst document:")
        first_doc = documents[0]
        print(f"ID: {first_doc['id']}")
        print(f"Kitab: {first_doc['kitab']}")
        print(f"Arab (clean): {first_doc['arab_asli'][:100]}...")
        print(f"Terjemah (clean): {first_doc['terjemah_bersih'][:100]}...") 