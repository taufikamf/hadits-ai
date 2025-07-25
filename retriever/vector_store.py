"""
Vector store service for hadits-ai using ChromaDB.
"""
import logging
from typing import List, Dict, Any, Optional
import uuid
import numpy as np
from tqdm import tqdm
import chromadb
from chromadb.config import Settings as ChromaSettings
import re

from config import settings
from embedding.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

# ChromaDB batch size limit (slightly below actual limit for safety)
MAX_BATCH_SIZE = 5000


class HaditsVectorStore:
    """
    Vector store for hadits documents using ChromaDB.
    """
    
    def __init__(self, collection_name: str = "hadits_collection"):
        self.collection_name = collection_name
        self.embedding_service = get_embedding_service()
        
        # Initialize ChromaDB client
        chroma_settings = ChromaSettings(
            persist_directory=settings.chroma_persist_directory,
            anonymized_telemetry=False
        )
        
        self.client = chromadb.Client(chroma_settings)
        
        # Initialize collection
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize or get the collection"""
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We'll handle embeddings manually
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def _generate_unique_id(self, doc: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a document by combining kitab and original ID.
        This ensures uniqueness across different datasets.
        """
        kitab = doc.get('kitab', '').lower().replace(' ', '_')
        original_id = str(doc.get('id', '')).strip()
        
        if not kitab or not original_id:
            # Fallback to UUID if either kitab or ID is missing
            return str(uuid.uuid4())
        
        # Create unique ID by combining kitab and original ID
        unique_id = f"{kitab}_{original_id}"
        return unique_id
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for better matching:
        - Convert to lowercase
        - Remove excessive whitespace
        - Remove punctuation
        - Handle common Indonesian text variations
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove punctuation except in common abbreviations
        text = re.sub(r'[^\w\s.]', ' ', text)
        
        # Handle common Indonesian variations
        replacements = {
            'sholat': 'salat',
            'shalat': 'salat',
            'solat': 'salat',
            'wudhu': 'wudu',
            'wudlu': 'wudu',
            'hadist': 'hadis',
            'hadits': 'hadis',
        }
        
        for old, new in replacements.items():
            text = re.sub(r'\b' + old + r'\b', new, text)
        
        return text.strip()
    
    def _prepare_text_for_embedding(self, doc: Dict[str, Any]) -> str:
        """
        Prepare document text for embedding with improved context.
        """
        # Get text fields
        arab_text = doc.get('arab_asli', '').strip()
        terjemah_text = doc.get('terjemah_bersih', '').strip()
        kitab = doc.get('kitab', '').strip()
        
        # Normalize Indonesian translation
        normalized_terjemah = self._normalize_text(terjemah_text)
        
        # Create context-rich text for embedding
        # Format: Kitab information, followed by translation, then Arabic
        # This order prioritizes matching in Indonesian while maintaining Arabic context
        embedding_text = f"Kitab: {kitab}\n"
        embedding_text += f"Terjemahan: {normalized_terjemah}\n"
        embedding_text += f"Arab: {arab_text}"
        
        return embedding_text
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to vector store with batching.
        
        Args:
            documents: List of processed hadits documents
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        all_ids = []
        total_docs = len(documents)
        
        try:
            # Process documents in batches
            for start_idx in range(0, total_docs, MAX_BATCH_SIZE):
                end_idx = min(start_idx + MAX_BATCH_SIZE, total_docs)
                batch = documents[start_idx:end_idx]
                
                logger.info(f"Processing batch {start_idx//MAX_BATCH_SIZE + 1}, documents {start_idx} to {end_idx}")
                
                # Prepare data for ChromaDB
                ids = []
                embeddings = []
                metadatas = []
                documents_content = []
                
                # Extract content for embedding
                contents_to_embed = []
                for doc in batch:
                    # Prepare optimized text for embedding
                    content = self._prepare_text_for_embedding(doc)
                    if not content:
                        logger.warning(f"Document {doc.get('id', 'unknown')} has no content for embedding")
                        continue
                    contents_to_embed.append(content)
                
                if not contents_to_embed:
                    logger.warning("No valid documents to embed in current batch")
                    continue
                
                # Generate embeddings for batch
                try:
                    embeddings_list = self.embedding_service.embed_documents(contents_to_embed)
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch: {e}")
                    raise
                
                # Prepare data for storage
                for i, doc in enumerate(batch):
                    if i >= len(embeddings_list):
                        break
                        
                    # Generate unique ID by combining kitab and original ID
                    doc_id = self._generate_unique_id(doc)
                    ids.append(doc_id)
                    
                    # Get embedding
                    embeddings.append(embeddings_list[i])
                    
                    # Prepare metadata with normalized text
                    metadata = {
                        'original_id': str(doc.get('id', '')),
                        'kitab': doc.get('kitab', ''),
                        'source': doc.get('metadata', {}).get('source', 'hadits_dataset'),
                        'normalized_terjemah': self._normalize_text(doc.get('terjemah_bersih', ''))
                    }
                    metadatas.append(metadata)
                    
                    # Full document content for ChromaDB document field
                    full_content = f"ID: {doc.get('id', '')}\nKitab: {doc.get('kitab', '')}\n"
                    full_content += f"Arab: {doc.get('arab_asli', '')}\n"
                    full_content += f"Terjemahan: {doc.get('terjemah_bersih', '')}"
                    documents_content.append(full_content)
                
                # Add batch to ChromaDB
                try:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        documents=documents_content
                    )
                    all_ids.extend(ids)
                    logger.info(f"Successfully added batch of {len(ids)} documents to vector store")
                    
                except Exception as e:
                    logger.error(f"Failed to add batch to ChromaDB: {e}")
                    raise
            
            logger.info(f"Successfully added all {len(all_ids)} documents to vector store")
            return all_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query in Indonesian
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of search results with scores and metadata
        """
        if not query.strip():
            return []
        
        try:
            # Normalize query text
            normalized_query = self._normalize_text(query)
            logger.info(f"Normalized query: {normalized_query}")
            
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(normalized_query)
            
            # Search in ChromaDB with higher initial k to allow for post-filtering
            initial_k = min(top_k * 2, 20)  # Get more results initially
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=initial_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['ids'] or not results['ids'][0]:
                return []
            
            # Process results
            search_results = []
            for i in range(len(results['ids'][0])):
                # ChromaDB returns distances, convert to similarity scores
                distance = results['distances'][0][i]
                similarity_score = 1 / (1 + distance)  # Convert distance to similarity
                
                if similarity_score < score_threshold:
                    continue
                
                # Parse document content to extract structured data
                document_content = results['documents'][0][i]
                metadata = results['metadatas'][0][i]
                
                # Extract structured information from document content
                lines = document_content.split('\n')
                parsed_doc = {'id': '', 'kitab': '', 'arab': '', 'terjemah': ''}
                
                for line in lines:
                    if line.startswith('ID: '):
                        parsed_doc['id'] = line[4:].strip()
                    elif line.startswith('Kitab: '):
                        parsed_doc['kitab'] = line[7:].strip()
                    elif line.startswith('Arab: '):
                        parsed_doc['arab'] = line[6:].strip()
                    elif line.startswith('Terjemahan: '):
                        parsed_doc['terjemah'] = line[12:].strip()
                
                search_result = {
                    'id': metadata.get('original_id', ''),  # Use original ID for consistency
                    'score': similarity_score,
                    'metadata': metadata,
                    'content': parsed_doc,
                    'kitab': parsed_doc['kitab'],
                    'arab': parsed_doc['arab'],
                    'terjemah': parsed_doc['terjemah']
                }
                
                search_results.append(search_result)
            
            # Sort by score (highest first) and limit to top_k
            search_results.sort(key=lambda x: x['score'], reverse=True)
            search_results = search_results[:top_k]
            
            logger.info(f"Found {len(search_results)} results for query: {query[:50]}...")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search in vector store: {e}")
            raise
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            # Reinitialize collection after deletion
            self._initialize_collection()
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'embedding_dimension': self.embedding_service.get_dimension()
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            # Try to reinitialize collection
            try:
                self._initialize_collection()
                count = self.collection.count()
                return {
                    'name': self.collection_name,
                    'count': count,
                    'embedding_dimension': self.embedding_service.get_dimension()
                }
            except Exception as e2:
                logger.error(f"Failed to reinitialize collection: {e2}")
                return {'name': self.collection_name, 'count': 0, 'error': str(e2)}


# Global vector store instance
_vector_store: Optional[HaditsVectorStore] = None


def get_vector_store() -> HaditsVectorStore:
    """Get global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = HaditsVectorStore()
    return _vector_store


def reset_vector_store():
    """Reset the global vector store instance"""
    global _vector_store
    _vector_store = None


# Example usage
if __name__ == "__main__":
    # Test the vector store
    vector_store = get_vector_store()
    
    # Test documents
    test_docs = [
        {
            'id': '1',
            'kitab': 'shahih_bukhari',
            'arab_asli': 'حَدَّثَنَا الْحُمَيْدِيُّ',
            'terjemah_bersih': 'Sesungguhnya setiap perbuatan tergantung niatnya',
            'content_for_embedding': 'Sesungguhnya setiap perbuatan tergantung niatnya',
            'metadata': {'source': 'test'}
        }
    ]
    
    # Add documents
    ids = vector_store.add_documents(test_docs)
    print(f"Added documents with IDs: {ids}")
    
    # Search
    results = vector_store.search("niat dalam Islam", top_k=3)
    print(f"Search results: {len(results)}")
    for result in results:
        print(f"Score: {result['score']:.3f}, Content: {result['terjemah'][:100]}...")
    
    # Collection info
    info = vector_store.get_collection_info()
    print(f"Collection info: {info}") 