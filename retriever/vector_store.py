"""
Vector store service for hadits-ai using ChromaDB.
Inspired by Dify's Vector factory pattern.
"""
import logging
import uuid
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from embedding.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class HaditsVectorStore:
    """
    Vector store for hadits documents using ChromaDB.
    Inspired by Dify's ChromaVector implementation.
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
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=None  # We'll handle embeddings manually
            )
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=None
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents to vector store.
        
        Args:
            documents: List of processed hadits documents
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        documents_content = []
        
        # Extract content for embedding
        contents_to_embed = []
        for doc in documents:
            content = doc.get('content_for_embedding', '')
            if not content:
                logger.warning(f"Document {doc.get('id', 'unknown')} has no content for embedding")
                continue
            contents_to_embed.append(content)
        
        if not contents_to_embed:
            logger.warning("No valid documents to embed")
            return []
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(contents_to_embed)} documents")
        try:
            embeddings_list = self.embedding_service.embed_documents(contents_to_embed)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
        
        # Prepare data for storage
        for i, doc in enumerate(documents):
            if i >= len(embeddings_list):
                break
                
            # Generate unique ID
            doc_id = doc.get('id', str(uuid.uuid4()))
            ids.append(str(doc_id))
            
            # Get embedding
            embeddings.append(embeddings_list[i])
            
            # Prepare metadata (exclude large content from metadata)
            metadata = {
                'id': str(doc_id),
                'kitab': doc.get('kitab', ''),
                'source': doc.get('metadata', {}).get('source', 'hadits_dataset')
            }
            metadatas.append(metadata)
            
            # Full document content for ChromaDB document field
            # This includes both Arabic and Indonesian for display
            full_content = f"ID: {doc_id}\nKitab: {doc.get('kitab', '')}\n"
            full_content += f"Arab: {doc.get('arab_asli', '')}\n"
            full_content += f"Terjemahan: {doc.get('terjemah_bersih', '')}"
            documents_content.append(full_content)
        
        # Add to ChromaDB
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents_content
            )
            logger.info(f"Successfully added {len(ids)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
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
        
        # Generate query embedding
        try:
            query_embedding = self.embedding_service.embed_query(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
        
        # Search in ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
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
                    'id': results['ids'][0][i],
                    'score': similarity_score,
                    'metadata': metadata,
                    'content': parsed_doc,
                    'kitab': parsed_doc['kitab'],
                    'arab': parsed_doc['arab'],
                    'terjemah': parsed_doc['terjemah']
                }
                
                search_results.append(search_result)
            
            # Sort by score (highest first)
            search_results.sort(key=lambda x: x['score'], reverse=True)
            
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
            return {'name': self.collection_name, 'count': 0, 'error': str(e)}


# Global vector store instance
_vector_store: Optional[HaditsVectorStore] = None


def get_vector_store() -> HaditsVectorStore:
    """Get global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = HaditsVectorStore()
    return _vector_store


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