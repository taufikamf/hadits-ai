"""
Simple vector store service using ChromaDB with simple embedding.
"""
import logging
from typing import List, Dict, Any, Optional

from embedding.simple_embedding import SimpleEmbeddingService

logger = logging.getLogger(__name__)


class SimpleVectorStore:
    """
    Simple vector store using ChromaDB with built-in embedding.
    """
    
    def __init__(self):
        self.embedding_service = SimpleEmbeddingService()
        logger.info("Initialized simple vector store")
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build index from processed documents.
        
        Args:
            documents: List of processed hadits documents
        """
        if not documents:
            logger.warning("No documents to index")
            return
        
        # Clear existing collection
        try:
            self.embedding_service.client.delete_collection("hadits_embeddings")
            logger.info("Cleared existing collection")
        except:
            pass
        
        # Recreate collection
        self.embedding_service.collection = self.embedding_service.client.create_collection(
            name="hadits_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare data for indexing
        texts = []
        ids = []
        metadata_list = []
        
        for doc in documents:
            content = doc.get('content_for_embedding', '')
            if not content:
                logger.warning(f"Document {doc.get('id', 'unknown')} has no content for embedding")
                continue
            
            texts.append(content)
            ids.append(str(doc['id']))
            metadata_list.append({
                'kitab': doc.get('kitab', ''),
                'arab_asli': doc.get('arab_asli', ''),
                'arab_bersih': doc.get('arab_bersih', ''),
                'terjemah': doc.get('terjemah_bersih', '')
            })
        
        # Add documents to collection
        if texts:
            self.embedding_service.collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadata_list
            )
            logger.info(f"Successfully indexed {len(texts)} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if not query.strip():
            return []
        
        try:
            # Search using ChromaDB
            results = self.embedding_service.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # Format results
            search_results = []
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                
                # Convert distance to similarity score
                similarity_score = 1 / (1 + distance)
                
                search_result = {
                    'id': doc_id,
                    'score': similarity_score,
                    'kitab': metadata.get('kitab', ''),
                    'arab_asli': metadata.get('arab_asli', ''),
                    'arab_bersih': metadata.get('arab_bersih', ''),
                    'terjemah': metadata.get('terjemah', ''),
                    'metadata': metadata
                }
                
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            count = self.embedding_service.collection.count()
            return {
                'name': 'hadits_embeddings',
                'document_count': count,
                'embedding_dimension': 384  # ChromaDB default
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


def get_simple_vector_store() -> SimpleVectorStore:
    """Get global simple vector store instance"""
    return SimpleVectorStore()