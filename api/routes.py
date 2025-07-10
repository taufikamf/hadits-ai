"""
FastAPI routes for hadits-ai API.
Inspired by Dify's API structure.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from config import settings
from retriever.vector_store import get_vector_store
from llm.gemini_service import get_llm_service
from data.data_loader import get_data_loader

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()


# Response models
class HaditsResult(BaseModel):
    """Single hadits result model"""
    id: str
    kitab: str
    arab: str
    terjemah: str
    score: float


class RAGResponse(BaseModel):
    """RAG response model"""
    query: str
    answer: str
    retrieved_hadits: List[HaditsResult]
    total_results: int
    processing_time_ms: float


class DatasetInfo(BaseModel):
    """Dataset information model"""
    total_documents: int
    kitab_distribution: Dict[str, int]
    embedding_dimension: int
    collection_name: str


class IndexStatus(BaseModel):
    """Index status model"""
    indexed: bool
    total_documents: int
    message: str


@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "hadits-ai",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation system for Islamic hadits",
        "endpoints": {
            "ask": "/ask?q=your_question",
            "index": "/index",
            "info": "/info",
            "health": "/health"
        }
    }


@router.get("/ask", response_model=RAGResponse)
async def ask_question(
    q: str = Query(..., description="Question about hadits in Indonesian"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of hadits to retrieve"),
    score_threshold: float = Query(default=0.3, ge=0.0, le=1.0, description="Minimum relevance score")
):
    """
    Ask a question about hadits using RAG pipeline.
    
    This endpoint performs the full RAG pipeline:
    1. Query → Semantic retrieval → Context → LLM → Response
    """
    import time
    start_time = time.time()
    
    try:
        # Validate query
        if not q or not q.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get services
        vector_store = get_vector_store()
        llm_service = get_llm_service()
        
        # Step 1: Semantic retrieval
        logger.info(f"Processing query: {q[:100]}...")
        retrieved_docs = vector_store.search(
            query=q.strip(),
            top_k=top_k,
            score_threshold=score_threshold
        )
        
        if not retrieved_docs:
            # No relevant hadits found
            return RAGResponse(
                query=q,
                answer="Maaf, saya tidak menemukan hadits yang relevan dengan pertanyaan Anda. Silakan coba dengan kata kunci yang berbeda.",
                retrieved_hadits=[],
                total_results=0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Step 2: Generate response using LLM
        answer = llm_service.generate_response(
            query=q,
            retrieved_docs=retrieved_docs
        )
        
        # Step 3: Format response
        hadits_results = []
        for doc in retrieved_docs:
            hadits_result = HaditsResult(
                id=str(doc['id']),
                kitab=doc['kitab'],
                arab=doc['arab'],
                terjemah=doc['terjemah'],
                score=doc['score']
            )
            hadits_results.append(hadits_result)
        
        processing_time = (time.time() - start_time) * 1000
        
        return RAGResponse(
            query=q,
            answer=answer,
            retrieved_hadits=hadits_results,
            total_results=len(hadits_results),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query '{q}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/index", response_model=IndexStatus)
async def rebuild_index():
    """
    Rebuild the vector index from the dataset.
    This will reload and re-process all hadits data.
    """
    try:
        # Get services
        data_loader = get_data_loader()
        vector_store = get_vector_store()
        
        # Load documents
        logger.info("Loading hadits dataset...")
        documents = data_loader.load_default_dataset()
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents found in dataset")
        
        # Clear existing collection
        logger.info("Clearing existing vector index...")
        try:
            vector_store.delete_collection()
            # Reinitialize vector store
            from retriever.vector_store import HaditsVectorStore
            global _vector_store
            _vector_store = HaditsVectorStore()
            vector_store = _vector_store
        except Exception as e:
            logger.warning(f"Failed to delete existing collection: {e}")
        
        # Add documents to vector store
        logger.info(f"Indexing {len(documents)} documents...")
        doc_ids = vector_store.add_documents(documents)
        
        return IndexStatus(
            indexed=True,
            total_documents=len(doc_ids),
            message=f"Successfully indexed {len(doc_ids)} hadits documents"
        )
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")


@router.get("/info", response_model=DatasetInfo)
async def get_dataset_info():
    """Get information about the current dataset and index"""
    try:
        vector_store = get_vector_store()
        data_loader = get_data_loader()
        
        # Get collection info
        collection_info = vector_store.get_collection_info()
        
        # Load dataset to get statistics
        documents = data_loader.load_default_dataset()
        dataset_stats = data_loader.get_dataset_stats(documents)
        
        return DatasetInfo(
            total_documents=collection_info.get('count', 0),
            kitab_distribution=dataset_stats.get('kitab_distribution', {}),
            embedding_dimension=collection_info.get('embedding_dimension', 0),
            collection_name=collection_info.get('name', 'hadits_collection')
        )
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test basic services
        vector_store = get_vector_store()
        collection_info = vector_store.get_collection_info()
        
        return {
            "status": "healthy",
            "services": {
                "vector_store": "ok",
                "llm_service": "ok",
                "data_loader": "ok"
            },
            "index_count": collection_info.get('count', 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        } 