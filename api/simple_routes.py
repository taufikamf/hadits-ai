"""
Simple FastAPI routes for hadits-ai API using simple vector store.
"""
import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from config import settings
from retriever.simple_vector_store import get_simple_vector_store
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
    retrieved_documents: List[Dict[str, Any]]
    total_results: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    timestamp: float


@router.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "hadits-ai",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation system for Islamic hadits",
        "endpoints": {
            "ask": "/ask?q=your_question",
            "health": "/health"
        }
    }


@router.get("/ask", response_model=RAGResponse)
async def ask_question(
    q: str = Query(..., description="Question about hadits in Indonesian"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of hadits to retrieve")
):
    """
    Ask a question about hadits using RAG pipeline.
    """
    start_time = time.time()
    
    try:
        # Validate query
        if not q or not q.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get services
        vector_store = get_simple_vector_store()
        llm_service = get_llm_service()
        
        # Step 1: Semantic retrieval
        logger.info(f"Processing query: {q[:100]}...")
        retrieved_docs = vector_store.search(
            query=q.strip(),
            top_k=top_k
        )
        
        if not retrieved_docs:
            # No relevant hadits found
            return RAGResponse(
                query=q,
                answer="Maaf, saya tidak menemukan hadits yang relevan dengan pertanyaan Anda. Silakan coba dengan kata kunci yang berbeda.",
                retrieved_documents=[],
                total_results=0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Step 2: Generate response using LLM
        answer = llm_service.generate_response(
            query=q,
            retrieved_docs=retrieved_docs
        )
        
        # Step 3: Format response
        retrieved_documents = []
        for doc in retrieved_docs:
            retrieved_documents.append({
                'id': str(doc['id']),
                'kitab': doc['kitab'],
                'arab_asli': doc['arab_asli'],
                'arab_bersih': doc['arab_bersih'],
                'terjemah': doc['terjemah'],
                'score': doc['score']
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        return RAGResponse(
            query=q,
            answer=answer,
            retrieved_documents=retrieved_documents,
            total_results=len(retrieved_documents),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query '{q}': {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector store
        vector_store = get_simple_vector_store()
        collection_info = vector_store.get_collection_info()
        
        # Check LLM service
        llm_service = get_llm_service()
        
        return HealthResponse(
            status="healthy",
            message=f"Service is running. Indexed documents: {collection_info.get('document_count', 0)}",
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            message=f"Service error: {str(e)}",
            timestamp=time.time()
        )