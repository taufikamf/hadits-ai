"""
FastAPI routes for hadits-ai API.
"""
import logging
import os
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
import shutil

from config import settings
from retriever.vector_store import get_vector_store, reset_vector_store
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
    unique_sources: List[str]


class IndexStatus(BaseModel):
    """Index status model"""
    indexed: bool
    total_documents: int
    message: str


class DatasetUploadResponse(BaseModel):
    """Dataset upload response model"""
    filename: str
    status: str
    documents_processed: int
    message: str


@router.get("/", response_model=Dict[str, Any])
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
            "health": "/health",
            "upload": "/dataset/upload",
            "datasets": "/dataset/list"
        }
    }


@router.get("/ask", response_model=RAGResponse)
async def ask_question(
    q: str = Query(..., description="Question about hadits in Indonesian"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of hadits to retrieve"),
    score_threshold: float = Query(default=0.3, ge=0.0, le=1.0, description="Minimum relevance score")
):
    """Ask a question about hadits using RAG pipeline"""
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
        # Try to reinitialize vector store
        try:
            reset_vector_store()
            vector_store = get_vector_store()
            # Try the query again
            return await ask_question(q=q, top_k=top_k, score_threshold=score_threshold)
        except Exception as e2:
            logger.error(f"Failed to recover from error: {e2}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/index", response_model=IndexStatus)
async def rebuild_index():
    """
    Rebuild the vector index from all datasets.
    This will reload and re-process all hadits data.
    """
    try:
        # Get services
        data_loader = get_data_loader()
        
        # Load all documents from all datasets
        logger.info("Loading all hadits datasets...")
        documents = data_loader.load_all_datasets()
        
        if not documents:
            raise HTTPException(status_code=400, detail="No documents found in any dataset")
        
        # Clear existing collection and reset vector store
        logger.info("Clearing existing vector index...")
        reset_vector_store()
        vector_store = get_vector_store()
        
        # Add documents to vector store
        logger.info(f"Indexing {len(documents)} documents...")
        doc_ids = vector_store.add_documents(documents)
        
        return IndexStatus(
            indexed=True,
            total_documents=len(doc_ids),
            message=f"Successfully indexed {len(doc_ids)} hadits documents from all datasets"
        )
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")


@router.get("/info", response_model=DatasetInfo)
async def get_dataset_info():
    """Get information about all loaded datasets and index"""
    try:
        vector_store = get_vector_store()
        data_loader = get_data_loader()
        
        # Get collection info
        collection_info = vector_store.get_collection_info()
        
        # Load all datasets to get statistics
        documents = data_loader.load_all_datasets()
        dataset_stats = data_loader.get_dataset_stats(documents)
        
        return DatasetInfo(
            total_documents=collection_info.get('count', 0),
            kitab_distribution=dataset_stats.get('kitab_distribution', {}),
            embedding_dimension=collection_info.get('embedding_dimension', 0),
            collection_name=collection_info.get('name', 'hadits_collection'),
            unique_sources=dataset_stats.get('unique_sources', [])
        )
        
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        # Try to reinitialize vector store
        try:
            reset_vector_store()
            return await get_dataset_info()
        except Exception as e2:
            logger.error(f"Failed to recover from error: {e2}")
            raise HTTPException(status_code=500, detail=f"Failed to get dataset info: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test basic services
        vector_store = get_vector_store()
        collection_info = vector_store.get_collection_info()
        
        # Get dataset info
        data_loader = get_data_loader()
        documents = data_loader.load_all_datasets()
        dataset_stats = data_loader.get_dataset_stats(documents)
        
        return {
            "status": "healthy",
            "services": {
                "vector_store": "ok",
                "llm_service": "ok",
                "data_loader": "ok"
            },
            "index_count": collection_info.get('count', 0),
            "datasets": {
                "total_documents": len(documents),
                "unique_sources": dataset_stats.get('unique_sources', [])
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        # Try to reinitialize vector store
        try:
            reset_vector_store()
            vector_store = get_vector_store()
            collection_info = vector_store.get_collection_info()
            return {
                "status": "recovered",
                "services": {
                    "vector_store": "reinitialized",
                    "llm_service": "ok",
                    "data_loader": "ok"
                },
                "index_count": collection_info.get('count', 0)
            }
        except Exception as e2:
            logger.error(f"Failed to recover from error: {e2}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


@router.post("/dataset/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a new hadits dataset CSV file.
    The file will be saved to the data directory and processed.
    """
    try:
        # Validate file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Save file to data directory
        data_loader = get_data_loader()
        file_path = os.path.join(data_loader.data_dir, file.filename)
        
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        finally:
            file.file.close()
        
        # Process the new dataset
        try:
            documents = data_loader.load_dataset(file_path)
            
            # Update vector store with new documents
            vector_store = get_vector_store()
            doc_ids = vector_store.add_documents(documents)
            
            return DatasetUploadResponse(
                filename=file.filename,
                status="success",
                documents_processed=len(doc_ids),
                message=f"Successfully processed and indexed {len(doc_ids)} documents from {file.filename}"
            )
            
        except Exception as e:
            # If processing fails, delete the uploaded file
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Failed to process dataset: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")


@router.get("/dataset/list")
async def list_datasets():
    """List all available datasets in the data directory"""
    try:
        data_loader = get_data_loader()
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(data_loader.data_dir) if f.endswith('.csv')]
        
        # Get stats for each dataset
        datasets = []
        for csv_file in csv_files:
            try:
                file_path = os.path.join(data_loader.data_dir, csv_file)
                documents = data_loader.load_dataset(file_path)
                stats = data_loader.get_dataset_stats(documents)
                
                datasets.append({
                    "filename": csv_file,
                    "total_documents": stats['total_documents'],
                    "kitab": os.path.splitext(csv_file)[0]
                })
                
            except Exception as e:
                logger.error(f"Failed to get stats for {csv_file}: {e}")
                datasets.append({
                    "filename": csv_file,
                    "error": str(e)
                })
        
        return {
            "total_datasets": len(csv_files),
            "datasets": datasets
        }
        
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list datasets: {str(e)}") 