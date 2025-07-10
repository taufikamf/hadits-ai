"""
Main FastAPI application for hadits-ai.
Entry point for the Retrieval-Augmented Generation system.
"""
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.routes import router
from utils.logger import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Hadits-AI",
    description="Retrieval-Augmented Generation system for Islamic hadits Q&A",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Add root redirect to docs
@app.get("/")
async def redirect_to_docs():
    """Redirect root to API documentation"""
    return {
        "message": "Welcome to Hadits-AI API",
        "docs": "/docs",
        "api": "/api/v1"
    }


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Hadits-AI service...")
    
    try:
        # Initialize services (this will load models)
        from retriever.vector_store import get_vector_store
        from llm.gemini_service import get_llm_service
        from data.data_loader import get_data_loader
        
        # Test services
        vector_store = get_vector_store()
        llm_service = get_llm_service()
        data_loader = get_data_loader()
        
        # Get collection info
        collection_info = vector_store.get_collection_info()
        logger.info(f"Vector store initialized: {collection_info}")
        
        # Check if index is empty and auto-index if needed
        if collection_info.get('count', 0) == 0:
            logger.info("No documents in vector store, auto-indexing default dataset...")
            try:
                documents = data_loader.load_default_dataset()
                if documents:
                    doc_ids = vector_store.add_documents(documents)
                    logger.info(f"Auto-indexed {len(doc_ids)} documents")
                else:
                    logger.warning("No documents found in default dataset")
            except Exception as e:
                logger.warning(f"Auto-indexing failed: {e}")
        
        logger.info("Hadits-AI service started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Hadits-AI service...")


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    ) 