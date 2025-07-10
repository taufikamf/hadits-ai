"""
Simple FastAPI application for hadits-ai.
Entry point for the Retrieval-Augmented Generation system.
"""
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.simple_routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO if settings.debug else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
        from retriever.simple_vector_store import get_simple_vector_store
        from llm.gemini_service import get_llm_service
        from data.data_loader import get_data_loader
        
        # Test services
        vector_store = get_simple_vector_store()
        llm_service = get_llm_service()
        data_loader = get_data_loader()
        
        # Get collection info
        collection_info = vector_store.get_collection_info()
        logger.info(f"Vector store initialized: {collection_info}")
        
        # Check if index is empty and auto-index if needed
        if collection_info.get('document_count', 0) == 0:
            logger.info("No documents in vector store, auto-indexing default dataset...")
            try:
                documents = data_loader.load_default_dataset()
                if documents:
                    vector_store.build_index(documents)
                    logger.info(f"Auto-indexed {len(documents)} documents")
                else:
                    logger.warning("No documents found in default dataset")
            except Exception as e:
                logger.warning(f"Auto-indexing failed: {e}")
        
        logger.info("Hadits-AI service started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "simple_main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )