"""
Hadith AI Service Package - Fixed V1
===================================

Service layer components for Hadith AI chatbot system.

This package provides:
- HadithAIService: Main service class with enhanced retrieval
- API Server: Flask-based REST API
- Configuration: Service configuration management
- Response handling: Standardized response formats

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

from .hadith_ai_service import (
    HadithAIService,
    ServiceConfig,
    ChatResponse,
    ChatSession,
    create_hadith_ai_service,
    quick_query
)

__version__ = "1.0.0"
__all__ = [
    "HadithAIService",
    "ServiceConfig", 
    "ChatResponse",
    "ChatSession",
    "create_hadith_ai_service",
    "quick_query"
]
