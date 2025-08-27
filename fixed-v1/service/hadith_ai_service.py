"""
Hadith AI Service - Fixed V1
==========================

Comprehensive service layer for Hadith AI chatbot integrating all enhanced components.
Provides high-level API for chat interfaces and applications.

Features:
- Enhanced retrieval with adaptive filtering
- Query preprocessing and intent analysis
- Response generation and formatting
- Session management and context tracking
- Comprehensive logging and analytics

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import json
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import enhanced components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.enhanced_retrieval_system import EnhancedRetrievalSystem, RetrievalConfig, RetrievalResult
from preprocessing.query_preprocessor import EnhancedQueryPreprocessor, analyze_query_intent
from generation.enhanced_response_generator import (
    EnhancedResponseGenerator, GenerationConfig, LLMProvider, ResponseMode, GenerationResult
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Chat session data structure."""
    session_id: str
    created_at: datetime
    last_activity: datetime
    query_count: int = 0
    total_results_returned: int = 0
    context_history: List[Dict[str, Any]] = field(default_factory=list)
    user_feedback: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ChatResponse:
    """Structured response from the Hadith AI service."""
    success: bool
    message: str
    results: List[RetrievalResult] = field(default_factory=list)
    query_analysis: Optional[Dict[str, Any]] = None
    response_time_ms: float = 0
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceConfig:
    """Configuration for Hadith AI service."""
    # Retrieval configuration
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # Generation configuration
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    enable_llm_generation: bool = True
    fallback_to_simple: bool = True
    
    # Response formatting
    max_results_display: int = 5
    result_text_preview_length: int = 200
    include_scores: bool = False
    
    # Session management
    enable_sessions: bool = True
    session_timeout_minutes: int = 30
    max_context_history: int = 10
    
    # Analytics and logging
    enable_analytics: bool = True
    log_queries: bool = True
    log_results: bool = True
    analytics_output: str = "logs/hadith_ai_analytics.jsonl"
    
    # Response templates
    greeting_message: str = "Assalamu'alaikum! Saya siap membantu Anda mencari hadits. Silakan ajukan pertanyaan tentang Islam."
    no_results_message: str = "Maaf, saya tidak menemukan hadits yang sesuai dengan pertanyaan Anda. Coba gunakan kata kunci yang lebih spesifik."
    error_message: str = "Maaf, terjadi kesalahan dalam pemrosesan. Silakan coba lagi."
    
    # Quality thresholds
    min_confidence_threshold: float = 0.1
    high_confidence_threshold: float = 0.6


class HadithAIService:
    """
    Main service class for Hadith AI chatbot with enhanced retrieval capabilities.
    """
    
    def __init__(self, config: ServiceConfig = None):
        """
        Initialize the Hadith AI service.
        
        Args:
            config (ServiceConfig): Service configuration
        """
        self.config = config or ServiceConfig()
        
        # Initialize components
        self.retrieval_system = None
        self.query_preprocessor = None
        self.response_generator = None
        
        # Session management
        self.sessions: Dict[str, ChatSession] = {}
        
        # Analytics storage
        self.query_analytics = []
        
        # Initialize service
        self._initialize_service()
        
        logger.info("Hadith AI Service initialized successfully")
    
    def _initialize_service(self):
        """Initialize all service components."""
        logger.info("Initializing Hadith AI service components...")
        
        try:
            # Initialize retrieval system
            self.retrieval_system = EnhancedRetrievalSystem(self.config.retrieval_config)
            
            # Initialize query preprocessor
            self.query_preprocessor = EnhancedQueryPreprocessor(
                self.config.retrieval_config.keywords_map_path
            )
            
            # Initialize response generator
            if self.config.enable_llm_generation:
                try:
                    self.response_generator = EnhancedResponseGenerator(self.config.generation_config)
                    logger.info("Response generator initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize response generator: {e}")
                    if not self.config.fallback_to_simple:
                        raise
                    else:
                        logger.info("Continuing with simple response generation")
                        self.response_generator = None
            else:
                logger.info("LLM generation disabled, using simple response formatting")
                self.response_generator = None
            
            # Setup analytics logging
            if self.config.enable_analytics:
                self._setup_analytics_logging()
            
            logger.info("Service components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing service: {e}")
            raise
    
    def _setup_analytics_logging(self):
        """Setup analytics logging directory and files."""
        analytics_path = Path(self.config.analytics_output)
        analytics_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Log service startup
        startup_event = {
            "event_type": "service_startup",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_results": self.config.max_results_display,
                "retrieval_top_k": self.config.retrieval_config.top_k,
                "enable_sessions": self.config.enable_sessions
            }
        }
        
        self._log_analytics_event(startup_event)
    
    def _log_analytics_event(self, event: Dict[str, Any]):
        """Log analytics event to file."""
        if not self.config.enable_analytics:
            return
        
        try:
            with open(self.config.analytics_output, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"Failed to log analytics event: {e}")
    
    def create_session(self) -> str:
        """
        Create a new chat session.
        
        Returns:
            str: Session ID
        """
        if not self.config.enable_sessions:
            return "default"
        
        session_id = f"session_{int(time.time() * 1000)}"
        
        self.sessions[session_id] = ChatSession(
            session_id=session_id,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        logger.info(f"Created new session: {session_id}")
        
        # Log session creation
        if self.config.enable_analytics:
            self._log_analytics_event({
                "event_type": "session_created",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get session by ID with timeout check.
        
        Args:
            session_id (str): Session ID
            
        Returns:
            Optional[ChatSession]: Session object or None if expired
        """
        if not self.config.enable_sessions or session_id == "default":
            return None
        
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        # Check timeout
        time_since_activity = datetime.now() - session.last_activity
        if time_since_activity.total_seconds() > (self.config.session_timeout_minutes * 60):
            logger.info(f"Session {session_id} expired")
            self.sessions.pop(session_id, None)
            return None
        
        return session
    
    def update_session_activity(self, session_id: str):
        """Update session last activity timestamp."""
        session = self.get_session(session_id)
        if session:
            session.last_activity = datetime.now()
    
    def process_query(self, query: str, session_id: str = None, 
                     max_results: int = None) -> ChatResponse:
        """
        Process a user query and return formatted response.
        
        Args:
            query (str): User query
            session_id (str, optional): Session ID
            max_results (int, optional): Maximum results to return
            
        Returns:
            ChatResponse: Formatted response
        """
        start_time = time.time()
        max_results = max_results or self.config.max_results_display
        
        try:
            logger.info(f"Processing query: '{query}' (session: {session_id})")
            
            # Update session activity
            if session_id:
                self.update_session_activity(session_id)
            
            # Validate query
            if not query or not query.strip():
                return ChatResponse(
                    success=False,
                    message="Silakan masukkan pertanyaan yang valid.",
                    session_id=session_id
                )
            
            # Analyze query intent
            query_analysis = analyze_query_intent(query)
            
            # Perform retrieval
            retrieval_results = self.retrieval_system.retrieve(query, top_k=max_results * 2)
            
            # Filter results by confidence
            filtered_results = [
                result for result in retrieval_results 
                if result.score >= self.config.min_confidence_threshold
            ]
            
            # Take top results
            top_results = filtered_results[:max_results]
            
            # Generate response message
            if self.response_generator and self.config.enable_llm_generation:
                try:
                    # Use async generation if available
                    response_message = self._generate_llm_response_sync(
                        query, top_results, query_analysis
                    )
                except Exception as e:
                    logger.warning(f"LLM generation failed: {e}")
                    if self.config.fallback_to_simple:
                        response_message = self._generate_simple_response_message(
                            query, top_results, query_analysis
                        )
                    else:
                        raise
            else:
                response_message = self._generate_simple_response_message(
                    query, top_results, query_analysis
                )
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            
            # Update session
            if session_id:
                session = self.get_session(session_id)
                if session:
                    session.query_count += 1
                    session.total_results_returned += len(top_results)
                    
                    # Add to context history
                    context_entry = {
                        "query": query,
                        "results_count": len(top_results),
                        "timestamp": datetime.now().isoformat(),
                        "query_analysis": query_analysis
                    }
                    
                    session.context_history.append(context_entry)
                    
                    # Limit context history size
                    if len(session.context_history) > self.config.max_context_history:
                        session.context_history = session.context_history[-self.config.max_context_history:]
            
            # Log analytics
            if self.config.enable_analytics:
                self._log_query_analytics(
                    query, query_analysis, top_results, response_time, session_id
                )
            
            # Create response
            response = ChatResponse(
                success=True,
                message=response_message,
                results=top_results,
                query_analysis=query_analysis,
                response_time_ms=response_time,
                session_id=session_id,
                metadata={
                    "total_candidates": len(retrieval_results),
                    "filtered_candidates": len(filtered_results),
                    "query_length": len(query.split()),
                    "avg_result_score": sum(r.score for r in top_results) / len(top_results) if top_results else 0
                }
            )
            
            logger.info(f"Query processed successfully: {len(top_results)} results in {response_time:.1f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            response_time = (time.time() - start_time) * 1000
            
            return ChatResponse(
                success=False,
                message=self.config.error_message,
                response_time_ms=response_time,
                session_id=session_id,
                metadata={"error": str(e)}
            )
    
    def _generate_llm_response_sync(self, query: str, results: List[RetrievalResult], 
                                   query_analysis: Dict[str, Any]) -> str:
        """
        Generate response using LLM (synchronous wrapper for async method).
        
        Args:
            query (str): Original query
            results (List[RetrievalResult]): Retrieval results
            query_analysis (Dict): Query analysis
            
        Returns:
            str: Generated response message
        """
        try:
            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, fallback to simple response
                    logger.info("Event loop already running, falling back to simple response generation")
                    return self._generate_simple_response_message(query, results, query_analysis)
            except RuntimeError:
                # No event loop exists, create a new one
                pass
            
            # Create new event loop for sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                generation_result = loop.run_until_complete(
                    self.response_generator.generate_response(query, results, query_analysis)
                )
                
                if generation_result.success:
                    return generation_result.content
                else:
                    logger.warning(f"LLM generation failed: {generation_result.error_message}")
                    raise Exception(generation_result.error_message)
            finally:
                loop.close()
                asyncio.set_event_loop(None)
                
        except Exception as e:
            logger.error(f"Error in LLM response generation: {e}")
            raise
    
    async def generate_response_async(self, query: str, results: List[RetrievalResult], 
                                    query_analysis: Dict[str, Any] = None) -> GenerationResult:
        """
        Generate response using LLM (async version).
        
        Args:
            query (str): Original query
            results (List[RetrievalResult]): Retrieval results
            query_analysis (Dict): Query analysis
            
        Returns:
            GenerationResult: Generation result with metadata
        """
        if not self.response_generator:
            raise Exception("Response generator not initialized")
        
        return await self.response_generator.generate_response(query, results, query_analysis or {})
    
    def _generate_simple_response_message(self, query: str, results: List[RetrievalResult], 
                                        query_analysis: Dict[str, Any]) -> str:
        """
        Generate simple human-friendly response message without LLM.
        
        Args:
            query (str): Original query
            results (List[RetrievalResult]): Retrieval results
            query_analysis (Dict): Query analysis
            
        Returns:
            str: Response message
        """
        if not results:
            return self.config.no_results_message
        
        # Determine response tone based on query analysis
        has_question = query_analysis.get('has_question', False)
        has_action = query_analysis.get('has_action_intent', False)
        islamic_strength = query_analysis.get('islamic_context_strength', 0)
        
        # Build response message
        message_parts = []
        
        # Greeting and context
        if islamic_strength > 0.5:
            message_parts.append("Berikut hadits yang sesuai dengan pertanyaan Anda:")
        elif has_question:
            message_parts.append("Saya menemukan beberapa hadits yang menjawab pertanyaan Anda:")
        elif has_action:
            message_parts.append("Berikut informasi dari hadits yang relevan:")
        else:
            message_parts.append("Hadits yang sesuai dengan pencarian Anda:")
        
        # Add result count
        if len(results) == 1:
            message_parts.append(f"\nDitemukan 1 hadits:")
        else:
            message_parts.append(f"\nDitemukan {len(results)} hadits:")
        
        # Format top results with detailed hadith information
        for i, result in enumerate(results[:self.config.max_results_display], 1):
            doc = result.document
            
            # Format each hadith clearly
            message_parts.append(f"\n**Hadits {i}:**")
            message_parts.append(f"- **Kitab**: {doc.get('kitab', 'Unknown')}")
            message_parts.append(f"- **ID**: {doc.get('id', 'Unknown')}")
            
            # Add Arabic text if available
            arab_text = doc.get('arab', '')
            if arab_text:
                if len(arab_text) > 300:
                    arab_text = arab_text[:300] + "... (teks diperpendek)"
                message_parts.append(f"- **Arab**: {arab_text}")
            
            # Add translation
            terjemah = doc.get('terjemah', '')
            if len(terjemah) > self.config.result_text_preview_length:
                terjemah = terjemah[:self.config.result_text_preview_length] + "..."
            message_parts.append(f"- **Terjemah**: {terjemah}")
            
            # Add keywords if available
            if result.matched_keywords:
                keywords_str = ", ".join(result.matched_keywords[:5])
                message_parts.append(f"- **Kata kunci**: {keywords_str}")
            
            # Add confidence indicator for high-confidence results
            if result.score >= self.config.high_confidence_threshold:
                message_parts.append("- ✅ **Hasil dengan kepercayaan tinggi**")
            
            message_parts.append("")  # Empty line between hadits
        
        # Add helpful note
        if len(results) > self.config.max_results_display:
            remaining = len(results) - self.config.max_results_display
            message_parts.append(f"📝 **Catatan**: Masih ada {remaining} hadits lainnya yang relevan.")
            message_parts.append("")
        
        # Add summary and follow-up
        message_parts.append("**Ringkasan**: Hadits-hadits di atas memberikan panduan tentang topik yang Anda tanyakan.")
        message_parts.append("")
        
        if islamic_strength > 0.3:
            message_parts.append("🤲 Semoga bermanfaat untuk memahami ajaran Islam.")
        
        message_parts.append("**Pertanyaan lanjut**: Apakah Anda ingin penjelasan lebih detail atau mencari hadits dengan topik lain?")
        
        return "\n".join(message_parts)
    
    async def process_query_async(self, query: str, session_id: str = None, 
                                max_results: int = None) -> ChatResponse:
        """
        Process a user query asynchronously with LLM generation.
        
        Args:
            query (str): User query
            session_id (str, optional): Session ID
            max_results (int, optional): Maximum results to return
            
        Returns:
            ChatResponse: Formatted response
        """
        start_time = time.time()
        max_results = max_results or self.config.max_results_display
        
        try:
            logger.info(f"Processing async query: '{query}' (session: {session_id})")
            
            # Update session activity
            if session_id:
                self.update_session_activity(session_id)
            
            # Validate query
            if not query or not query.strip():
                return ChatResponse(
                    success=False,
                    message="Silakan masukkan pertanyaan yang valid.",
                    session_id=session_id
                )
            
            # Analyze query intent
            query_analysis = analyze_query_intent(query)
            
            # Perform retrieval
            retrieval_results = self.retrieval_system.retrieve(query, top_k=max_results * 2)
            
            # Filter results by confidence
            filtered_results = [
                result for result in retrieval_results 
                if result.score >= self.config.min_confidence_threshold
            ]
            
            # Take top results
            top_results = filtered_results[:max_results]
            
            # Generate response message using LLM
            if self.response_generator and self.config.enable_llm_generation:
                try:
                    generation_result = await self.generate_response_async(query, top_results, query_analysis)
                    response_message = generation_result.content if generation_result.success else self.config.error_message
                except Exception as e:
                    logger.warning(f"LLM generation failed: {e}")
                    if self.config.fallback_to_simple:
                        response_message = self._generate_simple_response_message(
                            query, top_results, query_analysis
                        )
                    else:
                        response_message = self.config.error_message
            else:
                response_message = self._generate_simple_response_message(
                    query, top_results, query_analysis
                )
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            
            # Update session
            if session_id:
                session = self.get_session(session_id)
                if session:
                    session.query_count += 1
                    session.total_results_returned += len(top_results)
                    
                    # Add to context history
                    context_entry = {
                        "query": query,
                        "results_count": len(top_results),
                        "timestamp": datetime.now().isoformat(),
                        "query_analysis": query_analysis,
                        "generation_used": self.response_generator is not None
                    }
                    
                    session.context_history.append(context_entry)
                    
                    # Limit context history size
                    if len(session.context_history) > self.config.max_context_history:
                        session.context_history = session.context_history[-self.config.max_context_history:]
            
            # Log analytics
            if self.config.enable_analytics:
                self._log_query_analytics(
                    query, query_analysis, top_results, response_time, session_id
                )
            
            # Create response
            response = ChatResponse(
                success=True,
                message=response_message,
                results=top_results,
                query_analysis=query_analysis,
                response_time_ms=response_time,
                session_id=session_id,
                metadata={
                    "total_candidates": len(retrieval_results),
                    "filtered_candidates": len(filtered_results),
                    "query_length": len(query.split()),
                    "avg_result_score": sum(r.score for r in top_results) / len(top_results) if top_results else 0,
                    "llm_generation_used": self.response_generator is not None and self.config.enable_llm_generation
                }
            )
            
            logger.info(f"Async query processed successfully: {len(top_results)} results in {response_time:.1f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error processing async query: {e}")
            
            response_time = (time.time() - start_time) * 1000
            
            return ChatResponse(
                success=False,
                message=self.config.error_message,
                response_time_ms=response_time,
                session_id=session_id,
                metadata={"error": str(e)}
            )
    
    async def generate_streaming_response(self, query: str, session_id: str = None, 
                                        max_results: int = None):
        """
        Generate streaming response for real-time display.
        
        Args:
            query (str): User query
            session_id (str, optional): Session ID
            max_results (int, optional): Maximum results to return
            
        Yields:
            str: Response chunks
        """
        try:
            # First get retrieval results
            max_results = max_results or self.config.max_results_display
            
            # Analyze query intent
            query_analysis = analyze_query_intent(query)
            
            # Perform retrieval
            retrieval_results = self.retrieval_system.retrieve(query, top_k=max_results * 2)
            
            # Filter results by confidence
            filtered_results = [
                result for result in retrieval_results 
                if result.score >= self.config.min_confidence_threshold
            ]
            
            # Take top results
            top_results = filtered_results[:max_results]
            
            # Generate streaming response
            if self.response_generator and self.config.enable_llm_generation:
                try:
                    async for chunk in self.response_generator.generate_streaming_response(query, top_results):
                        yield chunk
                except Exception as e:
                    logger.warning(f"Streaming generation failed: {e}")
                    # Fallback to simple response
                    simple_response = self._generate_simple_response_message(query, top_results, query_analysis)
                    yield simple_response
            else:
                # Use simple response for streaming
                simple_response = self._generate_simple_response_message(query, top_results, query_analysis)
                
                # Stream it word by word
                words = simple_response.split()
                current_chunk = ""
                
                for word in words:
                    current_chunk += word + " "
                    
                    if len(current_chunk) > 50 or word.endswith(('.', '!', '?', ':')):
                        yield current_chunk.strip()
                        current_chunk = ""
                        await asyncio.sleep(0.05)  # Small delay for streaming effect
                
                if current_chunk.strip():
                    yield current_chunk.strip()
                    
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"Terjadi kesalahan: {str(e)}"
    
    def _log_query_analytics(self, query: str, query_analysis: Dict[str, Any], 
                           results: List[RetrievalResult], response_time: float, 
                           session_id: str):
        """Log detailed query analytics."""
        analytics_event = {
            "event_type": "query_processed",
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "query": {
                "original": query,
                "length": len(query.split()),
                "has_question": query_analysis.get('has_question', False),
                "has_action": query_analysis.get('has_action_intent', False),
                "islamic_context_strength": query_analysis.get('islamic_context_strength', 0),
                "key_terms": query_analysis.get('key_terms', [])
            },
            "results": {
                "count": len(results),
                "avg_score": sum(r.score for r in results) / len(results) if results else 0,
                "max_score": max(r.score for r in results) if results else 0,
                "min_score": min(r.score for r in results) if results else 0,
                "total_matched_keywords": sum(len(r.matched_keywords) for r in results)
            },
            "performance": {
                "response_time_ms": response_time
            }
        }
        
        self._log_analytics_event(analytics_event)
    
    def get_greeting(self, session_id: str = None) -> ChatResponse:
        """
        Get greeting message for new conversations.
        
        Args:
            session_id (str, optional): Session ID
            
        Returns:
            ChatResponse: Greeting response
        """
        return ChatResponse(
            success=True,
            message=self.config.greeting_message,
            session_id=session_id,
            metadata={"event": "greeting"}
        )
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id (str): Session ID
            
        Returns:
            Dict[str, Any]: Session statistics
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found or expired"}
        
        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "query_count": session.query_count,
            "total_results_returned": session.total_results_returned,
            "avg_results_per_query": session.total_results_returned / max(session.query_count, 1),
            "context_history_size": len(session.context_history),
            "last_activity": session.last_activity.isoformat()
        }
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Get service health and status information.
        
        Returns:
            Dict[str, Any]: Service health status
        """
        try:
            # Test retrieval system
            test_results = self.retrieval_system.retrieve("test", top_k=1)
            retrieval_healthy = len(test_results) >= 0  # Even 0 results is OK for health check
            
            # Count active sessions
            now = datetime.now()
            active_sessions = 0
            for session in self.sessions.values():
                time_since_activity = now - session.last_activity
                if time_since_activity.total_seconds() <= (self.config.session_timeout_minutes * 60):
                    active_sessions += 1
            
            return {
                "status": "healthy" if retrieval_healthy else "degraded",
                "timestamp": now.isoformat(),
                "components": {
                    "retrieval_system": "healthy" if retrieval_healthy else "error",
                    "query_preprocessor": "healthy",
                    "session_management": "healthy"
                },
                "statistics": {
                    "active_sessions": active_sessions,
                    "total_sessions_created": len(self.sessions),
                    "analytics_enabled": self.config.enable_analytics
                },
                "configuration": {
                    "max_results_display": self.config.max_results_display,
                    "session_timeout_minutes": self.config.session_timeout_minutes,
                    "min_confidence_threshold": self.config.min_confidence_threshold
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }


# Convenience functions for easy integration
def create_hadith_ai_service(config: ServiceConfig = None) -> HadithAIService:
    """
    Create and initialize Hadith AI service.
    
    Args:
        config (ServiceConfig, optional): Service configuration
        
    Returns:
        HadithAIService: Initialized service
    """
    return HadithAIService(config)


def quick_query(query: str, config: ServiceConfig = None) -> ChatResponse:
    """
    Quick query without session management.
    
    Args:
        query (str): User query
        config (ServiceConfig, optional): Service configuration
        
    Returns:
        ChatResponse: Response
    """
    service = HadithAIService(config)
    return service.process_query(query)


# Main execution for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hadith AI Service - Fixed V1")
    parser.add_argument("--query", help="Test query")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum results")
    parser.add_argument("--show-scores", action="store_true", help="Show detailed scores")
    parser.add_argument("--session", action="store_true", help="Use session management")
    parser.add_argument("--health", action="store_true", help="Check service health")
    
    args = parser.parse_args()
    
    try:
        # Create service configuration
        config = ServiceConfig(
            max_results_display=args.max_results,
            include_scores=args.show_scores,
            enable_sessions=args.session
        )
        
        # Initialize service
        logger.info("Initializing Hadith AI Service...")
        service = HadithAIService(config)
        
        if args.health:
            # Health check
            health = service.get_service_health()
            print("Service Health:")
            print(json.dumps(health, indent=2, ensure_ascii=False))
        
        elif args.query:
            # Process query
            session_id = service.create_session() if args.session else None
            
            # Get greeting
            greeting = service.get_greeting(session_id)
            print(f"🤖 {greeting.message}\n")
            
            # Process query
            response = service.process_query(args.query, session_id, args.max_results)
            
            print(f"Query: {args.query}")
            print(f"Response: {response.message}")
            print(f"Success: {response.success}")
            print(f"Response time: {response.response_time_ms:.1f}ms")
            print(f"Results found: {len(response.results)}")
            
            if response.query_analysis:
                print(f"Query analysis: {response.query_analysis}")
            
            if args.session and session_id:
                stats = service.get_session_stats(session_id)
                print(f"Session stats: {stats}")
        
        else:
            print("Use --query to test a query or --health to check service health")
        
    except Exception as e:
        logger.error(f"Error running Hadith AI Service: {e}")
        sys.exit(1)
