"""
Hadith AI API Server - Fixed V1
==============================

Flask-based REST API server for Hadith AI service.
Provides easy integration for web applications and mobile apps.

Features:
- RESTful API endpoints
- Session management
- CORS support
- Request validation
- Error handling
- Health monitoring

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, AsyncGenerator
from flask import Flask, request, jsonify, Response, make_response
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, InternalServerError

# Import our service
try:
    from .hadith_ai_service import HadithAIService, ServiceConfig, ChatResponse
except ImportError:
    from hadith_ai_service import HadithAIService, ServiceConfig, ChatResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instance
hadith_service = None


def validate_request_data(required_fields: list, data: dict) -> Optional[str]:
    """
    Validate request data for required fields.
    
    Args:
        required_fields (list): List of required field names
        data (dict): Request data
        
    Returns:
        Optional[str]: Error message if validation fails, None if successful
    """
    missing_fields = [field for field in required_fields if field not in data or not data[field]]
    
    if missing_fields:
        return f"Missing required fields: {', '.join(missing_fields)}"
    
    return None


def format_api_response(success: bool, data: Any = None, message: str = "", 
                       error: str = "", status_code: int = 200) -> tuple:
    """
    Format standardized API response.
    
    Args:
        success (bool): Success status
        data (Any): Response data
        message (str): Success message
        error (str): Error message
        status_code (int): HTTP status code
        
    Returns:
        tuple: (response_dict, status_code)
    """
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat(),
        "data": data,
        "message": message,
        "error": error
    }
    
    return jsonify(response), status_code


def convert_chat_response_to_dict(chat_response: ChatResponse) -> Dict[str, Any]:
    """
    Convert ChatResponse object to dictionary for JSON serialization.
    
    Args:
        chat_response (ChatResponse): Chat response object
        
    Returns:
        Dict[str, Any]: Serializable dictionary
    """
    # Convert results to dictionaries
    results_data = []
    for result in chat_response.results:
        result_dict = {
            "document_id": result.document_id,
            "document": result.document,
            "score": result.score,
            "semantic_score": result.semantic_score,
            "keyword_score": result.keyword_score,
            "literal_overlap_score": result.literal_overlap_score,
            "matched_keywords": result.matched_keywords,
            "rank": result.rank
        }
        results_data.append(result_dict)
    
    return {
        "success": chat_response.success,
        "message": chat_response.message,
        "results": results_data,
        "query_analysis": chat_response.query_analysis,
        "response_time_ms": chat_response.response_time_ms,
        "session_id": chat_response.session_id,
        "metadata": chat_response.metadata
    }


def create_app(service_config: ServiceConfig = None) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        service_config (ServiceConfig, optional): Service configuration
        
    Returns:
        Flask: Configured Flask app
    """
    app = Flask(__name__)
    app.secret_key = 'hadith_ai_secret_key_v1'  # Change in production
    
    # Enable CORS for cross-origin requests - permissive for frontend
    CORS(app, 
         supports_credentials=True,
         origins=["*"],  # Allow all origins for development
         allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    
    # Initialize service
    global hadith_service
    try:
        logger.info("Initializing Hadith AI Service...")
        hadith_service = HadithAIService(service_config)
        logger.info("Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")
        raise
    
    # API Routes
    # Handle preflight requests
    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            response = make_response()
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add('Access-Control-Allow-Headers', "*")
            response.headers.add('Access-Control-Allow-Methods', "*")
            return response

    @app.route('/', methods=['GET'])
    def home():
        """Home endpoint with API information."""
        return format_api_response(
            success=True,
            data={
                "service": "Hadith AI API - Fixed V1",
                "version": "1.0.0",
                "description": "Enhanced Hadith retrieval and question answering API with LLM generation",
                "endpoints": {
                    "GET /": "API information",
                    "GET /health": "Service health check",
                    "GET /sessions": "Get all chat sessions",
                    "POST /sessions": "Create new chat session",
                    "GET /sessions/{id}": "Get specific session with messages",
                    "PUT /sessions/{id}": "Update session title",
                    "DELETE /sessions/{id}": "Delete session",
                    "POST /sessions/{id}/ask": "Ask question in session (SSE streaming)",
                    "POST /ask": "Ask question without session (SSE streaming)",
                    "POST /chat": "Process chat query (basic)",
                    "POST /chat/async": "Process chat query with enhanced LLM generation",
                    "POST /chat/stream": "Process chat query with streaming response",
                    "GET /session/{session_id}/stats": "Get session statistics",
                    "POST /query": "Simple query without session",
                    "GET /greeting": "Get greeting message"
                },
                "features": [
                    "Enhanced keyword extraction",
                    "Semantic retrieval with FAISS",
                    "LLM response generation",
                    "Streaming responses",
                    "Session management",
                    "Multi-factor scoring"
                ]
            },
            message="Hadith AI API is running"
        )

    @app.route('/health', methods=['GET', 'OPTIONS'])
    def health_check():
        """Health check endpoint."""
        try:
            health_data = hadith_service.get_service_health()
            
            if health_data['status'] == 'healthy':
                return format_api_response(
                    success=True,
                    data=health_data,
                    message="Service is healthy"
                )
            else:
                return format_api_response(
                    success=False,
                    data=health_data,
                    error="Service is not fully healthy",
                    status_code=503
                )
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return format_api_response(
                success=False,
                error=f"Health check failed: {str(e)}",
                status_code=500
            )

    @app.route('/session/create', methods=['POST', 'OPTIONS'])
    def create_session():
        """Create a new chat session."""
        try:
            session_id = hadith_service.create_session()
            
            return format_api_response(
                success=True,
                data={
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat()
                },
                message="Session created successfully"
            )
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            return format_api_response(
                success=False,
                error=f"Failed to create session: {str(e)}",
                status_code=500
            )

    @app.route('/chat', methods=['POST', 'OPTIONS'])
    def chat():
        """
        Process a chat query with session management.
        
        Expected JSON payload:
        {
            "query": "Your question here",
            "session_id": "optional_session_id",
            "max_results": 5
        }
        """
        try:
            # Validate request
            if not request.is_json:
                return format_api_response(
                    success=False,
                    error="Request must be JSON",
                    status_code=400
                )
            
            data = request.get_json()
            
            # Validate required fields
            validation_error = validate_request_data(['query'], data)
            if validation_error:
                return format_api_response(
                    success=False,
                    error=validation_error,
                    status_code=400
                )
            
            query = data['query'].strip()
            session_id = data.get('session_id')
            max_results = data.get('max_results', 5)
            
            # Validate max_results
            if not isinstance(max_results, int) or max_results < 1 or max_results > 20:
                return format_api_response(
                    success=False,
                    error="max_results must be an integer between 1 and 20",
                    status_code=400
                )
            
            # Process query
            chat_response = hadith_service.process_query(
                query=query,
                session_id=session_id,
                max_results=max_results
            )
            
            # Convert to serializable format
            response_data = convert_chat_response_to_dict(chat_response)
            
            if chat_response.success:
                return format_api_response(
                    success=True,
                    data=response_data,
                    message="Query processed successfully"
                )
            else:
                return format_api_response(
                    success=False,
                    data=response_data,
                    error="Query processing failed",
                    status_code=400
                )
            
        except BadRequest as e:
            return format_api_response(
                success=False,
                error=f"Bad request: {str(e)}",
                status_code=400
            )
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return format_api_response(
                success=False,
                error=f"Internal server error: {str(e)}",
                status_code=500
            )

    @app.route('/query', methods=['POST'])
    def simple_query():
        """
        Simple query endpoint without session management.
        
        Expected JSON payload:
        {
            "query": "Your question here",
            "max_results": 5
        }
        """
        try:
            # Validate request
            if not request.is_json:
                return format_api_response(
                    success=False,
                    error="Request must be JSON",
                    status_code=400
                )
            
            data = request.get_json()
            
            # Validate required fields
            validation_error = validate_request_data(['query'], data)
            if validation_error:
                return format_api_response(
                    success=False,
                    error=validation_error,
                    status_code=400
                )
            
            query = data['query'].strip()
            max_results = data.get('max_results', 5)
            
            # Validate max_results
            if not isinstance(max_results, int) or max_results < 1 or max_results > 20:
                return format_api_response(
                    success=False,
                    error="max_results must be an integer between 1 and 20",
                    status_code=400
                )
            
            # Process query without session
            chat_response = hadith_service.process_query(
                query=query,
                max_results=max_results
            )
            
            # Convert to serializable format
            response_data = convert_chat_response_to_dict(chat_response)
            
            if chat_response.success:
                return format_api_response(
                    success=True,
                    data=response_data,
                    message="Query processed successfully"
                )
            else:
                return format_api_response(
                    success=False,
                    data=response_data,
                    error="Query processing failed",
                    status_code=400
                )
            
        except Exception as e:
            logger.error(f"Simple query failed: {e}")
            return format_api_response(
                success=False,
                error=f"Internal server error: {str(e)}",
                status_code=500
            )

    @app.route('/session/<session_id>/stats', methods=['GET'])
    def get_session_stats(session_id: str):
        """Get statistics for a specific session."""
        try:
            stats = hadith_service.get_session_stats(session_id)
            
            if "error" in stats:
                return format_api_response(
                    success=False,
                    error=stats["error"],
                    status_code=404
                )
            
            return format_api_response(
                success=True,
                data=stats,
                message="Session statistics retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Session stats retrieval failed: {e}")
            return format_api_response(
                success=False,
                error=f"Failed to retrieve session stats: {str(e)}",
                status_code=500
            )

    @app.route('/chat/stream', methods=['POST', 'OPTIONS'])
    def chat_stream():
        """
        Process a chat query with streaming response.
        
        Expected JSON payload:
        {
            "query": "Your question here",
            "session_id": "optional_session_id",
            "max_results": 5
        }
        """
        try:
            # Validate request
            if not request.is_json:
                return format_api_response(
                    success=False,
                    error="Request must be JSON",
                    status_code=400
                )
            
            data = request.get_json()
            
            # Validate required fields
            validation_error = validate_request_data(['query'], data)
            if validation_error:
                return format_api_response(
                    success=False,
                    error=validation_error,
                    status_code=400
                )
            
            query = data['query'].strip()
            session_id = data.get('session_id')
            max_results = data.get('max_results', 5)
            
            # Validate max_results
            if not isinstance(max_results, int) or max_results < 1 or max_results > 20:
                return format_api_response(
                    success=False,
                    error="max_results must be an integer between 1 and 20",
                    status_code=400
                )
            
            def generate_stream():
                """Generate streaming response."""
                try:
                    # Create event loop for async operations
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        async def async_stream():
                            async for chunk in hadith_service.generate_streaming_response(
                                query, session_id, max_results
                            ):
                                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
                            
                            yield f"data: {json.dumps({'event': 'complete'}, ensure_ascii=False)}\n\n"
                        
                        # Run async generator
                        async_gen = async_stream()
                        
                        while True:
                            try:
                                chunk = loop.run_until_complete(async_gen.__anext__())
                                yield chunk
                            except StopAsyncIteration:
                                break
                                
                    finally:
                        loop.close()
                        
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    error_data = json.dumps({
                        'error': f"Streaming error: {str(e)}"
                    }, ensure_ascii=False)
                    yield f"data: {error_data}\n\n"
            
            return Response(
                generate_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
            
        except Exception as e:
            logger.error(f"Chat streaming failed: {e}")
            return format_api_response(
                success=False,
                error=f"Internal server error: {str(e)}",
                status_code=500
            )
    
    @app.route('/chat/async', methods=['POST', 'OPTIONS'])
    def chat_async():
        """
        Process a chat query asynchronously with enhanced LLM generation.
        
        Expected JSON payload:
        {
            "query": "Your question here",
            "session_id": "optional_session_id",
            "max_results": 5
        }
        """
        try:
            # Validate request
            if not request.is_json:
                return format_api_response(
                    success=False,
                    error="Request must be JSON",
                    status_code=400
                )
            
            data = request.get_json()
            
            # Validate required fields
            validation_error = validate_request_data(['query'], data)
            if validation_error:
                return format_api_response(
                    success=False,
                    error=validation_error,
                    status_code=400
                )
            
            query = data['query'].strip()
            session_id = data.get('session_id')
            max_results = data.get('max_results', 5)
            
            # Validate max_results
            if not isinstance(max_results, int) or max_results < 1 or max_results > 20:
                return format_api_response(
                    success=False,
                    error="max_results must be an integer between 1 and 20",
                    status_code=400
                )
            
            # Process query asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                chat_response = loop.run_until_complete(
                    hadith_service.process_query_async(query, session_id, max_results)
                )
            finally:
                loop.close()
            
            # Convert to serializable format
            response_data = convert_chat_response_to_dict(chat_response)
            
            if chat_response.success:
                return format_api_response(
                    success=True,
                    data=response_data,
                    message="Query processed successfully with enhanced generation"
                )
            else:
                return format_api_response(
                    success=False,
                    data=response_data,
                    error="Query processing failed",
                    status_code=400
                )
            
        except Exception as e:
            logger.error(f"Async chat processing failed: {e}")
            return format_api_response(
                success=False,
                error=f"Internal server error: {str(e)}",
                status_code=500
            )

    @app.route('/sessions', methods=['GET', 'OPTIONS'])
    def get_sessions():
        """Get all chat sessions (compatible with main.py frontend)."""
        try:
            # Get all sessions from service (we'll need to implement this in service)
            sessions = hadith_service.get_all_sessions()
            
            return format_api_response(
                success=True,
                data={
                    "sessions": sessions,
                    "total": len(sessions)
                },
                message="Sessions retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Sessions retrieval failed: {e}")
            return format_api_response(
                success=False,
                error=f"Failed to retrieve sessions: {str(e)}",
                status_code=500
            )

    @app.route('/sessions', methods=['POST', 'OPTIONS'])
    def create_session_compatible():
        """Create a new chat session (compatible with main.py frontend)."""
        try:
            data = request.get_json() if request.is_json else {}
            title = data.get('title') if data else None
            
            session_id = hadith_service.create_session()
            
            # Get session details for compatibility
            session_info = hadith_service.get_session_info(session_id)
            
            return format_api_response(
                success=True,
                data={
                    "session_id": session_id,
                    "title": session_info.get('title', f'Chat Session'),
                    "created_at": session_info.get('created_at', datetime.now().isoformat())
                },
                message="Session created successfully"
            )
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            return format_api_response(
                success=False,
                error=f"Failed to create session: {str(e)}",
                status_code=500
            )

    @app.route('/sessions/<session_id>', methods=['GET', 'OPTIONS'])
    def get_session_details(session_id: str):
        """Get a specific session with all messages (compatible with main.py frontend)."""
        try:
            session_details = hadith_service.get_session_details(session_id)
            
            if not session_details:
                return format_api_response(
                    success=False,
                    error="Session not found",
                    status_code=404
                )
            
            return format_api_response(
                success=True,
                data=session_details,
                message="Session details retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Session details retrieval failed: {e}")
            return format_api_response(
                success=False,
                error=f"Failed to retrieve session details: {str(e)}",
                status_code=500
            )

    @app.route('/sessions/<session_id>', methods=['PUT', 'OPTIONS'])
    def update_session_title(session_id: str):
        """Update session title (compatible with main.py frontend)."""
        try:
            if not request.is_json:
                return format_api_response(
                    success=False,
                    error="Request must be JSON",
                    status_code=400
                )
            
            data = request.get_json()
            title = data.get('title')
            
            if not title:
                return format_api_response(
                    success=False,
                    error="Title is required",
                    status_code=400
                )
            
            success = hadith_service.update_session_title(session_id, title)
            
            if not success:
                return format_api_response(
                    success=False,
                    error="Session not found",
                    status_code=404
                )
            
            return format_api_response(
                success=True,
                data={"message": "Session updated successfully"},
                message="Session updated successfully"
            )
            
        except Exception as e:
            logger.error(f"Session update failed: {e}")
            return format_api_response(
                success=False,
                error=f"Failed to update session: {str(e)}",
                status_code=500
            )

    @app.route('/sessions/<session_id>', methods=['DELETE', 'OPTIONS'])
    def delete_session(session_id: str):
        """Delete a chat session (compatible with main.py frontend)."""
        try:
            success = hadith_service.delete_session(session_id)
            
            if not success:
                return format_api_response(
                    success=False,
                    error="Session not found",
                    status_code=404
                )
            
            return format_api_response(
                success=True,
                data={"message": "Session deleted successfully"},
                message="Session deleted successfully"
            )
            
        except Exception as e:
            logger.error(f"Session deletion failed: {e}")
            return format_api_response(
                success=False,
                error=f"Failed to delete session: {str(e)}",
                status_code=500
            )

    @app.route('/sessions/<session_id>/ask', methods=['POST', 'OPTIONS'])
    def ask_question_in_session(session_id: str):
        """Ask a question within a specific chat session (compatible with main.py frontend)."""
        try:
            if not request.is_json:
                return format_api_response(
                    success=False,
                    error="Request must be JSON",
                    status_code=400
                )
            
            data = request.get_json()
            question = data.get('question', '').strip()
            
            if not question:
                return format_api_response(
                    success=False,
                    error="Question cannot be empty",
                    status_code=400
                )
            
            # Check if session exists
            session_exists = hadith_service.session_exists(session_id)
            if not session_exists:
                return format_api_response(
                    success=False,
                    error="Session not found",
                    status_code=404
                )
            
            # For streaming response compatible with main.py
            def generate_stream():
                """Generate streaming response compatible with FastAPI SSE."""
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        async def async_stream():
                            async for chunk in hadith_service.generate_streaming_response(
                                question, session_id
                            ):
                                # Format for FastAPI SSE compatibility
                                if isinstance(chunk, dict):
                                    event_type = chunk.get('type', 'message')
                                    data = chunk.get('data', '')
                                else:
                                    event_type = 'message'
                                    data = str(chunk)
                                
                                yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                            
                            yield f"event: complete\ndata: {json.dumps('Response generation completed', ensure_ascii=False)}\n\n"
                        
                        async_gen = async_stream()
                        
                        while True:
                            try:
                                chunk = loop.run_until_complete(async_gen.__anext__())
                                yield chunk
                            except StopAsyncIteration:
                                break
                                
                    finally:
                        loop.close()
                        
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"event: error\ndata: {json.dumps(f'Streaming error: {str(e)}', ensure_ascii=False)}\n\n"
            
            return Response(
                generate_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
            
        except Exception as e:
            logger.error(f"Session question processing failed: {e}")
            return format_api_response(
                success=False,
                error=f"Internal server error: {str(e)}",
                status_code=500
            )

    @app.route('/ask', methods=['POST', 'OPTIONS'])
    def ask_question():
        """Ask a question without session (compatible with main.py frontend)."""
        try:
            if not request.is_json:
                return format_api_response(
                    success=False,
                    error="Request must be JSON",
                    status_code=400
                )
            
            data = request.get_json()
            question = data.get('question', '').strip()
            
            if not question:
                return format_api_response(
                    success=False,
                    error="Question cannot be empty",
                    status_code=400
                )
            
            # For streaming response compatible with main.py
            def generate_stream():
                """Generate streaming response compatible with FastAPI SSE."""
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        async def async_stream():
                            async for chunk in hadith_service.generate_streaming_response(question):
                                # Format for FastAPI SSE compatibility
                                if isinstance(chunk, dict):
                                    event_type = chunk.get('type', 'message')
                                    data = chunk.get('data', '')
                                else:
                                    event_type = 'message'
                                    data = str(chunk)
                                
                                yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                            
                            yield f"event: complete\ndata: {json.dumps('Response generation completed', ensure_ascii=False)}\n\n"
                        
                        async_gen = async_stream()
                        
                        while True:
                            try:
                                chunk = loop.run_until_complete(async_gen.__anext__())
                                yield chunk
                            except StopAsyncIteration:
                                break
                                
                    finally:
                        loop.close()
                        
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"event: error\ndata: {json.dumps(f'Streaming error: {str(e)}', ensure_ascii=False)}\n\n"
            
            return Response(
                generate_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
            
        except Exception as e:
            logger.error(f"Question processing failed: {e}")
            return format_api_response(
                success=False,
                error=f"Internal server error: {str(e)}",
                status_code=500
            )

    @app.route('/greeting', methods=['GET'])
    def get_greeting():
        """Get greeting message."""
        try:
            session_id = request.args.get('session_id')
            greeting_response = hadith_service.get_greeting(session_id)
            
            response_data = convert_chat_response_to_dict(greeting_response)
            
            return format_api_response(
                success=True,
                data=response_data,
                message="Greeting retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Greeting retrieval failed: {e}")
            return format_api_response(
                success=False,
                error=f"Failed to retrieve greeting: {str(e)}",
                status_code=500
            )

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return format_api_response(
            success=False,
            error="Endpoint not found",
            status_code=404
        )

    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 errors."""
        return format_api_response(
            success=False,
            error="Method not allowed",
            status_code=405
        )

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        logger.error(f"Internal server error: {error}")
        return format_api_response(
            success=False,
            error="Internal server error",
            status_code=500
        )

    # Request logging middleware
    @app.before_request
    def log_request_info():
        """Log request information."""
        logger.info(f"{request.method} {request.url} - {request.remote_addr}")

    @app.after_request
    def log_response_info(response):
        """Log response information."""
        logger.info(f"Response: {response.status_code}")
        return response
    
    return app


# Development server
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hadith AI API Server - Fixed V1")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--max-results", type=int, default=5, help="Default max results")
    
    args = parser.parse_args()
    
    # Create service configuration
    service_config = ServiceConfig(
        max_results_display=args.max_results,
        enable_sessions=True,
        enable_analytics=True
    )
    
    try:
        # Create app with custom configuration
        app = create_app(service_config)
        
        logger.info(f"Starting Hadith AI API Server on {args.host}:{args.port}")
        logger.info(f"Debug mode: {args.debug}")
        
        # Run development server
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        exit(1)